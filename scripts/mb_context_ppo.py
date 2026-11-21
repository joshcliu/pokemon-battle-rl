"""Train state-dependent tactical weights with PPO over legal joint commands."""

import argparse
import functools

import numpy as np
import torch
from mb_mechanics_es import FEATURES, fitted, mechanics
from mb_policy_iteration_eval import OUT, RUN, SEEDS, distribution, verify
from mb_residual_preview import pilot, save
from mb_rollout_branches import pairing
from poke_env.battle import Field, SideCondition
from torch import nn
from torch.distributions import Categorical
from vgc_posttrain import Match, atomic_rows, batch_obs

from pokemon_rl.league import stat

CONTEXT = ["intercept", "turn", "material", "speed", "attack_pressure", "forced_switch"]
GROUPS = 16
UPDATES = 64
TEMP = 0.65


def context(view, features):
    own = [m for m in view.active_pokemon if m and not m.fainted]
    foes = [m for m in view.opponent_active_pokemon if m and not m.fainted]
    lost = [sum(m.fainted for m in team.values()) for team in (view.team, view.opponent_team)]
    my_tail = 1 + int(SideCondition.TAILWIND in view.side_conditions)
    opp_tail = 1 + int(SideCondition.TAILWIND in view.opponent_side_conditions)
    speeds = [
        float(stat(m, "spe") * my_tail > stat(f, "spe") * opp_tail) for m in own for f in foes
    ]
    speed = (2 * np.mean(speeds) - 1) if speeds else 0.0
    if Field.TRICK_ROOM in view.fields:
        speed = -speed
    return np.array(
        [
            1.0,
            np.tanh((view.turn - 4) / 4),
            (lost[1] - lost[0]) / 4,
            speed,
            np.clip(features[:, 0].max() - 0.5, -0.5, 1),
            any(view.force_switch),
        ],
        dtype=np.float32,
    )


class ContextActor(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.register_buffer("initial", torch.as_tensor(fitted(seed), dtype=torch.float32))
        self.residual = nn.Parameter(torch.zeros(len(CONTEXT), len(FEATURES)))
        self.critic = nn.Parameter(torch.zeros(len(CONTEXT)))

    def forward(self, obs):
        phi, prior, ctx, parent_v = obs
        weights = self.initial + ctx @ self.residual
        logits = (prior + (phi * weights[:, None]).sum(2)) / TEMP
        return Categorical(logits=logits), parent_v + ctx @ self.critic


def features(model, obs, view, *, need_value):
    actions, prior = distribution(model, obs)
    phi = mechanics(view, actions).astype(np.float32)
    ctx = context(view, phi)
    v = model.base.predict_values(obs).item() if need_value else 0.0
    return actions, (torch.from_numpy(phi), prior, torch.from_numpy(ctx), v)


def pack(rows):
    phi, prior, ctx, v = zip(*rows, strict=True)
    return (
        nn.utils.rnn.pad_sequence(phi, batch_first=True, padding_value=0.0),
        nn.utils.rnn.pad_sequence(prior, batch_first=True, padding_value=-torch.inf),
        torch.stack(ctx),
        torch.tensor(v, dtype=torch.float32),
    )


def folder(s):
    return RUN / "context-ppo" / f"seed-{s}"


@functools.lru_cache(maxsize=4)
def checkpoint(seed):
    path = folder(seed) / "progress.pt"
    if not path.exists():
        path = OUT / "weights" / f"context-ppo-seed-{seed}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["update"] == UPDATES and state["seed"] == seed
    head = ContextActor(seed)
    head.load_state_dict(state["model"])
    return head.eval().requires_grad_(False)


def deploy(model, obs, view, seed):
    choices, row = features(model, obs, view, need_value=False)
    dist, _ = checkpoint(seed)(pack([row]))
    return choices[int(dist.logits[0].argmax())].tolist()


def train(seed):
    model = pilot(seed)
    head = ContextActor(seed)
    opt = torch.optim.Adam(head.parameters(), lr=0.012, eps=1e-5)
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "progress.pt"
    start, logs = 0, []
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["seed"] == seed
        start, logs = saved["update"], saved["logs"]
        head.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
    match = Match()
    for u in range(start, UPDATES):
        states, acts, old, advantages, targets, records = [], [], [], [], [], []
        for j in range(GROUPS):
            g = u * GROUPS + j
            own, foe, side = pairing(seed, g)
            if g % 2:
                own, foe = foe, own
            sim = 848_000_000 + seed * 1_000_000 + g
            group = []
            for rep in (0, 1):
                match.reset(own, foe, sim, side, preview="native")
                torch.manual_seed(sim + rep * 50_000_000)
                episode = []
                done = False
                with torch.no_grad():
                    while not done:
                        choices, row = features(
                            model, batch_obs(match.obs()), match.view, need_value=True
                        )
                        dist, v = head(pack([row]))
                        idx = dist.sample()
                        value, lp = v.item(), dist.log_prob(idx).item()
                        done, reward = match.step(choices[int(idx.item())].tolist())
                        episode.append((row, idx.item(), value, lp))
                group.append((episode, reward))
            for r, (episode, reward) in enumerate(group):
                v = [frame[2] for frame in episode]
                grouped = reward - group[1 - r][1]
                gae = 0.0
                values = []
                for t in reversed(range(len(v))):
                    delta = (reward if t + 1 == len(v) else 0.99 * v[t + 1]) - v[t]
                    gae = delta + 0.99 * 0.95 * gae
                    values.append(
                        (0.7 * gae + 0.3 * grouped * 0.99 ** (len(v) - 1 - t), v[t] + gae)
                    )
                for (row, idx, _, lp), (adv, y) in zip(episode, reversed(values), strict=True):
                    states.append(row)
                    acts.append(idx)
                    old.append(lp)
                    advantages.append(adv)
                    targets.append(y)
            from pathlib import Path

            records.append(
                dict(
                    group=g,
                    own=Path(own).stem,
                    foe=Path(foe).stem,
                    sim=sim,
                    side=side,
                    outcomes=" ".join(str(x[1]) for x in group),
                )
            )
        a = torch.tensor(acts)
        old, adv, y = map(
            lambda v: torch.tensor(v, dtype=torch.float32), (old, advantages, targets)
        )
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        rng = torch.Generator().manual_seed(4245 + seed * 1000 + u)
        for _ in range(3):
            for ix in torch.randperm(len(a), generator=rng).split(64):
                obs = pack([states[i] for i in ix.tolist()])
                dist, v = head(obs)
                ratio = (dist.log_prob(a[ix]) - old[ix]).exp()
                pg = torch.minimum(ratio * adv[ix], ratio.clamp(0.8, 1.2) * adv[ix]).mean()
                loss = (
                    -pg
                    + 0.25 * (v - y[ix]).square().mean()
                    - 0.001 * dist.entropy().mean()
                    + 0.001 * head.residual.square().mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
        n = (logs[-1]["decisions"] if logs else 0) + len(acts)
        logs.append(
            dict(
                update=u + 1,
                decisions=n,
                games=(u + 1) * 2 * GROUPS,
                loss=float(loss.detach()),
                wins=sum(r["outcomes"].split().count("1") for r in records),
            )
        )
        atomic_rows(dest / f"groups-{u + 1:03}.csv", records)
        save(
            path,
            dict(
                model=head.state_dict(),
                optimizer=opt.state_dict(),
                seed=seed,
                update=u + 1,
                logs=logs,
            ),
        )
        atomic_rows(OUT / "training" / f"context-ppo-{seed}.csv", logs)
        if (u + 1) % 8 == 0:
            print(seed, logs[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = p.parse_args()
    verify()
    train(args.seed)
