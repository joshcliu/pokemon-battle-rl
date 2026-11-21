"""Group-relative battle-policy PPO on reciprocal training rosters against a frozen population."""

import argparse
import copy
import functools

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, RUN, SEEDS, verify
from mb_residual_preview import pilot, save
from mb_rollout_branches import pairing
from mb_roster_ppo import batch, create_model
from poke_env.environment import DoublesEnv
from torch import nn
from vgc_posttrain import Match, atomic_rows, batch_obs

GROUP = 4
GROUPS_PER_UPDATE = 16
UPDATES = 64


class PopulationMatch(Match):
    def __init__(self):
        super().__init__(independent=True)
        self.opponent = copy.copy(self.opponent)
        self.opponent.generator = torch.Generator()
        self.native_order = self.opponent.order

    def opponent_policy(self, other):
        if other is None:
            self.opponent.order = self.native_order
        else:

            def order(view):
                raw = self.player.embed_battle(view, fake_rating=2000)
                mask = np.asarray(DoublesEnv.get_action_mask(view), dtype=np.int8)
                obs = batch_obs(dict(observation=raw, action_mask=mask))
                act = self.opponent.sample(other, obs)[0][0].numpy()
                return DoublesEnv.action_to_order(act, view, strict=True).message.removeprefix(
                    "/choose "
                )

            self.opponent.order = order


def folder(seed):
    return RUN / "group" / f"seed-{seed}"


@functools.lru_cache(maxsize=4)
def load_actor(seed, update):
    state = torch.load(folder(seed) / f"u-{update}.pt", map_location="cpu", weights_only=True)
    assert (state["seed"], state["update"]) == (seed, update)
    model = create_model("moe")
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def trajectories(model, matches, seed, g):
    own, foe, side = pairing(seed, g)
    direction = (g // 2) % 2
    if direction:
        own, foe = foe, own
    # 75% of groups face the deployment opponent. The remaining groups
    # face a stronger frozen pilot with an independent training seed.
    rival = "bc"
    other = None
    if g % 4 == (g // 4) % 4:
        opponent_seed = SEEDS[(SEEDS.index(seed) + 1 + (g // 4) % 3) % 4]
        assert opponent_seed != seed
        other, rival = pilot(opponent_seed), f"parent-{opponent_seed}"
    sim = 802_000_000 + seed * 1_000_000 + g
    for match in matches:
        match.opponent_policy(other)
        match.reset(own, foe, sim, side, preview="native")
    assert len({(m.lineup, m.opponent.lineup) for m in matches}) == 1
    # Batched sampling draws independent commands for each member.
    torch.manual_seed(883_000_000 + seed * 1_000_000 + g)
    alive = list(range(GROUP))
    logs = [[] for _ in matches]
    outcomes = np.zeros(GROUP)
    while alive:
        ob = [matches[i].obs() for i in alive]
        a, v, lp = model(batch(ob))
        survivors = []
        for j, i in enumerate(alive):
            done, r = matches[i].step(a[j].tolist())
            logs[i].append((ob[j], a[j], v[j].item(), lp[j].item()))
            if done:
                outcomes[i] = r
            else:
                survivors.append(i)
        alive = survivors
    data, actions, old, advantages, targets = [], [], [], [], []
    for i, episode in enumerate(logs):
        values = np.array([x[2] for x in episode])
        baseline = (outcomes.sum() - outcomes[i]) / (GROUP - 1)
        gae = 0.0
        ae = np.zeros(len(episode))
        for t in reversed(range(len(episode))):
            if t + 1 == len(episode):
                delta = outcomes[i] - values[t]
            else:
                delta = 0.99 * values[t + 1] - values[t]
            gae = delta + 0.99 * 0.95 * gae
            ae[t] = gae
        for t, (ob, a, _, lp) in enumerate(episode):
            data.append(ob)
            actions.append(a)
            old.append(lp)
            # Discount the shared endpoint signal to the decision in question.
            relative = 0.99 ** (len(episode) - t - 1) * (outcomes[i] - baseline)
            advantages.append(0.5 * relative + 0.5 * ae[t])
            targets.append(values[t] + ae[t])
    from pathlib import Path

    row = dict(
        group=g,
        sim=sim,
        own=Path(own).stem,
        foe=Path(foe).stem,
        side=side,
        opponent=rival,
        lineup=matches[0].lineup,
        opp_lineup=matches[0].opponent.lineup,
        outcomes=" ".join(str(int(x)) for x in outcomes),
        decisions=len(actions),
    )
    return (data, actions, old, advantages, targets), row


def train(seed, limit):
    dest = folder(seed)
    dest.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(pilot(seed)).eval().requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-5)
    path = dest / "progress.pt"
    start, total, logs = 0, 0, []
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert ck["seed"] == seed
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start, total, logs = ck["update"], ck["decisions"], ck["logs"]
    if start >= limit:
        return
    matches = [PopulationMatch() for _ in range(GROUP)]
    try:
        for u in range(start, limit):
            rows = []
            store = [[] for _ in range(5)]
            for j in range(GROUPS_PER_UPDATE):
                sample, row = trajectories(model, matches, seed, u * GROUPS_PER_UPDATE + j)
                for col, values in zip(store, sample, strict=True):
                    col.extend(values)
                rows.append(row)
            obs, actions, old, advantages, targets = store
            size = len(actions)
            ob = batch(obs)
            a = torch.stack(actions)
            lp, adv, y = [torch.tensor(x, dtype=torch.float32) for x in (old, advantages, targets)]
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            gen = torch.Generator().manual_seed(562191 + seed * 10000 + u)
            model.zero_grad()
            for _ in range(3):
                for ids in torch.randperm(size, generator=gen).split(64):
                    _, v, new, ent = model.decision(
                        {k: o[ids] for k, o in ob.items()}, action=a[ids]
                    )
                    ratio = (new - lp[ids]).exp()
                    pg = torch.minimum(ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]).mean()
                    loss = -pg + 0.5 * (v - y[ids]).square().mean() - 0.002 * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
                    opt.step()
            total += size
            logs.append(
                dict(
                    update=u + 1,
                    decisions=total,
                    games=(u + 1) * GROUPS_PER_UPDATE * GROUP,
                    groups_with_mixed_outcomes=sum(
                        len(set(r["outcomes"].split())) > 1 for r in rows
                    ),
                    wins=sum(r["outcomes"].split().count("1") for r in rows),
                    loss=float(loss.detach()),
                )
            )
            exact = dict(seed=seed, update=u + 1, decisions=total, model=model.state_dict())
            atomic_rows(dest / f"groups-{u + 1:03}.csv", rows)
            if u + 1 in (16, 64):
                save(dest / f"u-{u + 1}.pt", exact)
            save(path, dict(**exact, optimizer=opt.state_dict(), logs=logs))
            atomic_rows(OUT / "training" / f"group-{seed}.csv", logs)
            if (u + 1) % 4 == 0:
                print(seed, logs[-1], flush=True)
    finally:
        for m in matches:
            m.battle.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--updates", type=int, default=UPDATES, choices=range(1, UPDATES + 1))
    args = p.parse_args()
    verify()
    train(args.seed, args.updates)
