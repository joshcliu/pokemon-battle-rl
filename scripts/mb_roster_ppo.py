"""Budget-matched shared, specialist and sparsely routed battle PPO with native BC preview."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from mb_policy import FineTune, update_ppo
from mb_reward_factorial import SPEC as ROSTERS
from mb_reward_factorial import verify_assets
from torch import nn
from vgc_posttrain import ROOT, Match, atomic_rows, players

OUT = ROOT / "results/sketch5-mb-roster-ppo"
RUN = ROOT / "runs/sketch5-mb-roster-ppo"
SPEC = json.loads((OUT / "protocol.json").read_text())
TEAMS = ROSTERS["own_train"]
SEEDS = ROSTERS["training_seeds"]
ARMS = ["shared", "specialist", "moe"]
LONG_RUN = ROOT / "runs/sketch5-mb-long-ppo"
LONG_OUT = ROOT / "results/sketch5-mb-long-ppo"
LONG_ARMS = ("shared", "moe", "moe8", "moe1")


class RoutedPPO(FineTune):
    def __init__(self, experts=4, active=2):
        super().__init__("native")
        self.active = active
        width = self.base.action_net.in_features
        actions = self.base.action_net.out_features
        self.router = nn.Linear(width, experts)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 128), nn.Tanh(), nn.Linear(128, actions))
            for _ in range(experts)
        )
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.collect_routes = False
        self.route_counts = torch.zeros(experts, dtype=torch.long)

    def battle_logits(self, obs):
        pi, vf = self.base.extract_features(obs)
        x = self.base.mlp_extractor.forward_actor(pi)
        critic = self.base.mlp_extractor.forward_critic(vf)
        gate = self.router(x)
        scores, ids = gate.topk(self.active, dim=1)
        if self.active == 1:
            # A one-entry softmax has zero gradient. Use unit forward weight and
            # the selected full-softmax gate probability as a straight-through surrogate.
            probability = gate.softmax(dim=1).gather(1, ids)
            weights = 1 + (probability - probability.detach())
        else:
            weights = scores.softmax(dim=1)
        residual = x.new_zeros((len(x), self.base.action_net.out_features))
        for i, expert in enumerate(self.experts):
            batch, position = (ids == i).nonzero(as_tuple=True)
            if len(batch):
                residual = residual.index_add(
                    0, batch, weights[batch, position, None] * expert(x[batch])
                )
        if self.collect_routes:
            self.route_counts += torch.bincount(ids.detach().flatten(), minlength=len(self.experts))
        return self.base.action_net(x) + residual, self.base.value_net(critic)


def tag(arm, seed, index=None):
    return f"{arm}-{seed}" if index is None else f"{arm}-{seed}-{index}"


def create_model(arm):
    if arm in ("moe", "moe8", "moe1"):
        return RoutedPPO(experts=8 if arm == "moe8" else 4, active=1 if arm == "moe1" else 2)
    return FineTune("native")


class Stream:
    """An independent battle, team schedule and BC sampling stream for one learner roster."""

    def __init__(self, seed, index):
        self.match = Match(independent=True)
        self.match.opponent = copy.copy(self.match.opponent)
        self.match.opponent.generator = torch.Generator()
        self.seed, self.index = seed, index
        self.rng = np.random.default_rng([5401, seed, index])
        self.episodes = []
        self.reset()

    def reset(self):
        episode = len(self.episodes)
        own = TEAMS[self.index]
        foe = str(self.rng.choice(ROSTERS["foe_train"]))
        seed = 214_000_000 + self.seed * 100_000 + self.index * 10_000 + episode
        side = ("p1", "p2")[episode % 2]
        self.setup = (own, foe, seed, side)
        self.history = []
        self.match.reset(*self.setup, preview="native")
        self.record = dict(
            own=Path(own).stem,
            foe=Path(foe).stem,
            seed=seed,
            side=side,
            lineup=self.match.lineup,
            opponent_lineup=self.match.opponent.lineup,
        )

    def step(self, action):
        done, reward = self.match.step(action)
        if done:
            self.episodes.append(dict(**self.record, reward=reward, turns=self.match.battle.turn))
            self.reset()
        else:
            self.history.append(action)
        return done, reward


def batch(observations):
    return {k: torch.from_numpy(np.stack([o[k] for o in observations])) for k in observations[0]}


def train(arm, seed, index, *, long_run=False):
    assert seed in SEEDS and arm in (LONG_ARMS if long_run else ARMS)
    if (arm == "specialist") != (index is not None) or index is not None and index not in range(4):
        raise ValueError("Provide a roster index 0–3 only for a specialist")
    budget = SPEC["total_decisions_per_seed"] // (4 if arm == "specialist" else 1)
    if long_run:
        from mb_long_state import check_reference, load_progress, save_progress

        budget = json.loads((LONG_OUT / "protocol.json").read_text())["steps"]
    dest = (LONG_RUN if long_run else RUN) / tag(arm, seed, index)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "policy.pt").exists():
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    streams = [Stream(seed, i) for i in ([index] if index is not None else range(4))]
    assert len({id(s.match.battle) for s in streams}) == len(streams)
    model = create_model(arm)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-5)
    sampling = torch.Generator().manual_seed(seed + 400)
    ob = batch([s.match.obs() for s in streams])
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(231)
        expected = players()[0].policy(ob)
        torch.manual_seed(231)
        a, v, logp = model(ob)
        assert torch.equal(a, expected[0]) and torch.allclose(v, expected[1].flatten(), atol=1e-5)
        assert torch.allclose(logp, expected[2], atol=1e-5)
    torch.manual_seed(seed + 900)
    logs = []
    if long_run and (dest / "progress.pt").is_file():
        logs = load_progress(dest, model, optimizer, sampling, streams)
    start = logs[-1]["steps"] if logs else 0
    routed = isinstance(model, RoutedPPO)
    for steps in range(start + 256, budget + 1, 256):
        data, actions, old_logp, values, rewards, dones = [], [], [], [], [], []
        with torch.no_grad():
            if routed:
                model.collect_routes = True
                model.route_counts.zero_()
            for _ in range(256 // len(streams)):
                observations = [s.match.obs() for s in streams]
                a, v, lp = model(batch(observations))
                data.extend(observations)
                actions.extend(a.unbind())
                old_logp.extend(lp.tolist())
                values.extend(v.tolist())
                for s, choice in zip(streams, a.tolist(), strict=True):
                    ended, reward = s.step(choice)
                    rewards.append(reward)
                    dones.append(ended)
            bootstrap = model.base.predict_values(batch([s.match.obs() for s in streams])).flatten()
            if routed:
                model.collect_routes = False
                assert model.route_counts.sum() == 256 * model.active
        loss = update_ppo(
            model,
            optimizer,
            sampling,
            data,
            actions,
            old_logp,
            values,
            rewards,
            dones,
            bootstrap,
            streams=len(streams),
        )
        tally = {f"expert_{i}": int(n) for i, n in enumerate(model.route_counts)} if routed else {}
        games = sum(len(s.episodes) for s in streams)
        logs.append(
            dict(steps=steps, games=games, wins=sum(r == 1 for r in rewards), loss=loss, **tally)
        )
        atomic_rows(dest / "training.csv", logs)
        if steps % 4096 == 0 or steps == 256:
            print(dest.name, steps, games, flush=True)
        if long_run and steps == 65536:
            if arm in ("shared", "moe"):
                check_reference(RUN, dest, model)
            torch.save(model.state_dict(), dest / "65k.pt.tmp")
            (dest / "65k.pt.tmp").replace(dest / "65k.pt")
        if long_run and steps % 16384 == 0:
            save_progress(dest, model, optimizer, sampling, streams, logs)
    records = [row for s in streams for row in s.episodes]
    atomic_rows(dest / "episodes.csv", records)
    tmp = dest / "policy.tmp"
    torch.save(
        dict(
            state=model.state_dict(),
            arm=arm,
            seed=seed,
            index=index,
            steps=budget,
            team_decisions={Path(TEAMS[s.index]).stem: budget // len(streams) for s in streams},
            parameters=sum(p.numel() for p in model.parameters()),
        ),
        tmp,
    )
    tmp.replace(dest / "policy.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=sorted(set(ARMS) | set(LONG_ARMS)))
    p.add_argument("seed", type=int, choices=SEEDS)
    p.add_argument("--index", type=int)
    p.add_argument("--long", action="store_true")
    args = p.parse_args()
    verify_assets()
    train(args.arm, args.seed, args.index, long_run=args.long)
