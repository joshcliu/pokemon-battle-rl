"""Paired M-B battle-policy post-training and preview experiments on Sketch5."""

import argparse
import csv
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from evaluate_vgcbench import simulator
from evaluate_vgcbench_mb import SEED_BASE, check_simulator
from poke_env.environment import DoublesEnv
from train_vgcbench_mb import FORMAT, TEAM

from pokemon_rl.battle import ROOT
from pokemon_rl.vgcbench import CACHE, TrainingBC, learned_preview, load_bc_player

RUN = ROOT / "runs/sketch5-mb-posttraining-pilot"
REPORT = ROOT / "results/sketch5-mb-posttraining-pilot"
DEV = json.loads((ROOT / "results/sketch5-vgcbench-bc/protocol.json").read_text())["teams"]


def atomic_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


@lru_cache(maxsize=128)
def team_text(path):
    return (ROOT / path).read_text()


@lru_cache(maxsize=1)
def players():
    torch.set_num_threads(1)
    return load_bc_player(FORMAT), TrainingBC(FORMAT)


class Match:
    def __init__(self):
        self.battle = simulator(FORMAT, True, CACHE / "showdown")
        self.player, self.opponent = players()

    def reset(self, own, foe, seed, side, preview="fixed", lineup="1234"):
        self.side = side
        self.foe = "p2" if side == "p1" else "p1"
        self.requests = 0
        self.seed = seed
        teams = [team_text(own), team_text(foe)]
        if side == "p2":
            teams.reverse()

        def own_pick(_):
            view = self.battle.players[side]
            if preview == "native":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 500_000_000)
                    self.lineup = learned_preview(self.player, view)
            else:
                self.lineup = lineup
                mons = list(view.team.values())
                for slot in map(int, lineup):
                    mons[slot - 1]._selected_in_teampreview = True
            return tuple(map(int, self.lineup))

        self.battle.reset(
            seed,
            *teams,
            preview_pickers={
                side: own_pick,
                self.foe: lambda _: self.opponent.preview(self.battle.players[self.foe], seed),
            },
        )

    @property
    def view(self):
        return self.battle.players[self.side]

    def obs(self):
        return {
            "observation": self.player.embed_battle(self.view, fake_rating=2000),
            "action_mask": np.asarray(DoublesEnv.get_action_mask(self.view), dtype=np.int8),
        }

    def step(self, action):
        order = DoublesEnv.action_to_order(
            np.asarray(action, dtype=np.int64), self.view, strict=True
        )
        choices = {self.side: order.message.removeprefix("/choose ")}
        while True:
            if self.battle.needs_action(self.foe):
                choices[self.foe] = self.opponent.order(self.battle.players[self.foe])
            self.battle.step(choices)
            self.requests += 1
            if self.battle.ended:
                winner = "Learner" if self.side == "p1" else "Opponent"
                return True, (
                    1 if self.battle.winner == winner else -1 if self.battle.winner else 0
                )
            if self.battle.turn >= 200 or self.requests >= 600:
                raise RuntimeError(f"Battle cutoff for simulator seed {self.seed}")
            if self.battle.needs_action(self.side):
                return False, 0
            choices = {}


def batch_obs(obs):
    return {k: torch.as_tensor(v)[None] for k, v in obs.items()}


@lru_cache(maxsize=2)
def actor(label):
    if label == "bc":
        return players()[0].policy
    if label.startswith("history-"):
        from pokemon_rl.league import FrozenActor

        seed = int(label.split("-")[1])
        return FrozenActor(
            ROOT / f"results/sketch5-vgcbench-bc/weights/seed-{seed}.pt", deterministic=False
        )
    from mb_policy import load_trained

    return load_trained(RUN / label / "policy.pt")


@torch.no_grad()
def eval_job(task):
    label, selector, own, foe, seeds, mode, cell = task
    match = Match()
    policy = actor(label)
    rows = []
    for seed, side in seeds:
        lineup = "1234"
        if selector.startswith("preview-"):
            from mb_preview import choose_lineup

            lineup = choose_lineup(RUN / selector / "head.pt", own, foe)
        match.reset(own, foe, seed, side, selector, lineup)
        torch.manual_seed(seed + 700_000_000)
        done = False
        archived = label.startswith("history-")
        if archived:
            policy.reset()
        while not done:
            if archived:
                action = policy.choose(match.view, match.battle.legal_actions(side))
            else:
                act, _, _ = policy(batch_obs(match.obs()), deterministic=mode == "greedy")
                action = act[0].tolist()
            done, reward = match.step(action)
        rows.append(
            dict(
                cell=cell,
                policy=label,
                selector=selector,
                mode=mode,
                own=Path(own).stem,
                foe=Path(foe).stem,
                seed=seed,
                side=side,
                lineup=match.lineup,
                opponent_lineup=match.opponent.lineup,
                reward=reward,
                turns=match.battle.turn,
            )
        )
    return rows


def evaluate(name, jobs, tasks):
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [r for batch in pool.map(eval_job, tasks, chunksize=1) for r in batch]
    assert len(rows) == sum(len(t[4]) for t in tasks)
    atomic_rows(REPORT / (name + ".csv"), rows)
    print(name, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)


def development_tasks(policy, selector):
    tasks = []
    for i, team in enumerate(DEV):
        for start in range(0, 64, 8):
            seeds = [
                (SEED_BASE + (4 * i + 3) * 1000 + k, ("p1", "p2")[k % 2])
                for k in range(start, start + 8)
            ]
            tasks.append((policy, selector, TEAM, team, seeds, "sample", "development"))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["baseline", "train", "preview", "test"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--arm", help="Training arm or evaluation checkpoint label")
    ap.add_argument("--steps", type=int, default=4096)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0"
    check_simulator()
    if args.stage == "baseline":
        tasks = development_tasks("bc", "fixed") + development_tasks("bc", "native")
        evaluate("same-team-bc", args.jobs, tasks)
    elif args.stage == "train":
        from mb_policy import train

        train(args.arm or "native", args.seed, args.steps)
    elif args.stage == "preview":
        from mb_preview import train_preview

        train_preview(args.arm or "ppo", args.seed)
    else:
        from mb_matrix import test_matrix

        test_matrix(args.jobs, args.arm or "bc")


if __name__ == "__main__":
    main()
