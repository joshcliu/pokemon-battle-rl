"""Evaluate slot training on every familiar and held-out roster on a paired schedule."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from analyze_policy_size import locations
from analyze_rewards import records
from recover_action_checkpoints import FULL, ORIGINAL
from roster_transfer import BATTLES, EVAL_SEED, SEEDS, STYLES, split
from trace_battle_actions import improve_target, orders, rotate
from train_move_slots import BUDGETS, MODES, train_dir

from pokemon_rl.battle import ACTION_SIZE
from pokemon_rl.env import BattleEnv
from pokemon_rl.history import FullBattleHistory, make_policy

TESTS = ("native", "rotate", "retarget")


def checkpoint(root, seed, mode, steps):
    if mode == "fixed" and steps == BUDGETS[0]:
        return locations(ORIGINAL, seed, FULL)[0]
    return train_dir(root, seed, mode) / f"checkpoint-{steps}.pt"


def eval_path(root, seed, mode, steps, roster, test):
    return root / "eval" / str(steps) / mode / f"seed-{seed}" / Path(roster).stem / f"{test}.csv"


@torch.inference_mode()
def evaluate(root, seed, mode, steps, roster, scope):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint(root, seed, mode, steps), weights_only=True, map_location="cpu")
    assert (saved["steps"], saved["seed"], saved["transformer_chunks"], saved["hidden_size"]) == (
        steps,
        seed,
        "full",
        256,
    )
    model = make_policy(saved["obs_size"], saved["hidden_size"], "full")
    model.load_state_dict(saved["policy"])
    model.eval()
    for test in TESTS:
        dest = eval_path(root, seed, mode, steps, roster, test)
        if dest.is_file():
            with dest.open() as f:
                assert sum(1 for _ in csv.DictReader(f)) == BATTLES * len(STYLES)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for style in STYLES:
            base = BattleEnv(opponent=style, team=roster)
            if test == "rotate":
                base.learner_teams = [rotate(t) for t in base.learner_teams]
            env = FullBattleHistory(base)
            archived = []
            if scope == "test" and test == "native" and mode == "fixed" and steps == BUDGETS[0]:
                archive = locations(ORIGINAL, seed, FULL)[1]
                archived = records(archive / Path(roster).stem / "games" / f"{style}.jsonl")
            try:
                for i in range(BATTLES):
                    es = EVAL_SEED + i
                    torch.manual_seed(es)
                    obs, info = env.reset(seed=es)
                    better = eligible = ally = attacks = edits = decisions = 0
                    while True:
                        legal = base.legal()
                        mask = np.zeros((ACTION_SIZE, ACTION_SIZE), dtype=bool)
                        for a, b in legal:
                            mask[a, b] = True
                        action = model.sample(
                            torch.from_numpy(obs)[None], torch.from_numpy(mask)[None]
                        )[0]
                        pair = tuple(action[0].tolist())
                        player = base.battle.players["p1"]
                        if test == "retarget":
                            pair, n = improve_target(pair, legal, player)
                            edits += n
                        else:
                            for order in orders(pair, legal, player, {}):
                                attacks += order["kind"] == "attack"
                                ally += order["ally_attack"]
                                a, b = order["multiplier"], order["other_multiplier"]
                                if b is not None and a != b:
                                    eligible += 1
                                    better += a > b
                        obs, _, done, cutoff, info = env.step(np.asarray(pair))
                        decisions += 1
                        if done or cutoff:
                            break
                    if archived:
                        previous = archived[i]
                        assert (es, decisions) == (previous["seed"], previous["learner_decisions"])
                        assert all(
                            info[k] == previous[k]
                            for k in ("outcome", "turn", "simulator_seed", "opponent_team_index")
                        )
                    rows.append(
                        {
                            "steps": steps,
                            "training": mode,
                            "seed": seed,
                            "split": scope,
                            "roster": Path(roster).stem,
                            "test": test,
                            "style": style,
                            "eval_seed": es,
                            "simulator_seed": info["simulator_seed"],
                            "opponent_team_index": info["opponent_team_index"],
                            "outcome": info["outcome"],
                            "turns": info["turn"],
                            "decisions": decisions,
                            "better": better,
                            "eligible": eligible,
                            "ally_attacks": ally,
                            "attacks": attacks,
                            "edits": edits,
                        }
                    )
            finally:
                env.close()
        temp = dest.with_suffix(".tmp")
        write_csv(temp, rows)
        temp.replace(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--steps", choices=BUDGETS, type=int, required=True)
    p.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    p.add_argument("--jobs", type=int, default=24)
    args = p.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(evaluate, args.runs.resolve(), seed, mode, args.steps, roster, scope)
            for seed in SEEDS
            for mode in args.modes
            for scope, teams in split().items()
            for roster in teams
        ]
        for job in jobs:
            job.result()
