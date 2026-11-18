"""Compare short MC461 PPO continuations from the same champion on matched seeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_challenger_transfer import read
from opponent_diversity import ROOT, launch
from train_counterleague import ORIGINAL, save
from train_fixed_league import TEAM

SEEDS = (547, 557)
BUDGET = 8192


def run_one(root, arm, seed, teams):
    job = root / arm / str(seed)
    job.mkdir(parents=True)
    launch(
        job,
        "train",
        [
            "--team",
            TEAM,
            "--opponent-teams",
            *teams,
            "--opponent",
            "mixture",
            "--opponent-pool",
            str(root / f"{arm}.json"),
            "--init-checkpoint",
            ORIGINAL,
            "--transformer-chunks",
            "full",
            "--hidden-size",
            "256",
            "--learning-rate",
            "0.0002",
            "--reward-mode",
            "win-loss",
            "--steps",
            str(BUDGET),
            "--save-at",
            str(BUDGET),
            "--seed",
            str(seed),
            "--run-dir",
            str(job / "train"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    root = parser.parse_args().runs.resolve()
    defenders = read(ROOT / "results/sketch5-counterleague/opponents.json")
    rivals = read(ROOT / "results/sketch5-fixed-league/pool.json")
    old = ["damage", "switch", "setup", "tactical", *[pilots[2] for pilots in rivals.values()]]
    save(root / "defender.json", defenders)
    save(root / "mixed.json", {team: pilots * len(old) + old for team, pilots in defenders.items()})
    with ThreadPoolExecutor(max_workers=4) as workers:
        jobs = [
            workers.submit(run_one, root, arm, seed, list(defenders))
            for arm in ("defender", "mixed")
            for seed in SEEDS
        ]
        for job in jobs:
            job.result()
