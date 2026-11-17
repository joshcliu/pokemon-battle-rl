"""Train full-history PPO with fixed or per-battle shuffled learner move slots."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import SEEDS, launch
from roster_transfer import split

BUDGETS = (65536, 262144)
MODES = ("fixed", "shuffled")


def train_dir(root, seed, mode):
    return root / f"seed-{seed}" / mode / "train"


def train(root, seed, mode):
    dest = train_dir(root, seed, mode)
    dest.parent.mkdir(parents=True)
    launch(
        dest.parent,
        "train",
        [
            "--seed",
            str(seed),
            "--opponent",
            "mixture",
            "--reward-mode",
            "win-loss",
            "--hidden-size",
            "256",
            "--transformer-chunks",
            "full",
            "--learner-teams",
            *split()["train"],
            "--steps",
            str(BUDGETS[-1]),
            "--save-initial",
            "--save-at",
            *map(str, BUDGETS),
            *(["--shuffle-move-slots"] if mode == "shuffled" else []),
            "--run-dir",
            str(dest),
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(train, args.runs.resolve(), s, m) for s in SEEDS for m in MODES]
        for task in tasks:
            task.result()
