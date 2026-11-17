"""Train one- and eight-decision PPO policies on the frozen 15-team split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import SEEDS, launch
from roster_transfer import evaluate_one, split

CHUNKS = (1, 8)
WIDTH = 64
STEPS = 65_536


def train(root, seed, chunks):
    job = root / f"seed-{seed}/chunks-{chunks}"
    job.mkdir(parents=True)
    launch(
        job,
        "train",
        [
            "--seed",
            str(seed),
            "--steps",
            str(STEPS),
            "--opponent",
            "mixture",
            "--reward-mode",
            "win-loss",
            "--learner-teams",
            *split()["train"],
            "--transformer-chunks",
            str(chunks),
            "--hidden-size",
            str(WIDTH),
            "--save-initial",
            "--run-dir",
            str(job / "train"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "train":
        root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = []
        for chunks in CHUNKS:
            mode = f"chunks-{chunks}"
            for seed in SEEDS:
                if args.phase == "train":
                    jobs.append(pool.submit(train, root, seed, chunks))
                else:
                    checkpoint = root / f"seed-{seed}" / mode / "train/checkpoint.pt"
                    for roster in split()["test"]:
                        jobs.append(pool.submit(evaluate_one, root, seed, mode, roster, checkpoint))
        for job in jobs:
            job.result()
