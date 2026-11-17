"""Train history or model-size comparisons on the frozen 15-team roster split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import SEEDS, launch
from roster_transfer import evaluate_one, split

CHUNKS = (1, 8)
WIDTH = 64
STEPS = 65_536
LARGE = {
    "wide-feedforward": (528, None),
    "wide-transformer": (256, 1),
    "wide-transformer-8": (256, 8),
    "wide-transformer-full": (256, "full"),
}


def train(root, seed, name, width, chunks):
    job = root / f"seed-{seed}" / name
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
            *(["--transformer-chunks", str(chunks)] if chunks else []),
            "--hidden-size",
            str(width),
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
    parser.add_argument("--study", choices=("history", "size"), default="history")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "train":
        root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = []
        models = LARGE if args.study == "size" else {f"chunks-{n}": (WIDTH, n) for n in CHUNKS}
        for mode, (width, chunks) in models.items():
            for seed in SEEDS:
                if args.phase == "train":
                    jobs.append(pool.submit(train, root, seed, mode, width, chunks))
                else:
                    checkpoint = root / f"seed-{seed}" / mode / "train/checkpoint.pt"
                    for roster in split()["test"]:
                        jobs.append(pool.submit(evaluate_one, root, seed, mode, roster, checkpoint))
        for job in jobs:
            job.result()
