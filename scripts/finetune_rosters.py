"""Matched tournament-roster learning curves from a Sun policy and from scratch."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import ROOT, SEEDS, STEPS, launch, verified_mixture
from roster_transfer import evaluate_one, split

STAGES = (16384, 65536, STEPS)
SCRATCH_STUDY = ROOT / "runs/sketch5-roster-v1"
SUN_STUDY = ROOT / "runs/sketch5-long-v1"
SUN = "teams/sun.txt"
MODES = ("warm", "scratch")


def train_job(root, seed, mode, prior):
    job = root / f"seed-{seed}" / mode
    job.mkdir(parents=True)
    steps = STEPS if mode == "warm" else STAGES[1]
    options = [
        "--opponent",
        "mixture",
        "--reward-mode",
        "win-loss",
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--learner-teams",
        *split()["train"],
        "--save-initial",
        "--save-at",
        *map(str, STAGES[:2]),
        "--run-dir",
        str(job / "train"),
    ]
    if mode == "warm":
        options.extend(["--init-checkpoint", str(prior)])
    launch(job, "train", options)


def checkpoint(root, seed, mode, step):
    if mode == "scratch" and step == STEPS:
        return SCRATCH_STUDY / f"seed-{seed}/train/checkpoint.pt"
    train = root / f"seed-{seed}" / mode / "train"
    return train / ("checkpoint.pt" if step == STEPS else f"checkpoint-{step}.pt")


def previous_policy(mode, step):
    if step == 0:
        return "sun" if mode == "warm" else "untrained"
    if step == STEPS and mode == "scratch":
        return "tournament"
    return None


def evaluation(root, seed, mode, step, roster):
    name = Path(roster).stem
    previous = previous_policy(mode, step)
    if previous and roster not in split()["train"]:
        return SCRATCH_STUDY / f"seed-{seed}/eval" / previous / name / "games"
    return root / f"seed-{seed}/eval" / f"{mode}-{step}" / name / "games"


def schedule_evals(pool, root, seed, mode, step):
    source = checkpoint(root, seed, mode, step)
    teams = split()
    rosters = (
        teams["train"] if previous_policy(mode, step) else [SUN, *teams["test"], *teams["train"]]
    )
    return [
        pool.submit(evaluate_one, root, seed, f"{mode}-{step}", roster, source)
        for roster in rosters
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("train", "evaluate"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--steps", type=int, nargs="+", choices=(0, *STAGES), default=[0, *STAGES])
    args = p.parse_args()
    root = args.output.resolve()
    if args.phase == "train":
        root.mkdir(parents=True, exist_ok=False)
        prior = verified_mixture(SUN_STUDY)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [
                pool.submit(train_job, root, seed, mode, prior[seed])
                for seed in SEEDS
                for mode in MODES
            ]
            for job in jobs:
                job.result()
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [
                job
                for step in args.steps
                for seed in SEEDS
                for mode in MODES
                for job in schedule_evals(pool, root, seed, mode, step)
            ]
            for job in jobs:
                job.result()
