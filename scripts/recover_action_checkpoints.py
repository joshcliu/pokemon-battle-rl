"""Recover intermediate full-history policies and verify them against the original PPO trajectory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analyze_long_rewards import settings
from analyze_rewards import records
from analyze_roster_finetuning import identical_weights
from opponent_diversity import ROOT, SEEDS, launch
from roster_transfer import split

ORIGINAL = ROOT / "runs/sketch5-policy-size-v1"
FULL = "wide-transformer-full"
BUDGETS = (16384, 32768, 65536)


def recover(root, seed):
    job = root / f"seed-{seed}"
    job.mkdir(parents=True)
    launch(
        job,
        "train",
        [
            "--seed",
            str(seed),
            "--steps",
            str(BUDGETS[1]),
            "--opponent",
            "mixture",
            "--reward-mode",
            "win-loss",
            "--learner-teams",
            *split()["train"],
            "--transformer-chunks",
            "full",
            "--hidden-size",
            "256",
            "--save-initial",
            "--save-at",
            str(BUDGETS[0]),
            str(BUDGETS[1]),
            "--run-dir",
            str(job / "train"),
        ],
    )
    train = job / "train"
    original = ORIGINAL / f"seed-{seed}" / FULL / "train"
    assert identical_weights(train / "checkpoint-0.pt", original / "checkpoint-0.pt")
    for name in ("episodes", "metrics"):
        rows = records(train / f"{name}.jsonl")
        old = records(original / f"{name}.jsonl")[: len(rows)]
        assert len(rows) == len(old) and rows
        for a, b in zip(rows, old, strict=True):
            assert settings(a, {"steps_per_second"}) == settings(b, {"steps_per_second"})
    assert rows[-1]["steps"] == BUDGETS[1]
    print(f"Verified training prefix for seed {seed}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    root = p.parse_args().output.resolve()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(recover, root, seed) for seed in SEEDS]
        for job in jobs:
            job.result()
