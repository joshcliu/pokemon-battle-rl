"""Run the 262,144-step win/loss study with training seeds 2–5."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2, 3, 4, 5)
CONDITIONS = ("damage", "switch", "mixture")
OPPONENTS = ("random", "damage", "switch", "setup")
STEPS = 262144
EVAL_SEED = 500000
BATTLES = 100


def verified_mixture(source):
    with (ROOT / "results/sketch5-long-rewards/checkpoints.csv").open() as stream:
        hashes = {
            int(row["training_seed"]): row["sha256"]
            for row in csv.DictReader(stream)
            if int(row["steps"]) == STEPS and row["training_reward"] == "win-loss"
        }
    checkpoints = {}
    for seed in SEEDS:
        train = source / f"seed-{seed}" / "win-loss/train"
        checkpoint = train / "checkpoint.pt"
        args = json.loads((train / "manifest.json").read_text())["args"]
        settings = (args["seed"], args["opponent"], args["reward_mode"], args["steps"])
        if settings != (seed, "mixture", "win-loss", STEPS):
            raise ValueError(f"Unexpected training configuration in {train}")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != hashes[seed]:
            raise ValueError(f"Previous study checkpoint hash mismatch for {checkpoint}")
        checkpoints[seed] = checkpoint
    return checkpoints


def launch(job, command, arguments):
    with (job / f"{command}.log").open("x") as log:
        subprocess.run(
            [sys.executable, "-m", "pokemon_rl.cli", command, *arguments],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def run_one(output, seed, condition, prior):
    job = output / f"seed-{seed}" / condition
    job.mkdir(parents=True)
    if prior is None:
        train = job / "train"
        launch(job, "train", [
            "--seed", str(seed), "--opponent", condition, "--reward-mode", "win-loss",
            "--steps", str(STEPS), "--run-dir", str(train),
        ])
        checkpoint = train / "checkpoint.pt"
    else:
        checkpoint = prior
    launch(job, "evaluate", [
        "--checkpoint", str(checkpoint), "--opponents", *OPPONENTS, "--seed",
        str(EVAL_SEED), "--battles", str(BATTLES), "--run-dir", str(job / "eval"),
    ])
    print(f"Finished seed {seed}, training opponent {condition}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-mixture-from", type=Path, help="Verified long-reward study root")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    prior = verified_mixture(args.reuse_mixture_from.resolve()) if args.reuse_mixture_from else {}
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(run_one, root, seed, condition, prior.get(seed) if condition == "mixture" else None)
            for condition in CONDITIONS for seed in SEEDS
        ]
        for job in jobs:
            job.result()
