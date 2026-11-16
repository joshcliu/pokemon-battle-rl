"""Train the three pilot rewards on seeds 2–5 and evaluate the final checkpoints."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ("win-loss", "damage-only", "win-plus-hp")


def run_one(output, seed, reward):
    job = output / f"seed-{seed}" / reward
    job.mkdir(parents=True)
    cli = [sys.executable, "-m", "pokemon_rl.cli"]
    with (job / "console.log").open("w") as log:
        subprocess.run(
            cli
            + [
                "train",
                "--seed",
                str(seed),
                "--reward-mode",
                reward,
                "--steps",
                "16384",
                "--opponent",
                "mixture",
                "--run-dir",
                str(job / "train"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            cli
            + [
                "evaluate",
                "--seed",
                "10000",
                "--battles",
                "200",
                "--opponents",
                "random",
                "damage",
                "switch",
                "--checkpoint",
                str(job / "train/checkpoint.pt"),
                "--run-dir",
                str(job / "eval"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"Completed seed {seed}: {reward}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(run_one, out, seed, reward) for seed in (2, 3, 4, 5) for reward in REWARDS
        ]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
