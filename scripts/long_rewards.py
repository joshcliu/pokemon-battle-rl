"""Train two rewards on seeds 2–5 and evaluate the two requested budgets on Sketch5."""

import argparse
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "pokemon_rl.cli"]
BUDGETS = (65536, 262144)


def evaluate(job, budget, checkpoint):
    with (job / f"eval-{budget}.log").open("x") as log:
        subprocess.run(
            CLI + ["evaluate", "--checkpoint", str(checkpoint), "--seed", "10000",
                   "--battles", "200", "--opponents", "random", "damage", "switch",
                   "--run-dir", str(job / f"eval-{budget}")],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True,
        )


def run_one(root, seed, reward):
    job = root / f"seed-{seed}" / reward
    job.mkdir(parents=True)
    train = job / "train"
    with (job / "train.log").open("x") as log:
        proc = subprocess.Popen(
            CLI + ["train", "--seed", str(seed), "--reward-mode", reward, "--opponent", "mixture",
                   "--steps", str(BUDGETS[-1]), "--save-at", "16384", str(BUDGETS[0]),
                   "--run-dir", str(train)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        )
        try:
            early = train / f"checkpoint-{BUDGETS[0]}.pt"
            while not early.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"Training stopped before 64k; see {job / 'train.log'}")
                time.sleep(5)
            evaluate(job, BUDGETS[0], early)
            print(f"Evaluated {seed=} {reward=} at {BUDGETS[0]}", flush=True)
            if proc.wait():
                raise RuntimeError(f"Training failed; see {job / 'train.log'}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    evaluate(job, BUDGETS[-1], train / "checkpoint.pt")
    print(f"Evaluated {seed=} {reward=} at {BUDGETS[-1]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if platform.node().split(".")[0] != "sketch5":
        parser.error("Run this study on Sketch5")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(run_one, root, seed, mode)
                for seed in (2, 3, 4, 5) for mode in ("win-loss", "damage-only")]
        for job in jobs:
            job.result()
