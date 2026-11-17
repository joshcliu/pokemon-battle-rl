"""Restore the opponent study's setup replays and early learning-curve checkpoints."""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analyze_long_rewards import load
from analyze_rewards import records
from opponent_diversity import BATTLES, CONDITIONS, EVAL_SEED, ROOT, SEEDS

EARLY = (16384, 65536)
ORIGINAL = ROOT / "runs/sketch5-diversity-v1"
MIXTURE = ROOT / "runs/sketch5-long-v1"
STUDY = ROOT / "results/sketch5-opponent-diversity"


def archived_training(seed, condition):
    if condition == "mixture":
        return MIXTURE / f"seed-{seed}/win-loss/train"
    return ORIGINAL / f"seed-{seed}/{condition}/train"


def run(engine, job, command, options):
    with job.with_suffix(".log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pokemon_rl.cli",
                command,
                *map(str, options),
                "--run-dir",
                str(job),
            ],
            cwd=engine,
            env=dict(os.environ, PYTHONPATH=str(engine / "src")),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate(engine, job, checkpoint, replay_count):
    run(
        engine,
        job,
        "evaluate",
        [
            "--checkpoint",
            checkpoint,
            "--opponents",
            "setup",
            "--seed",
            EVAL_SEED,
            "--battles",
            BATTLES,
            "--replays",
            replay_count,
        ],
    )


def verify_replays(engine, job, seed, condition):
    checkpoint = archived_training(seed, condition) / "checkpoint.pt"
    original = ORIGINAL / f"seed-{seed}" / condition / "eval"
    replay = job / "final"
    evaluate(engine, replay, checkpoint, BATTLES)
    assert records(replay / "setup.jsonl") == records(original / "setup.jsonl")
    for old in (original / "replays").glob("setup-*.log"):
        new = replay / "replays" / old.name

        def events(path):
            return [line for line in path.read_text().splitlines() if not line.startswith("|t:|")]

        assert events(new) == events(old)
        assert load(new.with_suffix(".input.json")) == load(old.with_suffix(".input.json"))


def early_points(engine, job, seed, condition):
    source = archived_training(seed, condition)
    train = source
    if condition != "mixture":
        train = job / "prefix"
        run(
            engine,
            train,
            "train",
            [
                "--opponent",
                condition,
                "--seed",
                seed,
                "--steps",
                EARLY[-1],
                "--save-at",
                *EARLY,
                "--reward-mode",
                "win-loss",
            ],
        )
        for filename in ("episodes.jsonl", "metrics.jsonl"):
            replayed = records(train / filename)
            archived = records(source / filename)[: len(replayed)]
            assert len(replayed) == len(archived)
            for a, b in zip(replayed, archived, strict=True):
                assert all(a[k] == v for k, v in b.items() if k != "steps_per_second")
        assert load(train / "summary.json")["steps"] == EARLY[-1]
    for steps in EARLY:
        evaluate(engine, job / f"eval-{steps}", train / f"checkpoint-{steps}.pt", 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engine",
        required=True,
        type=Path,
        help="Checkout of the original evaluation commit with simulator cache",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    engine, output = args.engine.resolve(), args.output.resolve()
    with (STUDY / "checkpoints.csv").open() as stream:
        checkpoints = list(csv.DictReader(stream))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    assert {r["eval_commit"] for r in checkpoints} == {sha}
    for row in checkpoints:
        ckpt = (
            archived_training(int(row["training_seed"]), row["training_opponent"]) / "checkpoint.pt"
        )
        assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == row["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for seed in SEEDS:
            for condition in CONDITIONS:
                job = output / f"seed-{seed}" / condition
                job.mkdir(parents=True)
                if condition != "switch":
                    tasks.append(pool.submit(verify_replays, engine, job, seed, condition))
        for seed in SEEDS:
            for condition in CONDITIONS:
                job = output / f"seed-{seed}" / condition
                tasks.append(pool.submit(early_points, engine, job, seed, condition))
        for t in tasks:
            t.result()


if __name__ == "__main__":
    main()
