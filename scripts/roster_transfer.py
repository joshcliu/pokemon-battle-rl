"""Train on tournament rosters and evaluate both training regimes on matched battles."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import ROOT, SEEDS, STEPS, launch, verified_mixture

SOURCE = Path("teams/reg-mc-regionals")
EVAL_SEED = 610000
BATTLES = 100
STYLES = ("damage", "switch")
POLICIES = ("sun", "tournament", "untrained")


def split():
    with (ROOT / SOURCE / "split.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert (
        len({r["playable_four"] for r in rows})
        == len({r["base_species_six"] for r in rows})
        == len(rows)
    )
    return {
        label: [str(SOURCE / (r["team_id"] + ".txt")) for r in rows if r["split"] == label]
        for label in ("train", "test")
    }


def train_seed(output, seed, rosters):
    job = output / f"seed-{seed}"
    job.mkdir()
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
            *rosters,
            "--save-initial",
            "--run-dir",
            str(job / "train"),
        ],
    )


def evaluate_one(root, seed, policy, roster, checkpoint):
    job = root / f"seed-{seed}" / "eval" / policy / Path(roster).stem
    job.mkdir(parents=True)
    launch(
        job,
        "evaluate",
        [
            "--checkpoint",
            str(checkpoint),
            "--team",
            roster,
            "--opponents",
            *STYLES,
            "--battles",
            str(BATTLES),
            "--seed",
            str(EVAL_SEED),
            "--replays",
            "1",
            "--run-dir",
            str(job / "games"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-sun-from", required=True, type=Path)
    parser.add_argument("--eval-jobs", type=int, default=8)
    args = parser.parse_args()
    sun = verified_mixture(args.reuse_sun_from.resolve())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rosters = split()
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        jobs = [pool.submit(train_seed, root, seed, rosters["train"]) for seed in SEEDS]
        for job in jobs:
            job.result()
    with ThreadPoolExecutor(max_workers=args.eval_jobs) as pool:
        jobs = []
        for seed in SEEDS:
            train = root / f"seed-{seed}" / "train"
            paths = (sun[seed], train / "checkpoint.pt", train / "checkpoint-0.pt")
            for roster in ["teams/sun.txt", *rosters["test"]]:
                for policy, checkpoint in zip(POLICIES, paths, strict=True):
                    jobs.append(pool.submit(evaluate_one, root, seed, policy, roster, checkpoint))
        for job in jobs:
            job.result()
