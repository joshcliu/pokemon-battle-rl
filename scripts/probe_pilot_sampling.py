"""Compare sampled and greedy play on the first 40 matched seeds per bot and roster."""

import argparse
import csv
import gzip
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_mc461_transfer import cell_path, evaluate_cell
from evaluate_mixture_pilot import PILOT, REFERENCE
from pilot_opponent_mixture import SEEDS


def probe(out, root, workers):
    manifest = read(out / "manifest.json")
    names = [
        "original-65536",
        *(f"{arm}-{seed}" for arm in ("defender", "mixed") for seed in SEEDS),
    ]
    spec = dict(
        read(REFERENCE / "manifest.json"),
        candidates={m: manifest["models"][m] for m in names},
        battles_per_cell=40,
    )
    used = {
        *spec["candidates"].values(),
        *spec["opponent_teams"],
        spec["learner_team"],
        *[p for p in spec["opponents"].values() if p.endswith(".pt")],
    }
    assert all(digest(p) == manifest["sha256"][p] for p in used)
    rows = defaultdict(list)
    with gzip.open(out / "games.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            name = row["candidate"]
            if (
                row["field"] == "previous"
                and name in names
                and (int(row["seed"]) - spec["base"]) % 1000 < spec["battles_per_cell"]
            ):
                rows["greedy", name].append(row)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(evaluate_cell, root, spec, name, i, j, sample_learner=True)
            for name in names
            for i in range(len(spec["opponent_teams"]))
            for j in range(len(spec["opponents"]))
        ]
        for job in jobs:
            job.result()
    for name in names:
        for team in spec["opponent_teams"]:
            for family in spec["opponents"]:
                with cell_path(root, name, team, family).open() as stream:
                    rows["sampled", name].extend(csv.DictReader(stream))
    summary, effects = [], []
    for mode in ("greedy", "sampled"):
        for name in names:
            games = rows[mode, name]
            assert len(games) == len(spec["opponent_teams"]) * len(spec["opponents"]) * 40
            summary.append(dict(mode=mode, model=name, **score(games)))
            if name != "original-65536":
                effects.append(
                    dict(
                        mode=mode,
                        model=name,
                        baseline="original-65536",
                        **paired(games, rows[mode, "original-65536"]),
                    )
                )
    for name in names:
        effects.append(
            dict(
                mode="sampled-minus-greedy",
                model=name,
                baseline=name,
                **paired(rows["sampled", name], rows["greedy", name]),
            )
        )
    write_csv(out / "sampling-scores.csv", summary)
    write_csv(out / "sampling-effects.csv", effects)
    write_csv(out / "sampled-games.csv.gz", [r for name in names for r in rows["sampled", name]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/sketch5-mixture-pilot"))
    parser.add_argument("--work", type=Path, default=PILOT / "evaluation" / "sampled")
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    probe(args.output.resolve(), args.work.resolve(), args.jobs)
