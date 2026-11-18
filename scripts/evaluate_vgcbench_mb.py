"""Lock a legal M-B team test and compare matched PPO checkpoints on the benchmark simulator."""

import argparse
import csv
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_vgcbench import BOTS, CACHE, game, specification
from opponent_diversity import ROOT, SEEDS
from train_vgcbench_mb import FORMAT, POOL, RUN, TEAM

OUT = ROOT / "results/sketch5-vgcbench-mb"
SNAPSHOT = "913da3602a3aa1db79f9fdc5d5222eaf8d39569d"
N = 64
SEED_BASE = 945000000
STAGES = (0, 16384, 65536)
CHAMPION = "mc461-champion"
FAMILIES = BOTS[:4]


def candidates(step):
    models = {
        f"seed-{s}-step-{step}": str(
            (RUN / f"seed-{s}/train/checkpoint-{step}.pt").relative_to(ROOT)
        )
        for s in SEEDS
    }
    if step == 0:
        models[CHAMPION] = "results/sketch5-fixed-league/weights/expert-4.pt"
    return models


def protocol():
    prior = specification(N)
    paths = [TEAM, POOL, *read(ROOT / POOL), *prior["external"]]
    return dict(
        format=FORMAT,
        open_team_sheets=True,
        learner_team=TEAM,
        lineup="1234",
        teams=prior["external"],
        families=list(FAMILIES),
        battles_per_cell=N,
        seed_base=SEED_BASE,
        primary="Per-seed change from 0 to 65536 M-B decisions against BC with learned preview",
        training_parent_seeds=list(SEEDS),
        training_budgets=list(STAGES),
        training_simulator=prior["simulator"],
        evaluation_simulator=SNAPSHOT,
        source_revision=prior["source_revision"],
        model_revision=prior["model_revision"],
        model_sha256=prior["model_sha256"],
        versions=prior["versions"],
        training_pool=read(ROOT / POOL),
        sha256={p: digest(p) for p in paths},
    )


def cell(spec, label, path, i, j, start):
    team, family = spec["teams"][i], FAMILIES[j]
    dest = RUN / "crossplay" / label / Path(team).stem / f"{family}-{start}.csv"
    if dest.exists():
        return
    settings = {
        **spec,
        "learner": path,
        "fallback_margin": 0.1,
        "simulator_path": CACHE / "showdown",
    }
    rows = []
    for k in range(start, start + 8):
        seed = SEED_BASE + (4 * i + j) * 1000 + k
        row, _ = game(settings, team, family, "original-four", seed, "p1" if k % 2 == 0 else "p2")
        rows.append(dict(candidate=label, **row))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    write_csv(tmp, rows)
    tmp.replace(dest)


def lock(path, value):
    if path.exists():
        assert read(path) == value
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, choices=STAGES, default=0)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0"
    spec = protocol()
    RUN.mkdir(parents=True, exist_ok=True)
    lock(RUN / "protocol.json", spec)
    models = candidates(a.step)
    lock(
        RUN / f"checkpoints-{a.step}.json",
        {label: dict(path=path, sha256=digest(path)) for label, path in models.items()},
    )
    if a.smoke or a.verify:
        records = []
        for j, family in enumerate(FAMILIES):
            for k in (0, 1):
                i = (j * 2 + k) % len(spec["teams"])
                label, path = list(models.items())[i % len(models)]
                team = spec["teams"][i]
                seed = SEED_BASE + (4 * i + j) * 1000 + k if a.verify else 101298 + i
                settings = {
                    **spec,
                    "learner": path,
                    "fallback_margin": 0.1,
                    "simulator_path": CACHE / "showdown",
                }
                row, _ = game(
                    settings, team, family, "original-four", seed, "p1" if k == 0 else "p2"
                )
                if a.verify:
                    with (
                        RUN / "crossplay" / label / Path(team).stem / f"{family}-0.csv"
                    ).open() as stream:
                        recorded = list(csv.DictReader(stream))[k]
                    assert all(recorded[key] == str(val) for key, val in row.items())
                records.append(dict(candidate=label, **row))
        write_csv(RUN / f"{'reproduced' if a.verify else 'smoke'}-{a.step}.csv", records)
        print(f"Checked {len(records)} games", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=a.jobs, mp_context=multiprocessing.get_context("spawn")
    ) as workers:
        futures = {
            workers.submit(cell, spec, label, path, i, j, k): (label, i, j, k)
            for label, path in models.items()
            for i in range(len(spec["teams"]))
            for j in range(len(FAMILIES))
            for k in range(0, N, 8)
        }
        errors = []
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future], repr(exc)))
                print(f"Failed {errors[-1]}", flush=True)
            if n % 100 == 0:
                print(f"Finished {n}/{len(futures)} batches", flush=True)
        if errors:
            raise RuntimeError(errors[:5])


if __name__ == "__main__":
    main()
