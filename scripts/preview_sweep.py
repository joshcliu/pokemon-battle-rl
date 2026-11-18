"""Matched MC461 preview sweeps and confirmation games against the established nine pilots."""

import argparse
import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from fixed_league import matches
from roster_transfer import ROOT, split

from pokemon_rl.battle import LocalBattle

RUN = ROOT / "runs/sketch5-preview-v1"
OUT = ROOT / "results/sketch5-preview"
REF = ROOT / "results/sketch5-mc461-vs-seven/manifest.json"
LINEUPS = tuple(
    "".join(map(str, lead + back))
    for lead in combinations(range(1, 7), 2)
    for back in combinations(sorted(set(range(1, 7)) - set(lead)), 2)
)
BASE = {"train": 731000000, "test": 732000000, "confirm": 733000000}
COUNT = {"train": 12, "test": 12, "confirm": 128}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def read_games(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


@lru_cache
def resources():
    torch.set_num_threads(1)
    return LocalBattle(), {}


@lru_cache
def load_head(root):
    from preview_head import load_model

    return load_model(root / "preview-head.pt")


def play(stage, team, i, label, family, opponent, j, spec, root, selector=None):
    base = spec["base"][stage] + (i * len(spec["opponents"]) + j) * 1000
    selections = []

    def choose(seed, visible):
        assert len(visible) == 6 and visible == tuple(sorted(visible))
        if selector is not None:
            order = selector(visible)
        elif label == "learned":
            order = load_head(root).choose(visible)
        elif label == "random":
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7329]))
            order = LINEUPS[int(rng.integers(len(LINEUPS)))]
        elif label == "fixed":
            order = spec["fixed"]
        elif label == "default":
            order = "1234"
        elif label == "sweep-winner":
            order = spec["winners"][team]
        else:
            order = label
        selections.append((order, "|".join(visible)))
        return tuple(map(int, order))

    tag = dict(candidate=label, roster=Path(team).stem, family=family)
    rows = matches(
        tag,
        spec["parent"],
        spec["team"],
        opponent,
        team,
        base,
        0,
        spec["counts"][stage],
        learner_preview=choose,
        resources=resources(),
    )
    for r, (order, visible) in zip(rows, selections, strict=True):
        r.update(lineup=order, preview=visible)
    return rows


def cell(root, stage, spec, team, i, label, bots):
    dest = root / stage / Path(team).stem / f"{label}-{bots[0][1]}.csv.gz"
    if dest.exists():
        return
    rows = [
        r
        for j, family, bot in bots
        for r in play(stage, team, i, label, family, bot, j, spec, root)
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.gz")
    write_csv(tmp, rows)
    tmp.replace(dest)


def initial_manifest():
    ref = read(REF)
    parent = ref["candidates"]["parent"]
    teams = split()
    files = {
        parent,
        ref["learner_team"],
        *teams["train"],
        *teams["test"],
        *(v for v in ref["opponents"].values() if v.endswith(".pt")),
    }
    hashes = {p: digest(p) for p in sorted(files)}
    assert all(hashes[p] == h for p, h in ref["sha256"].items() if p in files)
    return dict(
        parent=parent,
        team=ref["learner_team"],
        opponents=ref["opponents"],
        teams=teams,
        lineups=LINEUPS,
        counts=COUNT,
        base=BASE,
        sha256=hashes,
        simulator=read(ROOT / "simulator.lock.json"),
    )


def launch(stage, root, jobs):
    path = root / "manifest.json"
    if not path.exists():
        assert stage == "train"
        save(path, initial_manifest())
    spec = read(path)
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    assert spec["counts"] == COUNT and spec["base"] == BASE and tuple(spec["lineups"]) == LINEUPS
    teams = spec["teams"]["train" if stage == "train" else "test"]
    labels = list(LINEUPS)
    if stage != "train":
        lock = read(root / "selection-lock.json")
        assert digest(root / "preview-head.pt") == lock["head_sha256"]
    if stage == "confirm":
        teams = spec["teams"]["train"] + spec["teams"]["test"]
        spec.update(fixed=lock["fixed"], winners=read(root / "sweep-winners.json"))
        labels = ["learned", "fixed", "default", "random", "sweep-winner"]
    bots = [(j, k, v) for j, (k, v) in enumerate(spec["opponents"].items())]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = [
            pool.submit(cell, root, stage, spec, t, i, label, group)
            for i, t in enumerate(teams)
            for label in labels
            for group in ([bots] if stage != "confirm" else [[b] for b in bots])
        ]
        for n, job in enumerate(as_completed(tasks), 1):
            job.result()
            if n % 100 == 0 or n == len(tasks):
                print(f"{stage}: {n}/{len(tasks)} cells", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=BASE)
    p.add_argument("--root", type=Path, default=RUN)
    p.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    args = p.parse_args()
    launch(args.stage, args.root, args.jobs)
