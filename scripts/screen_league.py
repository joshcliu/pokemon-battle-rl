"""Select a fixed-team policy on development games and freeze it before confirmation."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from fixed_league import evaluate, name
from train_fixed_league import ARCHIVE, ROOT, TEAM


def ranking(directory):
    wins = Counter()
    with (directory / "scores.csv").open() as f:
        for r in csv.DictReader(f):
            wins[r["candidate"]] += int(r["wins"])
    return sorted(wins, key=lambda k: (-wins[k], k))


def candidates(root, stage):
    if stage == "archive":
        return {
            **{
                f"archive-{s}": name(ARCHIVE / f"seed-{s}/fixed/train/checkpoint-262144.pt")
                for s in (2, 3, 4, 5)
            },
            "damage": "damage",
            "tactical": "tactical",
        }
    return {
        f"{stage}-{p.parents[1].name}-{p.stem.split('-')[-1]}": name(p)
        for p in sorted((root / stage).glob("*/train/checkpoint-*.pt"))
        if p.stem != "checkpoint-0"
    }


def lock(root, model, label, team, validation, final_seed):
    pool = json.loads((root / "pool.json").read_text())
    weights = [model, *[k for keys in pool.values() for k in keys if k.endswith(".pt")]]
    hashes = {k: hashlib.sha256((ROOT / k).read_bytes()).hexdigest() for k in weights}
    data = {
        "label": label,
        "model": model,
        "team": team,
        "team_sha256": hashlib.sha256((ROOT / team).read_bytes()).hexdigest(),
        "models_sha256": hashes,
        "pool": pool,
        "pool_sha256": hashlib.sha256((root / "pool.json").read_bytes()).hexdigest(),
        "validation": str(validation),
        "confirmation_base": final_seed,
        "games_per_cell": 400,
    }
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["screen", "validate", "confirm"])
    p.add_argument("--group", action="append", required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--team", default=TEAM)
    p.add_argument("--base", type=int, required=True)
    p.add_argument("--final-base", type=int, default=30000000)
    p.add_argument("--battles", type=int, default=48)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    root = a.runs.resolve()
    pool = json.loads((root / "pool.json").read_text())
    if a.stage == "screen":
        models = candidates(root, a.group[0])
    elif a.stage == "validate":
        models = {}
        for group in a.group:
            src = root / group
            catalog = json.loads((src / "models.json").read_text())
            best = [label for label in ranking(src) if catalog[label].endswith(".pt")][:4]
            models.update({label: catalog[label] for label in best})
    else:
        frozen = json.loads((root / a.group[0] / "locked.json").read_text())
        assert a.base == frozen["confirmation_base"] and frozen["team"] == a.team
        assert frozen == lock(
            root, frozen["model"], frozen["label"], a.team, frozen["validation"], a.base
        )
        a.battles = frozen["games_per_cell"]
        models = {frozen["label"]: frozen["model"]}
    assert models
    evaluate(a.out, pool, a.team, models, a.base, a.battles, a.jobs)
    (a.out / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    if a.stage == "validate":
        best = ranking(a.out)[0]
        (a.out / "locked.json").write_text(
            json.dumps(lock(root, models[best], best, a.team, str(a.out), a.final_base), indent=2)
            + "\n"
        )
