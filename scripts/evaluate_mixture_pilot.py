"""Measure checkpoint progress and the short opponent-mixture pilot on matched battles."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_mc461_transfer import cell_path, evaluate_cell
from pilot_opponent_mixture import BUDGET, SEEDS
from train_fixed_league import ROOT

REFERENCE = ROOT / "results/sketch5-mc461-vs-seven"
PILOT = Path("runs/sketch5-mixture-pilot-v1")
HISTORY = {
    "original": "runs/sketch5-fixed-league-v1/league/warm-2/train",
    "mirror": "runs/sketch5-counterleague-v1/shared/367/train",
    "fixed": "runs/sketch5-counterleague-v1/fixed-counter/379/train",
}
ARCHIVED = {"original-65536": "parent", "mirror-65536": "challenger-mirror", "fixed-65536": "challenger-fixed"}


def checkpoints(ref):
    models = {"initializer": "runs/sketch5-move-slots-v1/seed-2/fixed/train/checkpoint-262144.pt"}
    for name, train in HISTORY.items():
        for step in (16384, 32768, 65536):
            label = f"{name}-{step}"
            models[label] = ref["candidates"][ARCHIVED[label]] if label in ARCHIVED else f"{train}/checkpoint-{step}.pt"
    for arm in ("defender", "mixed"):
        for seed in SEEDS:
            models[f"{arm}-{seed}"] = str(PILOT / arm / str(seed) / "train" / f"checkpoint-{BUDGET}.pt")
    return models


def schedules(ref, models):
    broad = dict(ref, candidates=models, battles_per_cell=80)
    locked = read(ROOT / "results/sketch5-counterleague/locked.json")
    defenders = read(ROOT / "results/sketch5-counterleague/opponents.json")
    for i in range(len(ref["opponent_teams"])):
        for j in range(len(ref["opponents"])):
            yield "previous", broad, i, j
    for i, team in enumerate(locked["team_order"]):
        spec = dict(broad, opponent_teams=[team], opponents={"defender": defenders[team][0]},
                    base=locked["base"] + 50000 + i * 10000, battles_per_cell=200)
        yield "defenders", spec, 0, 0


def archived_rows(ref):
    labels = {old: new for new, old in ARCHIVED.items()}
    with gzip.open(REFERENCE / "games.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if (int(row["seed"]) - ref["base"]) % 1000 < 80:
                yield dict(row, candidate=labels[row["candidate"]])


def training(out):
    runs = {label: ROOT / path for label, path in HISTORY.items()}
    runs.update({f"{arm}-{seed}": ROOT / PILOT / arm / str(seed) / "train"
                 for arm in ("defender", "mixed") for seed in SEEDS})
    curves, used = [], set()
    for name, path in runs.items():
        bins = defaultdict(list)
        with (path / "episodes.jsonl").open() as stream:
            for line in stream:
                ep = json.loads(line)
                bins[(ep["step"] - 1) // BUDGET].append(ep)
                used.add(ep["simulator_seed"])
        for index, episodes in sorted(bins.items()):
            curves.append(dict(model=name, end_step=(index + 1) * BUDGET, **score(episodes)))
    write_csv(out / "training-curves.csv", curves)
    return used


def evaluate(root, out, workers):
    ref = read(REFERENCE / "manifest.json")
    models = checkpoints(ref)
    out.mkdir(parents=True, exist_ok=True)
    training_seeds = training(out)
    cells = list(schedules(ref, models))
    files = set(ref["sha256"]) | set(models.values())
    files.update(bot for _, spec, _, _ in cells for bot in spec["opponents"].values() if bot.endswith(".pt"))
    hashes = {f: digest(f) for f in sorted(files)}
    assert all(hashes[f] == h for f, h in ref["sha256"].items())
    spec_path = root / "evaluation-manifest.json"
    manifest = dict(models=models, sha256=hashes, previous_base=ref["base"], previous_count=80,
                    defender_count=200, simulator=read(ROOT / "simulator.lock.json"))
    if spec_path.exists():
        assert read(spec_path) == manifest
    root.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(evaluate_cell, root / field, spec, name, i, j)
                for name in models for field, spec, i, j in cells
                if field != "previous" or name not in ARCHIVED]
        for k, job in enumerate(as_completed(jobs), 1):
            job.result()
            if k % 100 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} new cells complete", flush=True)
    groups = defaultdict(list)
    for row in archived_rows(ref):
        groups["previous", row["candidate"]].append(row)
    for name in models:
        for field, spec, i, j in cells:
            team = spec["opponent_teams"][i]
            family, bot = list(spec["opponents"].items())[j]
            path = cell_path(root / field, name, team, family)
            if field == "previous" and name in ARCHIVED:
                rows = [r for r in groups[field, name] if r["opponent_team"] == team and r["family"] == family]
            else:
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                groups[field, name].extend(rows)
            base = spec["base"] + (i * len(spec["opponents"]) + j) * 1000
            n = spec["battles_per_cell"]
            assert len(rows) == n and {(int(r["seed"]), r["side"]) for r in rows} == {
                (base + k, "p1" if k % 2 == 0 else "p2") for k in range(n)}
            assert all(r["team"] == ref["learner_team"] and r["opponent_team"] == team
                       and r["opponent"] == bot and r["learner"] == models[name]
                       and r["candidate"] == name and r["family"] == family for r in rows)
            assert not any(int(r["seed"]) in training_seeds for r in rows)
    comparisons = [(name, "original-65536") for name in models if name != "original-65536"]
    comparisons += [(f"mixed-{s}", f"defender-{s}") for s in SEEDS]
    summary, effects = [], []
    for field in ("previous", "defenders"):
        for name in models:
            rows = groups[field, name]
            summary.append(dict(field=field, model=name, **score(rows)))
            if field == "previous":
                for tag, learned in (("previous-ppo", True), ("previous-heuristics", False)):
                    sub = [r for r in rows if r["family"].startswith("ppo-") == learned]
                    summary.append(dict(field=tag, model=name, **score(sub)))
        for a, b in comparisons:
            effects.append(dict(field=field, model=a, baseline=b, **paired(groups[field, a], groups[field, b])))
    write_csv(out / "scores.csv", summary)
    write_csv(out / "paired.csv", effects)
    write_csv(out / "games.csv.gz", [dict(field=f, **r) for (f, _), rows in groups.items() for r in rows])
    (out / "manifest.json").write_text(spec_path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--work", type=Path, default=PILOT / "evaluation")
    parser.add_argument("--output", type=Path, default=Path("results/sketch5-mixture-pilot"))
    args = parser.parse_args()
    evaluate(args.work.resolve(), args.output.resolve(), args.jobs)
