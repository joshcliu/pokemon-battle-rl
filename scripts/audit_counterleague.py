"""Verify training provenance, replay the exported counterleague, and archive its evidence."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from analyze_long_rewards import write_csv
from evaluate_counterleague import ALL_TEAMS, PUBLIC, digest, matchups, read
from report_fixed_league import games
from train_counterleague import ORIGINAL, ROOT, TEAM, TEAMS, save


def audit_training(root, public):
    original = torch.load(ROOT / ORIGINAL, weights_only=True, map_location="cpu")["policy"]
    frozen = read(root / "defender.json")
    assert digest(frozen["model"]) == frozen["sha256"]
    assert digest(ORIGINAL) == frozen["original_sha256"]
    pool = read(root / "shared-pool.json")
    rows, exposure = [], []
    for kind in ("adapt", "counter", "shared", "fixed-counter", "fixed-experts"):
        for path in sorted((root / kind).glob("*/train/manifest.json")):
            directory = path.parent
            manifest = read(path)
            args = manifest["args"]
            assert args["reward_mode"] == "win-loss"
            assert all(digest(t) == info["sha256"] for t, info in manifest["teams"].items())
            opponents = read(directory / "opponent-pool.json")
            assert set(args["opponent_teams"]) == set(opponents)
            if kind == "adapt":
                assert args["init_checkpoint"] == ORIGINAL and args["transformer_chunks"] == "full"
                initial = torch.load(
                    directory / "checkpoint-0.pt", weights_only=True, map_location="cpu"
                )
                assert initial["policy"].keys() == original.keys()
                assert all(torch.equal(original[k], v) for k, v in initial["policy"].items())
                assert opponents == read(root / "adapt-pool.json")
                assert args["learner_teams"] == list(TEAMS)
            else:
                assert all(opponents[t] == pool[t] for t in opponents)
            counts = Counter()
            with (directory / "episodes.jsonl").open() as f:
                for line in f:
                    episode = json.loads(line)
                    own = episode["learner_team"]
                    other = args["opponent_teams"][episode["opponent_team_index"]]
                    used = episode["opponent"]
                    assert any(used == k or used == str(ROOT / k) for k in opponents[other])
                    if kind in ("counter", "shared"):
                        assert own == other
                    if kind in ("fixed-counter", "fixed-experts"):
                        assert own == TEAM
                    counts[own, other] += 1
            if kind == "adapt":
                assert set(counts) == {(t, o) for t in TEAMS for o in opponents}
            if kind == "shared":
                assert set(counts) == {(t, t) for t in ALL_TEAMS}
            if kind == "fixed-counter":
                assert set(counts) == {(TEAM, t) for t in ALL_TEAMS}
            summary = read(directory / "summary.json")
            assert args["steps"] == summary["steps"] and sum(counts.values()) == summary["episodes"]
            records = {"group": kind, "run": directory.parent.name, "seed": args["seed"]}
            rows.append(
                {
                    **records,
                    "steps": summary["steps"],
                    "episodes": summary["episodes"],
                    "width": args["hidden_size"],
                    "history": args.get("transformer_chunks") or "current",
                    "learning_rate": args["learning_rate"],
                    "git_commit": manifest["git_commit"],
                    "init_checkpoint": args.get("init_checkpoint") or "",
                    "opponent_pool_sha256": digest(directory / "opponent-pool.json"),
                }
            )
            exposure.extend(
                {**records, "learner_team": t, "opponent_team": o, "episodes": n}
                for (t, o), n in sorted(counts.items())
            )
    assert Counter(r["group"] for r in rows) == {
        "adapt": 4,
        "counter": 20,
        "shared": 3,
        "fixed-counter": 3,
        "fixed-experts": 12,
    }
    write_csv(public / "training.csv", rows)
    write_csv(public / "exposure.csv", exposure)


def compare_replay(root, public, jobs):
    lock = read(public / "locked.json")
    catalog = lock["candidates"]
    results = {}
    for mode, fixed in [("confirmation", False), ("fixed-confirmation", True)]:
        models = {
            k: (v, ALL_TEAMS)
            for k, v in catalog.items()
            if (k in {lock["fixed_team"], lock["fixed_primary"]}) == fixed
        }
        dest = root / f"source-{mode}"
        if not (dest / "games.csv.gz").exists():
            matchups(
                dest,
                models,
                lock["pool"],
                lock["base"] + (50000 if fixed else 0),
                lock["games_per_team"],
                jobs,
                fixed_team=fixed,
            )
        source = games(dest)
        actual = games(public / mode)
        fields = (
            "candidate",
            "team",
            "opponent_team",
            "seed",
            "side",
            "visible_leads",
            "outcome",
            "turn",
        )

        def compare(rows):
            return sorted(tuple(r[k] for k in fields) for r in rows)

        assert len(source) == len(models) * len(ALL_TEAMS) * lock["games_per_team"]
        assert compare(source) == compare(actual)
        results[mode] = len(source)
    save(
        public / "export-replay-check.json",
        {"matching_games": results, "matching_fields": list(fields)},
    )


def archive(root, public):
    for stage in (
        "zero-shot",
        "defender-selection",
        "calibration",
        "baseline",
        "fixed-baseline",
        "counter-selection",
        "counter-validation",
        "fixed-selection",
        "fixed-validation",
        "retention",
    ):
        shutil.copyfile(root / stage / "scores.csv", public / (stage + "-scores.csv"))
    for file in ("defender.json", "adapt-pool.json"):
        shutil.copyfile(root / file, public / file)
    field = {}
    for stage in ("zero-shot", "calibration"):
        data = games(root / stage)
        for r in data:
            key = (r["team"], r["opponent_team"], r["opponent"], r["seed"], r["side"])
            entry = field.setdefault(key, {})
            assert stage not in entry
            entry[stage] = int(r["outcome"] == "win")
    assert len(field) == 4 * 13 * 96 and all(len(v) == 2 for v in field.values())
    scores = []
    for t in (None, *TEAMS):
        subset = [v for k, v in field.items() if t is None or k[0] == t]
        scores.append(
            {
                "roster": Path(t).stem if t else "all",
                "games_each": len(subset),
                **{s: sum(v[s] for v in subset) for s in ("zero-shot", "calibration")},
                "loss_to_win": sum(v["zero-shot"] < v["calibration"] for v in subset),
                "win_to_loss": sum(v["zero-shot"] > v["calibration"] for v in subset),
            }
        )
    write_csv(public / "adaptation.csv", scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--public", type=Path, default=PUBLIC)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    root, public = a.runs.resolve(), a.public.resolve()
    audit_training(root, public)
    archive(root, public)
    compare_replay(root, public, a.jobs)
