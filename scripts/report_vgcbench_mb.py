"""Audit all matched M-B games and summarize the win-rate change across training seeds."""

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from analyze_challenger_transfer import key, paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_vgcbench_mb import CHAMPION, FAMILIES, OUT, SEED_BASE, STAGES, N, candidates
from opponent_diversity import SEEDS
from train_vgcbench_mb import FORMAT, RUN, TEAM


def collect(spec):
    records = defaultdict(list)
    previews = defaultdict(set)
    scopes = {}
    for step in STAGES:
        models = candidates(step)
        locked = read(RUN / f"checkpoints-{step}.json")
        assert set(locked) == set(models)
        for label, path in models.items():
            assert digest(path) == locked[label]["sha256"]
            scopes[label] = set()
            for i, team in enumerate(spec["teams"]):
                for j, family in enumerate(FAMILIES):
                    base = SEED_BASE + (4 * i + j) * 1000
                    for start in range(0, N, 8):
                        with (
                            RUN / "crossplay" / label / Path(team).stem / f"{family}-{start}.csv"
                        ).open() as stream:
                            batch = list(csv.DictReader(stream))
                        assert len(batch) == 8
                        for k, r in enumerate(batch, start):
                            side = "p1" if k % 2 == 0 else "p2"
                            assert (
                                r["candidate"],
                                r["opponent_team"],
                                r["roster"],
                                r["family"],
                                int(r["seed"]),
                                r["side"],
                            ) == (label, team, Path(team).stem, family, base + k, side)
                            assert (
                                r["outcome"] in ("win", "loss", "tie", "cutoff")
                                and r["lineup"] == "1234"
                            )
                            code = r["opponent_lineup"]
                            assert len(set(code)) == len(code) == 4 and set(code) <= set("123456")
                            previews[key(r)].add(code)
                            records[label].append(r)
                            scopes[label].add(key(r))
    assert all(len(v) == 1 for v in previews.values())
    assert all(
        v == scopes[CHAMPION] and len(v) == len(spec["teams"]) * len(FAMILIES) * N
        for v in scopes.values()
    )
    return records


def averaged_effect(pairs):
    """Average four model differences per battle before estimating stratified variance."""
    by_match = defaultdict(list)
    for first, second in pairs:
        ref = {key(r): r["outcome"] == "win" for r in second}
        for r in first:
            by_match[key(r)].append(int(r["outcome"] == "win") - int(ref[key(r)]))
    strata = defaultdict(list)
    for k, changes in by_match.items():
        assert len(changes) == len(SEEDS)
        strata[k[:3] + (k[-1],)].append(sum(changes) / len(changes))
    n = len(by_match)
    delta = 100 * sum(map(sum, strata.values())) / n
    variance = sum(
        (len(v) * sum(x * x for x in v) - sum(v) ** 2) / (len(v) - 1) for v in strata.values()
    )
    margin = 1.959963984540054 * 100 * math.sqrt(variance) / n
    return dict(
        paired_battle_seeds=n, delta_pp=delta, low95_pp=delta - margin, high95_pp=delta + margin
    )


def report():
    spec = read(RUN / "protocol.json")
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    records = collect(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUN / "protocol.json", OUT / "protocol.json")
    for step in STAGES:
        shutil.copyfile(RUN / f"checkpoints-{step}.json", OUT / f"checkpoints-{step}.json")
        shutil.copyfile(RUN / f"reproduced-{step}.csv", OUT / f"reproduced-{step}.csv")
    all_games = [r for g in records.values() for r in g]
    write_csv(OUT / "games.csv.gz", all_games)
    scores, changes, average = [], [], []
    for team in [None, *spec["teams"]]:
        scope = Path(team).stem if team else "external-12"
        for family in FAMILIES:
            g = {
                label: [
                    r
                    for r in pool
                    if r["family"] == family and (team is None or r["opponent_team"] == team)
                ]
                for label, pool in records.items()
            }
            for label, rows in g.items():
                scores.append(dict(subset=scope, family=family, candidate=label, **score(rows)))
            for stage in STAGES[1:]:
                pairs = []
                for s in SEEDS:
                    label = f"seed-{s}-step-{stage}"
                    ref = f"seed-{s}-step-0"
                    changes.append(
                        dict(
                            subset=scope,
                            family=family,
                            candidate=label,
                            reference=ref,
                            **paired(g[label], g[ref]),
                        )
                    )
                    pairs.append((g[label], g[ref]))
                average.append(
                    dict(subset=scope, family=family, step=stage, **averaged_effect(pairs))
                )
    write_csv(OUT / "win-rates.csv", scores)
    write_csv(OUT / "paired-changes.csv", changes)
    write_csv(OUT / "mean-changes.csv", average)
    training = []
    exports = []
    (OUT / "weights").mkdir(exist_ok=True)
    for seed in SEEDS:
        root = RUN / f"seed-{seed}/train"
        args = read(root / "manifest.json")["args"]
        assert (
            args["format"],
            args["team"],
            args["steps"],
            args["seed"],
            args["open_team_sheets"],
            args["transformer_chunks"],
        ) == (FORMAT, TEAM, 65536, seed + 110, True, "full")
        assert read(root / "summary.json")["steps"] == 65536
        cp = root / "checkpoint-65536.pt"
        exported = OUT / f"weights/seed-{seed}.pt"
        checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
        del checkpoint["optimizer"]
        torch.save(checkpoint, exported)
        restored = torch.load(exported, map_location="cpu", weights_only=True)
        assert all(torch.equal(v, restored["policy"][k]) for k, v in checkpoint["policy"].items())
        exports.append(
            dict(
                seed=seed,
                source_sha256=digest(cp),
                weights=str(exported.relative_to(OUT)),
                sha256=digest(exported),
            )
        )
        source = torch.load(args["init_checkpoint"], map_location="cpu", weights_only=True)
        initial = torch.load(root / "checkpoint-0.pt", map_location="cpu", weights_only=True)
        assert all(
            torch.equal(value, initial["policy"][key]) for key, value in source["policy"].items()
        )
        with (root / "metrics.jsonl").open() as stream:
            for line in stream:
                metric = json.loads(line)
                training.append(
                    dict(
                        seed=seed,
                        steps=metric["steps"],
                        episodes=metric["episodes"],
                        wins=metric["outcomes"].get("win", 0),
                        losses=metric["outcomes"].get("loss", 0),
                    )
                )
    write_csv(OUT / "training-curves.csv", training)
    write_csv(OUT / "weights.csv", exports)
    audit = dict(
        games=len(all_games),
        outcomes=Counter(r["outcome"] for r in all_games),
        native_defaults=sum(int(r["opponent_defaults"]) for r in all_games),
        replicated_games=3 * 8,
        matched_opponent_previews=True,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    report()
