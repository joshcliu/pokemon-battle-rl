"""Audit the matched external evaluation and report team- and opponent-specific results."""

import argparse
import csv
import itertools
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from fit_expanded_preview import POLICIES

from pokemon_rl.battle import ROOT


def collect(run, spec):
    groups = defaultdict(list)
    independent = 0
    for i, team in enumerate(spec["teams"]):
        for j, family in enumerate(spec["opponents"]):
            keys = set()
            observed = defaultdict(set)
            base = spec["seed_base"] + (i * 4 + min(j, 3)) * 1000
            for k in range(0, spec["battles_per_cell"], 8):
                path = run / "cells" / Path(team).stem / f"{family}-{k}.csv"
                with path.open() as stream:
                    for r in csv.DictReader(stream):
                        label = r["candidate"]
                        seed = int(r["seed"])
                        assert base + k <= seed < base + min(k + 8, spec["battles_per_cell"])
                        assert r["side"] == ("p1" if (seed - base) % 2 == 0 else "p2")
                        assert (r["opponent_team"], r["roster"], r["family"]) == (
                            team,
                            Path(team).stem,
                            family,
                        )
                        assert label in POLICIES and (label, seed) not in keys
                        assert r["outcome"] in ("win", "loss", "tie", "cutoff")
                        for field in ("lineup", "opponent_lineup"):
                            code = r[field]
                            assert len(code) == len(set(code)) == 4 and set(code) <= set("123456")
                        assert label != "original-four" or r["lineup"] == "1234"
                        assert family != "bc-first-four" or r["opponent_lineup"] == "1234"
                        keys.add((label, seed))
                        observed[seed, r["lineup"]].add(
                            (
                                r["outcome"],
                                r["opponent_lineup"],
                                r["turn"],
                                r["opponent_defaults"],
                                r.get("opponent_repairs", "0"),
                            )
                        )
                        groups[label].append(r)
            assert all(len(v) == 1 for v in observed.values())
            assert keys == set(
                itertools.product(POLICIES, range(base, base + spec["battles_per_cell"]))
            )
            independent += len(observed)
    return groups, independent


def report(run, out):
    spec = read(run / "manifest.json")
    assert spec["selectors"] == list(POLICIES)
    assert all(digest(k) == v for k, v in spec["sha256"].items())
    groups, simulated = collect(run, spec)
    rows = [r for group in groups.values() for r in group]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "manifest.json", out / "manifest.json")
    write_csv(out / "games.csv.gz", rows)
    external = set(spec["external"])
    subsets = {
        "external-12": external,
        "previous-seven": set(spec["teams"]) - external,
        **{Path(t).stem: {t} for t in spec["teams"]},
    }
    scores, deltas, controls = [], [], []
    for subset, teams in subsets.items():
        selected = {p: [r for r in groups[p] if r["opponent_team"] in teams] for p in POLICIES}
        for family in spec["opponents"]:
            g = {p: [r for r in pool if r["family"] == family] for p, pool in selected.items()}
            for p in POLICIES:
                scores.append(dict(subset=subset, family=family, candidate=p, **score(g[p])))
            for p in POLICIES[1:]:
                deltas.append(
                    dict(
                        subset=subset,
                        family=family,
                        candidate=p,
                        reference=POLICIES[0],
                        **paired(g[p], g[POLICIES[0]]),
                    )
                )
        for p, pool in selected.items():

            def bc(family):
                return [dict(r, family="bc") for r in pool if r["family"] == family]

            controls.append(
                dict(subset=subset, candidate=p, **paired(bc("bc-learned"), bc("bc-first-four")))
            )
    write_csv(out / "win-rates.csv", scores)
    write_csv(out / "preview-differences.csv", deltas)
    write_csv(out / "opponent-preview-effect.csv", controls)
    choices = []
    for team in spec["teams"]:
        for p in POLICIES:
            counts = Counter(r["lineup"] for r in groups[p] if r["opponent_team"] == team)
            assert len(counts) == 1
            choices.append(dict(team=Path(team).stem, candidate=p, lineup=next(iter(counts))))
    write_csv(out / "team-choices.csv", choices)
    import json

    audit = dict(
        scored_outcomes=len(rows),
        unique_simulated_games=simulated,
        outcomes=Counter(r["outcome"] for r in rows),
        opponent_defaults=sum(int(r["opponent_defaults"]) for r in rows),
        opponent_repairs=sum(int(r.get("opponent_repairs", 0)) for r in rows),
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "runs/sketch5-vgcbench-v1")
    p.add_argument("--out", type=Path, default=ROOT / "results/sketch5-vgcbench")
    args = p.parse_args()
    report(args.run, args.out)
