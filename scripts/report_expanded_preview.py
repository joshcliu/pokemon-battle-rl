"""Audit lineup selection on both sides and compare matched held-out outcomes."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from expand_preview import OUT, PREVIOUS, RUN
from fit_expanded_preview import POLICIES, models, pick, prediction
from preview_sweep import read_games, save
from report_preview_sweep import pair_name


def report(root, out):
    spec, lock = read(root / "manifest.json"), read(root / "selection-lock.json")
    assert digest(root / "preview-head.pt") == lock["head_sha256"]
    assert digest(root / "training-sweep.csv") == lock["training_table_sha256"]
    assert digest(root / "cv.csv") == lock["cv_sha256"]
    assert digest(PREVIOUS / "preview-head.pt") == spec["prior_head_sha256"]
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    matched = {}
    choices, all_games = [], []
    margin = lock["fallback_margin"]
    new, _ = models(root)
    for i, team in enumerate(spec["teams"]["test"]):
        roster = Path(team).stem
        public = set()
        for label in POLICIES:
            rows = []
            for j, (family, opp) in enumerate(spec["opponents"].items()):
                games = read_games(root / "confirm" / roster / f"{label}-{family}.csv.gz")
                base = spec["base"]["confirm"] + (i * len(spec["opponents"]) + j) * 1000
                n = spec["counts"]["confirm"]
                assert len(games) == n and {(int(r["seed"]), r["side"]) for r in games} == {
                    (base + k, "p1" if k % 2 == 0 else "p2") for k in range(n)
                }
                for r in games:
                    assert (
                        r["team"],
                        r["learner"],
                        r["opponent_team"],
                        r["opponent"],
                        r["family"],
                        r["candidate"],
                        r["roster"],
                    ) == (spec["team"], spec["parent"], team, opp, family, label, roster)
                    assert r["outcome"] in ("win", "loss", "tie", "cutoff")
                    public.add(r["preview"])
                    k = (roster, family, r["seed"], r["side"], r["lineup"])
                    assert matched.setdefault(k, (r["outcome"], r["turn"])) == (
                        r["outcome"],
                        r["turn"],
                    )
                rows.extend(games)
            assert len(public) == 1
            visible = tuple(next(iter(public)).split("|"))
            assert len(visible) == 6 and tuple(sorted(visible)) == visible
            selected = pick(root, label, margin, visible)
            assert {r["lineup"] for r in rows} == {selected}
            best, best_value, default_value = prediction(new, visible)
            choices.append(
                dict(
                    team=roster,
                    policy=label,
                    public_six="|".join(visible),
                    lineup=selected,
                    leads=pair_name(selected[:2]),
                    back=pair_name(selected[2:]),
                    expanded_argmax=best,
                    expanded_best_pred=best_value,
                    expanded_original_pred=default_value,
                    **score(rows),
                )
            )
            grouped[roster, label] = rows
            grouped["all-seven", label].extend(rows)
            if roster != "MC405":
                grouped["other-six", label].extend(rows)
            all_games.extend(rows)
    comparisons = [
        ("expanded-head", "original-four"),
        ("expanded-head", "previous-head"),
        ("expanded-fallback", "original-four"),
        ("expanded-fallback", "previous-head"),
        ("expanded-fallback", "expanded-head"),
        ("previous-head", "original-four"),
    ]
    effects, scores = [], []
    rosters = [Path(t).stem for t in spec["teams"]["test"]]
    for roster in [*rosters, "other-six", "all-seven"]:
        for label in POLICIES:
            scores.append(dict(team=roster, policy=label, **score(grouped[roster, label])))
        for a, b in comparisons:
            effects.append(
                dict(
                    team=roster,
                    policy=a,
                    baseline=b,
                    **paired(grouped[roster, a], grouped[roster, b]),
                )
            )
    sensitivity = []
    for a, b in comparisons:
        ds = [
            r["delta_pp"]
            for r in effects
            if r["team"] in rosters and r["policy"] == a and r["baseline"] == b
        ]
        samples = np.random.default_rng(81337).choice(ds, size=(50000, len(ds))).mean(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        sensitivity.append(
            dict(
                policy=a,
                baseline=b,
                positive=sum(d > 0 for d in ds),
                equal=sum(d == 0 for d in ds),
                negative=sum(d < 0 for d in ds),
                roster_low95_pp=lo,
                roster_high95_pp=hi,
            )
        )
    write_csv(out / "confirmation-games.csv.gz", all_games)
    write_csv(out / "team-choices.csv", choices)
    write_csv(out / "win-rates.csv", scores)
    write_csv(out / "paired.csv", effects)
    write_csv(out / "roster-sensitivity.csv", sensitivity)
    save(
        out / "audit.json",
        dict(
            confirmation_games=len(all_games),
            unique_seed_lineups=len(matched),
            identical_repeats=len(all_games) - len(matched),
            frozen_checkpoint=digest(spec["parent"]),
        ),
    )
    for f in (
        "manifest.json",
        "selection-lock.json",
        "preview-head.pt",
        "training-sweep.csv",
        "training-previews.csv",
        "cv.csv",
        "fallback-cv.csv",
    ):
        shutil.copyfile(root / f, out / f)
    for r in scores:
        if r["team"] in ("all-seven", "MC405"):
            print(
                r["team"], r["policy"], r["wins"], r["battles"], round(r["win_pct"], 2), flush=True
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=RUN)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    report(a.root, a.out)
