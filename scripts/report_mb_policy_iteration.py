"""Audit complete paired development evaluations and quantify roster/seed uncertainty."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sketch5-mb-policy-iteration"
SPEC = json.loads((OUT / "protocol.json").read_text())


def save(name, rows):
    p = OUT / name
    temp = p.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    temp.replace(p)


def interval(x):
    # Paired directions, common game seeds across pilots, stratified simulator seats.
    # Array order is pilot, team A, team B, replicate after averaging directions.
    rng = np.random.default_rng(42881)
    s, a, b, k = x.shape
    result = []
    for _ in range(4000):
        ss = rng.integers(s, size=s)
        aa = rng.integers(a, size=a)
        bb = rng.integers(b, size=b)
        kk = 2 * rng.integers(k // 2, size=(a, b, k)) + np.arange(k) % 2
        y = x[ss].mean(0)
        result.append(y[aa[:, None, None], bb[None, :, None], kk].mean())
    return np.round(np.percentile(result, (2.5, 97.5)) * 100, 2).tolist()


def report():
    summary, by_team, effects = [], [], []
    seeds = SPEC["training_seeds"]
    for split in ("validation", "custom-dev", "transfer"):
        if split in ("validation", "transfer"):
            group_a, group_b = SPEC["validation_a"], SPEC["validation_b"]
            directions = 2
        else:
            group_a = [SPEC["custom"]]
            group_b = SPEC["validation_a"] + SPEC["validation_b"]
            directions = 1
        groups = defaultdict(dict)
        for p in sorted((OUT / "scores").glob(f"{split}-*.csv")):
            with p.open() as f:
                rows = list(csv.DictReader(f))
            method, seed = rows[0]["method"], int(rows[0]["training_seed"])
            assert p.name == f"{split}-{method}-{seed}.csv" and seed in seeds
            groups[method][seed] = rows
        native = None
        equality, greedy, matrices = {}, {}, {}
        for mode in sorted(groups, key=lambda key: (key != "native", key != "map", key)):
            if set(groups[mode]) != set(seeds):
                continue
            outcomes = np.full((len(seeds), len(group_a), len(group_b), directions, 8), np.nan)
            teams = defaultdict(list)
            for si, seed in enumerate(seeds):
                for r in groups[mode][seed]:
                    i, j, d, k = [int(r[x]) for x in ("i", "j", "direction", "k")]
                    pos = si, i, j, d, k
                    x, y = group_a[i], group_b[j]
                    if d:
                        x, y = y, x
                    assert r["method"] == mode and int(r["training_seed"]) == seed
                    if split == "transfer":
                        assert r["opponent"] == "shared-2-262144-normal"
                    assert (r["own"], r["foe"]) == (Path(x).stem, Path(y).stem)
                    assert r["side"] == ("p1", "p2")[k % 2] and np.isnan(outcomes[pos])
                    assert r["reward"] in ("-1", "0", "1")
                    assert 0 <= d < directions and 0 <= k < 8 and int(r["turns"]) > 0
                    start = (
                        (610_000_000 if split in ("validation", "transfer") else 620_000_000)
                        + 10000 * i
                        + 100 * j
                        + 20 * d
                    )
                    assert int(r["sim"]) == start + k
                    for col in ("lineup", "opp_lineup"):
                        assert len(set(r[col])) == len(r[col]) == 4 and set(r[col]) <= set("123456")
                    if mode == "native":
                        equality[pos] = (r["lineup"], r["opp_lineup"])
                    else:
                        assert r["opp_lineup"] == equality[pos][1]
                        if not mode.startswith("search-") and mode not in (
                            "lineup-q-map",
                            "adaptive-es",
                        ):
                            assert r["lineup"] == equality[pos][0]
                    if mode == "map":
                        greedy[pos] = r["lineup"], r["reward"], r["turns"]
                    if (
                        mode == "opening-map"
                        and r["changed"] == "False"
                        or (
                            mode in ("search-map", "lineup-q-map") and r["lineup"] == greedy[pos][0]
                        )
                    ):
                        assert (r["lineup"], r["reward"], r["turns"]) == greedy[pos]
                    outcomes[pos] = int(r["reward"] == "1")
                    teams[r["own"], r["foe"]].append(outcomes[pos])
            assert np.isfinite(outcomes).all()
            x = outcomes.mean(axis=3)
            matrices[mode] = x
            if mode == "native":
                native = x.copy()
            assert native is not None
            lower, upper = interval(x)
            dl, du = interval(x - native)
            summary.append(
                dict(
                    split=split,
                    method=mode,
                    wins=int(outcomes.sum()),
                    games=outcomes.size,
                    win_pct=round(100 * x.mean(), 2),
                    lo95=lower,
                    hi95=upper,
                    vs_native_pp=round(100 * (x - native).mean(), 2),
                    diff_lo95=dl,
                    diff_hi95=du,
                    seeds_pct=" / ".join(f"{100 * p.mean():.2f}" for p in x),
                    directions_pct=" / ".join(
                        f"{100 * outcomes[:, :, :, d].mean():.2f}" for d in range(directions)
                    ),
                )
            )
            for (own, foe), v in sorted(teams.items()):
                by_team.append(
                    dict(
                        split=split,
                        method=mode,
                        own=own,
                        foe=foe,
                        wins=int(sum(v)),
                        games=len(v),
                        win_pct=round(100 * np.mean(v), 2),
                    )
                )
        for name, x in matrices.items():
            controls = [] if name in ("map", "native", "cool") else ["map"]
            if name.startswith("search-") and name != "search-map":
                controls += ["search-map", name.removeprefix("search-")]
            if name in ("endgame-es", "belief-es", "context-ppo") or name.startswith("joint-ppo-"):
                controls.append("es-trained")
            if name == "adaptive-es":
                controls.extend(["es-trained", "search-es-trained"])
            if name == "fitted-td":
                controls.append("fitted-terminal")
            if name == "es-trained":
                controls.append("es-initial")
            if name.startswith("group-") and name.endswith("-native"):
                controls = ["native"]
            for control in controls:
                if control not in matrices:
                    continue
                delta = x - matrices[control]
                low, high = interval(delta)
                effects.append(
                    dict(
                        split=split,
                        method=name,
                        control=control,
                        difference_pp=round(100 * delta.mean(), 2),
                        lo95=low,
                        hi95=high,
                        seeds_pp=" / ".join(f"{100 * v.mean():+.2f}" for v in delta),
                    )
                )
    if effects:
        save("paired-effects.csv", effects)
    if summary:
        save("development.csv", summary)
        save("matchups.csv", by_team)
        names = {
            "native": "Frozen PPO, sampled",
            "map": "Frozen PPO, most probable command",
            "cool": "Frozen PPO, temperature 0.5",
            "es-initial": "PPO with hand-set mechanics weights",
            "es-trained": "PPO with learned mechanics weights",
            "search-map": "Preview search + frozen PPO",
            "search-es-trained": "Preview search + learned mechanics",
            "mcq-40-2": "Monte Carlo action value (weight 2)",
            "mcq-40-8": "Monte Carlo action value (weight 8)",
            "lineup-q-map": "Offline lineup value + frozen PPO",
            "opening-map": "Opening search + frozen PPO",
            "context-ppo": "State-dependent mechanics PPO",
            "adaptive-es": "Public-matchup policy search",
            "fitted-terminal": "Fitted mechanics from terminal outcomes",
            "fitted-td": "Fitted mechanics with three-step targets",
            "endgame-es": "Endgame value search + learned mechanics",
            "belief-es": "Full-battle belief search + learned mechanics",
            "joint-ppo-16384": "Joint-action tactical PPO (16k)",
            "joint-ppo-65536": "Joint-action tactical PPO (65k)",
        }
        headings = {
            "validation": "Reciprocal development against released BC",
            "custom-dev": "Original Sun Room on the 12 development opponents",
            "transfer": "Reciprocal development against the independent shared PPO",
        }
        doc = [
            "# Four-seed win rates",
            "",
            "Each row includes every scheduled game for four pilots. Intervals resample both roster axes, training seeds and games within seat. The custom-team intervals condition on that one learner team. All tables are development diagnostics. The 16 reserved confirmation teams have a separate schedule.",
            "",
        ]
        for split, heading in headings.items():
            doc += [
                "## " + heading,
                "",
                "| Agent | Wins / games | Win rate (95% interval) | Four seed rates (%) |",
                "|---|---:|---:|---|",
            ]
            for r in sorted(
                (r for r in summary if r["split"] == split), key=lambda r: -r["win_pct"]
            ):
                doc.append(
                    f"| {names.get(r['method'], r['method'])} | {r['wins']} / {r['games']} | {r['win_pct']:.2f}% ({r['lo95']:.2f}–{r['hi95']:.2f}) | {r['seeds_pct']} |"
                )
            doc.append("")
        doc += [
            "[Exact roster matrices and methods](README.md) · [Paired gain intervals](paired-effects.csv) · [Per-matchup wins](matchups.csv) · [Machine-readable totals](development.csv)",
            "",
        ]
        (OUT / "scorecard.md").write_text("\n".join(doc))
        print(
            *(
                f"{r['split']} {r['method']}: {r['wins']}/{r['games']} ({r['win_pct']}%)"
                for r in summary
            ),
            sep="\n",
        )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()
