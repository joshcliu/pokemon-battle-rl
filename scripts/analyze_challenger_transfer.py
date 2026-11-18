"""Audit matched challenger-transfer games and summarize generalization and uncertainty."""

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import FRESH, MODELS, digest, read

from pokemon_rl.experiments import wilson

LABELS = {
    "tournament-mlp": "Tournament MLP, 262k",
    "tournament-full65": "Tournament Transformer, 65k",
    "tournament-full262": "Tournament Transformer, 262k",
    "parent": "Original league champion",
    "adapted": "Adapted defender",
    "challenger-mirror": "Mirror challenger",
    "challenger-fixed": "Fixed-MC461 challenger",
}
SUBSETS = {"all-seven": None, "unseen-five": FRESH, "trained-two": {"MC646", "MC751"}}
POOLS = {
    "all-prior-bots": None,
    "earlier-damage-switch": {"damage", "switch"},
    "setup-tactical": {"setup", "tactical"},
    "learned-ppo": {"ppo"},
    **{f: {f} for f in ("damage", "switch", "setup", "tactical", "random")},
}
COMPARISONS = [
    (model, baseline)
    for model in ("challenger-mirror", "challenger-fixed")
    for baseline in (
        "parent",
        "tournament-full65",
        "tournament-full262",
        "tournament-mlp",
        "adapted",
    )
]


def key(r):
    return r["roster"], r["family"], r["opponent_team"], r["seed"], r["side"]


def score(rows):
    wins = sum(r["outcome"] == "win" for r in rows)
    lo, hi = wilson(wins, len(rows))
    return {
        "wins": wins,
        "battles": len(rows),
        "win_pct": 100 * wins / len(rows),
        "low95_pct": 100 * lo,
        "high95_pct": 100 * hi,
        "ties": sum(r["outcome"] == "tie" for r in rows),
        "cutoffs": sum(r["outcome"] == "cutoff" for r in rows),
    }


def paired(a, b):
    first = {key(r): r["outcome"] == "win" for r in a}
    second = {key(r): r["outcome"] == "win" for r in b}
    assert first.keys() == second.keys() and len(first) == len(a) == len(b)
    groups = defaultdict(list)
    gained = lost = 0
    for k, outcome in first.items():
        change = int(outcome) - int(second[k])
        groups[k[:3] + (k[-1],)].append(change)
        gained += change == 1
        lost += change == -1
    # Variance of a paired mean within each fixed team/opponent/side stratum.
    variance = 0
    for values in groups.values():
        n = len(values)
        total = sum(values)
        variance += n / (n - 1) * (sum(x * x for x in values) - total * total / n)
    delta = 100 * (gained - lost) / len(a)
    margin = 1.959963984540054 * 100 * math.sqrt(variance) / len(a)
    return {
        "battles_each": len(a),
        "gained_wins": gained,
        "lost_wins": lost,
        "delta_pp": delta,
        "low95_pp": delta - margin,
        "high95_pp": delta + margin,
    }


def collect(root):
    spec = read(root / "manifest.json")
    assert all(digest(r["path"]) == r["sha256"] for r in spec["models"].values())
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    all_rows = []
    for mode in ("greedy", "sampled"):
        opponents = [
            (j, row)
            for j, row in enumerate(spec["opponents"])
            if mode == "greedy" or row[0] in ("damage", "switch")
        ]
        count = spec[f"{mode}_games_per_cell"]
        for label, checkpoint in spec["models"].items():
            for i, team in enumerate(spec["teams"]):
                for j, (family, other_team, other) in opponents:
                    path = (
                        root
                        / mode
                        / label
                        / Path(team).stem
                        / f"{family}-{Path(other_team).stem}.csv"
                    )
                    with path.open() as f:
                        rows = list(csv.DictReader(f))
                    base = (
                        spec["base"]
                        + 2000000 * (mode == "sampled")
                        + (i * len(spec["opponents"]) + j) * 1000
                    )
                    assert len(rows) == count
                    assert {(int(r["seed"]), r["side"]) for r in rows} == {
                        (base + k, "p1" if k % 2 == 0 else "p2") for k in range(count)
                    }
                    assert all(
                        (
                            r["candidate"],
                            r["mode"],
                            r["team"],
                            r["opponent_team"],
                            r["opponent"],
                            r["learner"],
                            r["family"],
                            r["roster"],
                        )
                        == (
                            label,
                            mode,
                            team,
                            other_team,
                            other,
                            checkpoint["path"],
                            family,
                            Path(team).stem,
                        )
                        for r in rows
                    )
                    assert all(r["outcome"] in ("win", "loss", "tie", "cutoff") for r in rows)
                    all_rows.extend(rows)
    assert len(all_rows) == 156800
    return spec, all_rows


def analyze(root, out):
    spec, rows = collect(root)
    out.mkdir(parents=True, exist_ok=True)
    summaries, effects, teams, cells = [], [], [], []
    indexed = {}
    for mode in ("greedy", "sampled"):
        for candidate in MODELS:
            indexed[mode, candidate] = [
                r for r in rows if (r["mode"], r["candidate"]) == (mode, candidate)
            ]
        for subset, allowed in SUBSETS.items():
            for pool, families in POOLS.items():
                groups = {
                    model: [
                        r
                        for r in indexed[mode, model]
                        if (allowed is None or r["roster"] in allowed)
                        and (families is None or r["family"] in families)
                    ]
                    for model in MODELS
                }
                if not next(iter(groups.values())):
                    continue
                for candidate, data in groups.items():
                    summaries.append(
                        {
                            "mode": mode,
                            "subset": subset,
                            "pool": pool,
                            "candidate": candidate,
                            **score(data),
                        }
                    )
                for candidate, baseline in COMPARISONS:
                    effects.append(
                        {
                            "mode": mode,
                            "subset": subset,
                            "pool": pool,
                            "candidate": candidate,
                            "baseline": baseline,
                            **paired(groups[candidate], groups[baseline]),
                        }
                    )
        for model in MODELS:
            for roster in spec["teams"]:
                stem = Path(roster).stem
                data = [r for r in indexed[mode, model] if r["roster"] == stem]
                teams.append(
                    {
                        "mode": mode,
                        "candidate": model,
                        "roster": stem,
                        "piloted_during_counter_training": stem not in FRESH,
                        **score(data),
                    }
                )
                for family, other, _ in spec["opponents"]:
                    group = [
                        r for r in data if r["family"] == family and r["opponent_team"] == other
                    ]
                    if group:
                        cells.append(
                            {
                                "mode": mode,
                                "candidate": model,
                                "roster": stem,
                                "family": family,
                                "opponent_team": other,
                                **score(group),
                            }
                        )
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "paired.csv", effects)
    write_csv(out / "per-team.csv", teams)
    write_csv(out / "per-matchup.csv", cells)
    write_csv(out / "games.csv.gz", rows)
    shutil.copyfile(root / "manifest.json", out / "manifest.json")
    plot(out, summaries, teams)
    return summaries, effects


def plot(out, summaries, teams):
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(MODELS)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(models))
    for offset, pool, color, label in [
        (-0.18, "earlier-damage-switch", "#17657a", "Damage and switching"),
        (0.18, "all-prior-bots", "#ad542c", "All prior bots"),
    ]:
        lookup = {
            r["candidate"]: r
            for r in summaries
            if (r["mode"], r["subset"], r["pool"]) == ("greedy", "unseen-five", pool)
        }
        ys = [lookup[m]["win_pct"] for m in models]
        errors = [
            [lookup[m]["win_pct"] - lookup[m]["low95_pct"] for m in models],
            [lookup[m]["high95_pct"] - lookup[m]["win_pct"] for m in models],
        ]
        ax.bar(x + offset, ys, width=0.34, color=color, label=label, yerr=errors, capsize=2)
    ax.set(
        xticks=x,
        xticklabels=[LABELS[m] for m in models],
        ylabel="Win rate (%)",
        title="Transfer to the five teams never piloted during training",
    )
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out / "unseen-teams.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    rosters = ["MC407", "MC778", "MC462", "MC403", "MC405", "MC646", "MC751"]
    chosen = ["tournament-full262", "parent", "challenger-mirror", "challenger-fixed"]
    lookup = {(r["candidate"], r["roster"]): r for r in teams if r["mode"] == "greedy"}
    data = np.array([[lookup[m, t]["win_pct"] for t in rosters] for m in chosen])
    heat = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(chosen)):
        for j in range(len(rosters)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if data[i, j] > 55 else "#222",
            )
    ax.set(
        yticks=range(len(chosen)),
        yticklabels=[LABELS[m] for m in chosen],
        xticks=range(len(rosters)),
        xticklabels=[t + ("*" if t in ("MC646", "MC751") else "") for t in rosters],
        title="All prior opponents • * team used in recent counter-training",
    )
    fig.colorbar(heat, ax=ax, label="Win rate (%)")
    fig.savefig(out / "by-team.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summaries, _ = analyze(a.runs.resolve(), a.output.resolve())
    for r in summaries:
        if r["subset"] == "unseen-five" and r["pool"] in (
            "all-prior-bots",
            "earlier-damage-switch",
        ):
            print(r["mode"], r["pool"], r["candidate"], round(r["win_pct"], 2), flush=True)
