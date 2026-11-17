"""Summarize archived training trajectories and matched setup checkpoint evaluations."""

import argparse
import csv
import gzip
import os
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from analyze_long_rewards import load, write_csv
from analyze_rewards import records
from opponent_diversity import BATTLES, CONDITIONS, EVAL_SEED, OPPONENTS, SEEDS, STEPS
from reconstruct_setup_games import EARLY, ORIGINAL, STUDY, archived_training

from pokemon_rl.experiments import wilson

WINDOW = 8192
STAGES = (*EARLY, STEPS)
COLORS = {"damage": "#9b441e", "switch": "#24736c", "mixture": "#41488b"}
LABELS = {"damage": "Damage", "switch": "Switching", "mixture": "50/50 mix"}


def seed_interval(values):
    # Two-sided Student-t interval across four training seeds (df = 3), fixed evaluation schedule.
    assert len(values) == 4
    midpoint = mean(values)
    half = 3.182446305284263 * stdev(values) / sqrt(len(values))
    return {"estimate": midpoint, "seed_t_low": midpoint - half, "seed_t_high": midpoint + half}


def training_windows():
    rows = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            source = archived_training(seed, cond)
            assert load(source / "summary.json")["steps"] == STEPS
            bins = [Counter() for _ in range(STEPS // WINDOW)]
            for game in records(source / "episodes.jsonl"):
                bins[(game["step"] - 1) // WINDOW][game["outcome"]] += 1
            assert sum(b.total() for b in bins) == load(source / "summary.json")["episodes"]
            for i, counts in enumerate(bins, 1):
                completed = counts.total() - counts["truncated"]
                rows.append(
                    {
                        "training_seed": seed,
                        "training_opponent": cond,
                        "ending_step": i * WINDOW,
                        "wins": counts["win"],
                        "completed": completed,
                        "truncated": counts["truncated"],
                        "win_pct": 100 * counts["win"] / completed,
                    }
                )
    return rows


def intervals(raw):
    with gzip.open(STUDY / "evaluation-games.csv.gz", "rt") as f:
        archive = {
            (int(r["training_seed"]), r["training_opponent"], r["opponent"], int(r["eval_seed"])): r
            for r in csv.DictReader(f)
        }
    rates, per_policy, estimates, gaps, early_games = {}, [], [], [], []
    for stage in STAGES:
        for seed in SEEDS:
            for cond in CONDITIONS:
                e = (
                    ORIGINAL / f"seed-{seed}" / cond / "eval"
                    if stage == STEPS
                    else raw / f"seed-{seed}" / cond / f"eval-{stage}"
                )
                for opp in OPPONENTS if stage == STEPS else ("setup",):
                    games = records(e / f"{opp}.jsonl")
                    assert [r["seed"] for r in games] == list(range(EVAL_SEED, EVAL_SEED + BATTLES))
                    for g in games:
                        ref = archive[seed, cond, opp, g["seed"]]
                        assert ref["opponent_team"] == ("balance", "sun")[g["opponent_team_index"]]
                        assert int(ref["simulator_seed"]) == g["simulator_seed"]
                        assert g["outcome"] in {"win", "loss"}
                        if stage == STEPS:
                            assert g["outcome"] == ref["outcome"] and g["turn"] == int(ref["turns"])
                        else:
                            early_games.append(
                                {
                                    "steps": stage,
                                    "training_seed": seed,
                                    "training_opponent": cond,
                                    **g,
                                }
                            )
                    wins = np.array([g["outcome"] == "win" for g in games], dtype=float)
                    rates[stage, cond, opp, seed] = wins
                    lo, hi = wilson(wins.sum(), len(wins))
                    per_policy.append(
                        {
                            "steps": stage,
                            "training_opponent": cond,
                            "opponent": opp,
                            "training_seed": seed,
                            "wins": int(wins.sum()),
                            "battles": len(wins),
                            "win_pct": 100 * wins.mean(),
                            "wilson_low": 100 * lo,
                            "wilson_high": 100 * hi,
                        }
                    )
    # Resample shared battle-seed columns jointly, holding the four policies fixed.
    rng = np.random.default_rng(7920)
    boot = rng.integers(0, BATTLES, size=(25000, BATTLES))

    def describe(array):
        info = seed_interval(list(array.mean(axis=1) * 100))
        low, high = np.percentile(array.mean(axis=0)[boot].mean(axis=1) * 100, [2.5, 97.5])
        return info | {"battle_boot_low": low, "battle_boot_high": high}

    for stage in STAGES:
        for opp in OPPONENTS if stage == STEPS else ("setup",):
            arrays = {c: np.stack([rates[stage, c, opp, s] for s in SEEDS]) for c in CONDITIONS}
            for c, arr in arrays.items():
                estimates.append(
                    {"steps": stage, "training_opponent": c, "opponent": opp, **describe(arr)}
                )
            for a, b in (("damage", "mixture"), ("mixture", "switch"), ("damage", "switch")):
                gaps.append(
                    {
                        "steps": stage,
                        "opponent": opp,
                        "first": a,
                        "second": b,
                        **describe(arrays[a] - arrays[b]),
                    }
                )
    return per_policy, estimates, gaps, early_games


def plot(windows, estimates, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.6), layout="constrained", sharey=True)
    for j, c in enumerate(CONDITIONS):
        rows = [r for r in windows if r["training_opponent"] == c]
        x = np.arange(WINDOW, STEPS + 1, WINDOW)
        series = []
        for s in SEEDS:
            y = [r["win_pct"] for r in rows if r["training_seed"] == s]
            left.plot(x, y, color=COLORS[c], lw=0.8, alpha=0.18)
            series.append(y)
        left.plot(x, np.mean(series, axis=0), color=COLORS[c], lw=2.5, label=LABELS[c])
        held = [r for r in estimates if r["opponent"] == "setup" and r["training_opponent"] == c]
        y = np.array([r["estimate"] for r in held])
        errs = np.array(
            [[r["estimate"] - r["seed_t_low"], r["seed_t_high"] - r["estimate"]] for r in held]
        ).T
        right.errorbar(
            np.arange(3) + (j - 1) * 0.065,
            y,
            yerr=errs,
            color=COLORS[c],
            lw=1.9,
            capsize=4,
            marker="o",
        )
    left.set(
        title="During training", xlabel="Learner decisions", ylabel="Win rate (%)", ylim=(0, 100)
    )
    left.set_xticks(
        [0, 65536, 131072, 196608, 262144],
        ["0", "65,536", "131,072", "196,608", "262,144"],
        fontsize=8,
    )
    left.legend(frameon=False, loc="lower right")
    right.set(
        title="Against the held-out setup bot",
        xlabel="Checkpoint (learner decisions)",
        xlim=(-0.23, 2.24),
    )
    right.set_xticks(range(3), [f"{n:,}" for n in STAGES])
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "learning-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = training_windows()
    individual, summary, gaps, early = intervals(args.runs)
    for name, rows in [
        ("training-windows.csv", windows),
        ("checkpoint-intervals.csv", individual),
        ("win-rate-intervals.csv", summary),
        ("paired-intervals.csv", gaps),
        ("early-setup-games.csv.gz", early),
    ]:
        write_csv(args.output / name, rows)
    plot(windows, summary, args.output)
