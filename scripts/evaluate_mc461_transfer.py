"""Keep the learner on MC461 while the earlier bots pilot the seven test rosters."""

import argparse
import csv
import gzip
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import FAMILIES, FRESH, MODELS, digest, read
from fixed_league import matches
from roster_transfer import ROOT, split
from train_fixed_league import TEAM

CANDIDATES = {k: MODELS[k] for k in ("parent", "challenger-mirror", "challenger-fixed")}
LABELS = {
    "parent": "Original Transformer",
    "challenger-mirror": "Mirror challenger",
    "challenger-fixed": "MC461-trained challenger",
}


def manifest():
    pool = read(ROOT / "results/sketch5-fixed-league/pool.json")
    opponents = {f: f for f in FAMILIES}
    opponents.update({f"ppo-{Path(t).stem}": bots[2] for t, bots in pool.items()})
    rosters = split()["test"]
    files = {
        TEAM,
        *rosters,
        *CANDIDATES.values(),
        *(k for k in opponents.values() if k.endswith(".pt")),
    }
    used = set()
    for directory in (
        "sketch5-move-slots-v1",
        "sketch5-fixed-league-v1",
        "sketch5-counterleague-v1",
    ):
        for log in (ROOT / "runs" / directory).rglob("episodes.jsonl"):
            with log.open() as stream:
                used.update(
                    json.loads(line)["simulator_seed"] for line in stream if line.endswith("\n")
                )
    for path in (ROOT / "results").rglob("*games.csv.gz"):
        with gzip.open(path, "rt") as stream:
            for row in csv.DictReader(stream):
                used.add(int(row.get("simulator_seed") or row["seed"]))
    base = 201000000
    count = 400
    while any(
        base + cell * 1000 + k in used
        for cell in range(len(rosters) * len(opponents))
        for k in range(count)
    ):
        base += 1000000
    return {
        "learner_team": TEAM,
        "opponent_teams": rosters,
        "candidates": CANDIDATES,
        "opponents": opponents,
        "battles_per_cell": count,
        "base": base,
        "greedy_neural_actions": True,
        "sha256": {p: digest(p) for p in sorted(files)},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "simulator": read(ROOT / "simulator.lock.json"),
    }


def cell_path(root, model, team, family):
    return root / "cells" / model / Path(team).stem / f"{family}.csv"


def evaluate_cell(root, spec, model, team_index, opponent_index):
    team = spec["opponent_teams"][team_index]
    family, bot = list(spec["opponents"].items())[opponent_index]
    dest = cell_path(root, model, team, family)
    n = spec["battles_per_cell"]
    base = spec["base"] + (team_index * len(spec["opponents"]) + opponent_index) * 1000
    if dest.exists():
        with dest.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == n and {int(r["seed"]) for r in rows} == set(range(base, base + n))
        return
    condition = {"candidate": model, "roster": Path(team).stem, "family": family}
    rows = matches(
        condition, spec["candidates"][model], spec["learner_team"], bot, team, base, 0, n
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    write_csv(temp, rows)
    temp.replace(dest)


def audit(root, spec):
    rows = []
    assert spec["learner_team"] == TEAM and spec["opponent_teams"] == split()["test"]
    assert spec["candidates"] == CANDIDATES
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    for model, cp in spec["candidates"].items():
        for i, team in enumerate(spec["opponent_teams"]):
            for j, (family, bot) in enumerate(spec["opponents"].items()):
                with cell_path(root, model, team, family).open() as stream:
                    group = list(csv.DictReader(stream))
                base = spec["base"] + (i * len(spec["opponents"]) + j) * 1000
                n = spec["battles_per_cell"]
                assert len(group) == n
                assert {(int(r["seed"]), r["side"]) for r in group} == {
                    (base + k, "p1" if k % 2 == 0 else "p2") for k in range(n)
                }
                assert all(
                    (
                        r["candidate"],
                        r["roster"],
                        r["family"],
                        r["learner"],
                        r["team"],
                        r["opponent"],
                        r["opponent_team"],
                    )
                    == (model, Path(team).stem, family, cp, TEAM, bot, team)
                    for r in group
                )
                assert all(r["outcome"] in ("win", "loss", "tie", "cutoff") for r in group)
                rows.extend(group)
    assert len(rows) == 75600
    return rows


def analyze(root, out, spec):
    rows = audit(root, spec)
    subsets = {
        "all-seven": None,
        "unseen-five": FRESH,
        "seen-two": {"MC646", "MC751"},
        **{Path(t).stem: {Path(t).stem} for t in spec["opponent_teams"]},
    }
    fields = {
        "all-nine": None,
        "heuristics": set(FAMILIES),
        "damage-switch": {"damage", "switch"},
        "damage-tactical": {"damage", "tactical"},
        "frozen-ppo": {k for k in spec["opponents"] if k.startswith("ppo-")},
        **{k: {k} for k in spec["opponents"]},
    }
    summary, effects = [], []
    indexed = {model: [r for r in rows if r["candidate"] == model] for model in CANDIDATES}
    for subset, teams in subsets.items():
        for field, families in fields.items():
            selected = {
                model: [
                    r
                    for r in indexed[model]
                    if (teams is None or r["roster"] in teams)
                    and (families is None or r["family"] in families)
                ]
                for model in CANDIDATES
            }
            for model, group in selected.items():
                summary.append(
                    {"subset": subset, "field": field, "candidate": model, **score(group)}
                )
            for candidate in ("challenger-mirror", "challenger-fixed"):
                effects.append(
                    {
                        "subset": subset,
                        "field": field,
                        "candidate": candidate,
                        "baseline": "parent",
                        **paired(selected[candidate], selected["parent"]),
                    }
                )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "games.csv.gz", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired.csv", effects)
    (out / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    plot(out, summary)
    for r in summary:
        if r["subset"] == "all-seven" and r["field"] in ("all-nine", "heuristics", "frozen-ppo"):
            print(
                r["field"],
                r["candidate"],
                r["wins"],
                r["battles"],
                round(r["win_pct"], 2),
                flush=True,
            )


def plot(out, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rosters = [Path(t).stem for t in split()["test"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained", sharey=True)
    positions = np.arange(len(rosters))
    colors = ("#7c8187", "#17657a", "#ad542c")
    for ax, field, title in zip(
        axes,
        ("heuristics", "frozen-ppo"),
        ("Five earlier heuristic bots", "Four frozen PPO bots"),
        strict=True,
    ):
        lookup = {(r["candidate"], r["subset"]): r for r in summary if r["field"] == field}
        for i, (candidate, color) in enumerate(zip(CANDIDATES, colors, strict=True)):
            values = [lookup[candidate, t] for t in rosters]
            ys = [r["win_pct"] for r in values]
            err = [
                [r["win_pct"] - r["low95_pct"] for r in values],
                [r["high95_pct"] - r["win_pct"] for r in values],
            ]
            ax.bar(
                positions + (i - 1) * 0.25,
                ys,
                width=0.23,
                color=color,
                label=LABELS[candidate],
                yerr=err,
                capsize=1.5,
            )
        ax.set(
            xticks=positions,
            xticklabels=rosters,
            ylim=(0, 105),
            title=title,
            xlabel="Opponent's team",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=35)
    axes[0].set_ylabel("MC461 learner win rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="outside upper center")
    fig.savefig(out / "mc461-vs-seven.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args()
    root = a.runs.resolve()
    path = root / "manifest.json"
    if not path.exists():
        spec = manifest()
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, indent=2) + "\n")
    spec = read(path)
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    assert spec["simulator"] == read(ROOT / "simulator.lock.json")
    if not a.analyze_only:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            tasks = [
                pool.submit(evaluate_cell, root, spec, model, i, j)
                for model in CANDIDATES
                for i in range(7)
                for j in range(len(spec["opponents"]))
            ]
            for n, job in enumerate(as_completed(tasks), 1):
                job.result()
                if n % 20 == 0 or n == len(tasks):
                    print(f"{n}/{len(tasks)} matchup cells complete", flush=True)
    analyze(root, a.output.resolve(), spec)
