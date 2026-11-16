"""Validate and summarize the twelve new reward-replication runs."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ("win-loss", "damage-only", "win-plus-hp")
OPPONENTS = ("random", "damage", "switch")


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_settings(actual, reference, overrides):
    for key, value in (reference | overrides).items():
        if key not in {"run_dir", "checkpoint"} and actual.get(key) != value:
            raise ValueError(f"Unexpected {key}: {actual.get(key)!r} (expected {value!r})")


def analyze(runs, output):
    output.mkdir(parents=True, exist_ok=False)
    pilot = ROOT / "results/pilot"
    train_manifest = json.loads((pilot / "training/manifest.json").read_text())
    eval_manifest = json.loads((pilot / "evaluation/manifest.json").read_text())
    original_battles = records(pilot / "evaluation/damage.jsonl")
    expected_schedule = {
        r["seed"]: (r["simulator_seed"], r["opponent_team_index"]) for r in original_battles
    }
    per_seed, games, intervals, provenance = [], [], [], []
    for seed in (2, 3, 4, 5):
        for mode in REWARDS:
            job = runs / f"seed-{seed}" / mode
            t, e = job / "train", job / "eval"
            train = json.loads((t / "manifest.json").read_text())
            evaluation = json.loads((e / "manifest.json").read_text())
            check_settings(train["args"], train_manifest["args"], {"seed": seed, "reward_mode": mode})
            check_settings(evaluation["args"], eval_manifest["args"], {"battles": 200})
            for manifest in (train, evaluation):
                assert manifest["teams"] == train_manifest["teams"]
                assert manifest["simulator"] == train_manifest["simulator"]
                assert manifest["git_commit"] == train["git_commit"]
                assert not manifest["git_dirty"]
            summary = json.loads((t / "summary.json").read_text())
            assert summary["steps"] == 16384
            episodes = records(t / "episodes.jsonl")
            assert dict(Counter(r["outcome"] for r in episodes)) == summary["outcomes"]
            assert not (e / "errors.jsonl").exists()
            key = {"training_seed": seed, "reward_mode": mode}
            assert train["host"] == evaluation["host"]
            provenance.append(key | {"source_commit": train["git_commit"], "host": train["host"],
                "checkpoint_sha256": hashlib.sha256((t / "checkpoint.pt").read_bytes()).hexdigest(),
                "episodes": len(episodes), "steps": summary["steps"]})
            for end in (4096, 8192, 12288, 16384):
                window = [r for r in episodes if end - 4096 < r["step"] <= end]
                counts = Counter(r["outcome"] for r in window)
                intervals.append(key | {"ending_step": end, "episodes": len(window),
                    **{k: counts[k] for k in ("win", "loss", "tie", "truncated")}})
            rates = {}
            evaluation_summary = json.loads((e / "summary.json").read_text())
            for style in OPPONENTS:
                battles = records(e / f"{style}.jsonl")
                assert [r["seed"] for r in battles] == list(range(10000, 10200))
                assert evaluation_summary[style]["battles"] == len(battles)
                assert all(r["opponent"] == style and r["outcome"] in ("win", "loss", "tie", "truncated") for r in battles)
                for row in battles:
                    schedule = row["simulator_seed"], row["opponent_team_index"]
                    assert expected_schedule.setdefault(row["seed"], schedule) == schedule
                    games.append(key | {"eval_seed": row["seed"], "opponent": style,
                        "opponent_team": Path(train["args"]["opponent_teams"][row["opponent_team_index"]]).stem,
                        "simulator_seed": row["simulator_seed"], "outcome": row["outcome"], "turns": row["turn"]})
                rates[style] = 100 * mean(x["outcome"] == "win" for x in battles)
            per_seed.append(key | rates | {"damage_switch_mean": mean((rates["damage"], rates["switch"]))})
    save_rows(output / "win-rates.csv", per_seed)
    save_rows(output / "evaluation-games.csv.gz", games)
    save_rows(output / "training-windows.csv", intervals)
    save_rows(output / "checkpoints.csv", provenance)
    plot(per_seed, output)


def plot(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True, layout="constrained")
    for opponent, ax in zip(("damage", "switch"), axes, strict=True):
        for seed in (2, 3, 4, 5):
            ax.plot(range(3), [r[opponent] for r in rows if r["training_seed"] == seed],
                    marker="o", alpha=.7, label=f"Seed {seed}")
        ax.set(title=f"Against the {opponent} bot", ylim=(0, 100), xticks=range(3),
               xticklabels=["Win/loss", "Damage", "Win + HP"])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final checkpoint win rate (%)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.suptitle("Reward comparison after 16,384 PPO decisions")
    fig.savefig(output / "seed-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.runs, args.output)
