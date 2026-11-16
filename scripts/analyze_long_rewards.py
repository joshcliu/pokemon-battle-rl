"""Audit the matched longer reward study and compare win rates by training seed."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from pokemon_rl.battle import ROOT

REWARDS = ("win-loss", "damage-only")
STYLES = ("random", "damage", "switch")


def load(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prior_schedule():
    path = ROOT / "results/sketch5-rewards/evaluation-games.csv.gz"
    with gzip.open(path, "rt") as stream:
        return {
            int(x["eval_seed"]): (int(x["simulator_seed"]), x["opponent_team"])
            for x in csv.DictReader(stream)
        }


def settings(args, skip):
    return {k: v for k, v in args.items() if k not in skip}


def same_policy(path1, path2):
    a, b = [torch.load(p, map_location="cpu", weights_only=True) for p in (path1, path2)]
    assert (a["steps"], a["seed"]) == (b["steps"], b["seed"])
    return a["policy"].keys() == b["policy"].keys() and all(
        torch.equal(v, b["policy"][k]) for k, v in a["policy"].items()
    )


def analyze(runs, budgets, output):
    schedule = prior_schedule()
    prior = ROOT / "runs/sketch5-rewards"
    scores, games, checkpoints = [], [], []
    for seed in range(2, 6):
        for mode in REWARDS:
            rel = Path(f"seed-{seed}") / mode
            t = runs / rel / "train"
            m = load(t / "manifest.json")
            old = load(prior / rel / "train/manifest.json")
            assert m["args"]["seed"] == seed and m["args"]["reward_mode"] == mode
            assert m["host"] == old["host"] == "sketch5.csail.mit.edu"
            assert all(m[k] == old[k] for k in ("teams", "simulator", "torch", "numpy", "python"))
            excluded = {"run_dir", "steps", "save_at"}
            assert settings(m["args"], excluded) == settings(old["args"], excluded)
            same_16k = same_policy(t / "checkpoint-16384.pt", prior / rel / "train/checkpoint.pt")
            for steps in budgets:
                e = runs / rel / f"eval-{steps}"
                c = t / ("checkpoint.pt" if steps == 262144 else f"checkpoint-{steps}.pt")
                manifest = load(e / "manifest.json")
                assert Path(manifest["args"]["checkpoint"]).resolve() == c.resolve()
                ref_eval = load(prior / rel / "eval/manifest.json")
                assert settings(manifest["args"], {"checkpoint", "run_dir"}) == settings(
                    ref_eval["args"], {"checkpoint", "run_dir"}
                )
                assert all(
                    manifest[k] == m[k] for k in ("git_commit", "host", "teams", "simulator")
                )
                model = torch.load(c, map_location="cpu", weights_only=True)
                assert (model["steps"], model["seed"], model["args"]["reward_mode"]) == (
                    steps,
                    seed,
                    mode,
                )
                key = {"steps": steps, "training_seed": seed, "training_reward": mode}
                checkpoints.append(
                    key
                    | {
                        "host": m["host"],
                        "source_commit": m["git_commit"],
                        "sha256": hashlib.sha256(c.read_bytes()).hexdigest(),
                        "matches_prior_16k": same_16k,
                    }
                )
                rates = {}
                summary = load(e / "summary.json")
                assert set(summary) == set(STYLES) and not (e / "errors.jsonl").exists()
                for style in STYLES:
                    rows = [json.loads(s) for s in (e / f"{style}.jsonl").read_text().splitlines()]
                    assert [r["seed"] for r in rows] == list(range(10000, 10200))
                    counts = Counter(r["outcome"] for r in rows)
                    assert counts == Counter(summary[style]["outcomes"])
                    assert all(r["opponent"] == style for r in rows)
                    for r in rows:
                        team = Path(m["args"]["opponent_teams"][r["opponent_team_index"]]).stem
                        assert (r["simulator_seed"], team) == schedule[r["seed"]]
                        games.append(
                            key
                            | {
                                "opponent": style,
                                "eval_seed": r["seed"],
                                "simulator_seed": r["simulator_seed"],
                                "opponent_team": team,
                                "turns": r["turn"],
                                "outcome": r["outcome"],
                            }
                        )
                    rates[style] = 100 * counts["win"] / len(rows)
                scores.append(
                    key | rates | {"damage_switch_mean": (rates["damage"] + rates["switch"]) / 2}
                )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "checkpoints.csv", checkpoints)
    write_csv(output / "win-rates.csv", scores)
    write_csv(output / "evaluation-games.csv.gz", games)
    comparisons = []
    for steps in budgets:
        group = [r for r in scores if r["steps"] == steps]
        means = {
            mode: {
                k: mean(x[k] for x in group if x["training_reward"] == mode)
                for k in (*STYLES, "damage_switch_mean")
            }
            for mode in REWARDS
        }
        gaps = []
        for seed in range(2, 6):
            a, b = [
                next(
                    x["damage_switch_mean"]
                    for x in group
                    if x["training_seed"] == seed and x["training_reward"] == mode
                )
                for mode in REWARDS
            ]
            gaps.append(b - a)
        comparisons.append(
            {
                "steps": steps,
                "mean_win_rates_pct": means,
                "paired_gaps_pp": gaps,
                "mean_gap_pp": mean(gaps),
            }
        )
    result = {
        "comparisons": comparisons,
        "outcomes": dict(Counter(g["outcome"] for g in games)),
        "all_16k_weights_match_prior": all(r["matches_prior_16k"] for r in checkpoints),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return scores


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (ROOT / "results/sketch5-rewards/win-rates.csv").open() as stream:
        early = [
            {
                "steps": 16384,
                "training_seed": int(r["training_seed"]),
                "training_reward": r["reward_mode"],
                "damage_switch_mean": float(r["damage_switch_mean"]),
            }
            for r in csv.DictReader(stream)
            if r["reward_mode"] in REWARDS
        ]
    fig, ax = plt.subplots(figsize=(7.1, 4.5), layout="constrained")
    for mode, label, color in zip(
        REWARDS, ("Win/loss", "Direct damage"), ("#48698a", "#bc531c"), strict=True
    ):
        series = [r for r in early + rows if r["training_reward"] == mode]
        budgets = sorted({r["steps"] for r in series})
        for seed in range(2, 6):
            y = [
                r["damage_switch_mean"]
                for r in sorted(series, key=lambda r: r["steps"])
                if r["training_seed"] == seed
            ]
            ax.plot(budgets, y, color=color, alpha=0.18, lw=1)
        ax.plot(
            budgets,
            [mean(r["damage_switch_mean"] for r in series if r["steps"] == n) for n in budgets],
            color=color,
            label=label,
            lw=2.5,
            marker="o",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [f"{n:,}" for n in budgets])
    ax.set(
        xlabel="PPO learner decisions",
        ylabel="Win rate vs. damage and switching bots (%)",
        ylim=(0, 100),
        title="Reward comparison on Sketch5",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output / "learning-curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-64k", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    torch.set_num_threads(1)
    rows = analyze(args.runs, (65536,) if args.through_64k else (65536, 262144), args.output)
    plot(rows, args.output)
