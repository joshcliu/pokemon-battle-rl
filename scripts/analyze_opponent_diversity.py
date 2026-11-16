"""Audit the opponent study and calculate win rates with training seeds weighted equally."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import torch
from analyze_long_rewards import load, settings, write_csv
from analyze_rewards import records
from opponent_diversity import BATTLES, CONDITIONS, EVAL_SEED, OPPONENTS, ROOT, SEEDS, STEPS

OUTCOMES = ("win", "loss", "tie", "truncated")


def win_rate(rows):
    completed = [r for r in rows if r["outcome"] != "truncated"]
    return 100 * sum(r["outcome"] == "win" for r in completed) / len(completed)


def audit_train(checkpoint, seed, condition, reference):
    train = checkpoint.parent
    manifest = load(train / "manifest.json")
    args = manifest["args"]
    assert (args["seed"], args["opponent"], args["steps"], args["reward_mode"]) == (
        seed,
        condition,
        STEPS,
        "win-loss",
    )
    skip = {"seed", "opponent", "run_dir", "save_at"}
    assert settings(args, skip) == settings(reference["args"], skip)
    final = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert (final["steps"], final["seed"], final["args"]) == (STEPS, seed, args)
    summary = load(train / "summary.json")
    episodes = records(train / "episodes.jsonl")
    assert len(episodes) == summary["episodes"] and summary["steps"] == STEPS
    assert Counter(r["outcome"] for r in episodes) == Counter(summary["outcomes"])
    styles = Counter(r["opponent"] for r in episodes)
    assert set(styles) == ({"damage", "switch"} if condition == "mixture" else {condition})
    return manifest, {
        "train_commit": manifest["git_commit"],
        "host": manifest["host"],
        "steps": STEPS,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_episodes": len(episodes),
        "damage_episodes": styles["damage"],
        "switch_episodes": styles["switch"],
    }


def audit_games(directory, style, summary, schedule):
    rows = records(directory / f"{style}.jsonl")
    assert [r["seed"] for r in rows] == list(range(EVAL_SEED, EVAL_SEED + BATTLES))
    counts = Counter(r["outcome"] for r in rows)
    assert set(counts) <= set(OUTCOMES) and counts == Counter(summary[style]["outcomes"])
    for row in rows:
        assert row["opponent"] == style
        draw = (row["simulator_seed"], row["opponent_team_index"])
        assert schedule.setdefault(row["seed"], draw) == draw
    return rows


def analyze(runs, output):
    reference = load(runs / f"seed-{SEEDS[0]}/damage/train/manifest.json")
    reference_eval = load(runs / f"seed-{SEEDS[0]}/damage/eval/manifest.json")
    schedule, by_seed = {}, {}
    scores, games, hashes = [], [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            key = {"training_seed": seed, "training_opponent": condition}
            directory = runs / f"seed-{seed}" / condition / "eval"
            evaluation = load(directory / "manifest.json")
            args = evaluation["args"]
            assert settings(args, {"checkpoint", "run_dir"}) == settings(
                reference_eval["args"], {"checkpoint", "run_dir"}
            )
            assert (args["opponents"], args["battles"], args["seed"], args["deterministic"]) == (
                list(OPPONENTS),
                BATTLES,
                EVAL_SEED,
                False,
            )
            assert not (directory / "errors.jsonl").exists()
            train, provenance = audit_train(Path(args["checkpoint"]), seed, condition, reference)
            for field in ("host", "teams", "simulator", "python", "torch", "numpy"):
                assert reference[field] == train[field] == evaluation[field]
            assert args["team"] == train["args"]["team"]
            assert args["opponent_teams"] == train["args"]["opponent_teams"]
            assert not train["git_dirty"] and not evaluation["git_dirty"]
            hashes.append(key | provenance | {"eval_commit": evaluation["git_commit"]})
            summary = load(directory / "summary.json")
            rates = {}
            for style in OPPONENTS:
                rows = audit_games(directory, style, summary, schedule)
                rates[style] = win_rate(rows)
                if style == "setup":
                    for idx, path in enumerate(args["opponent_teams"]):
                        rates[f"setup_on_{Path(path).stem}"] = win_rate(
                            [r for r in rows if r["opponent_team_index"] == idx]
                        )
                for row in rows:
                    team = Path(args["opponent_teams"][row["opponent_team_index"]]).stem
                    games.append(
                        key
                        | {
                            "opponent": style,
                            "opponent_team": team,
                            "eval_seed": row["seed"],
                            "simulator_seed": row["simulator_seed"],
                            "turns": row["turn"],
                            "outcome": row["outcome"],
                        }
                    )
            rates["damage_switch_mean"] = mean((rates["damage"], rates["switch"]))
            scores.append(key | rates)
            by_seed[seed, condition] = rates
    means = {
        c: {k: mean(by_seed[s, c][k] for s in SEEDS) for k in by_seed[SEEDS[0], c]}
        for c in CONDITIONS
    }
    deviations = {c: stdev(by_seed[s, c]["setup"] for s in SEEDS) for c in CONDITIONS}
    paired = {
        c: [by_seed[s, "mixture"]["setup"] - by_seed[s, c]["setup"] for s in SEEDS]
        for c in ("damage", "switch")
    }
    result = {
        "means_pct": means,
        "setup_sample_sd_pp": deviations,
        "mixture_minus_specialist_setup_pp_in_seed_order": paired,
        "eval_outcomes": dict(Counter(r["outcome"] for r in games)),
        "setup_outcomes": dict(Counter(r["outcome"] for r in games if r["opponent"] == "setup")),
        "teams_per_100": dict(
            Counter(
                Path(reference["args"]["opponent_teams"][idx]).stem for _, idx in schedule.values()
            )
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "win-rates.csv", scores)
    write_csv(output / "checkpoints.csv", hashes)
    write_csv(output / "evaluation-games.csv.gz", games)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(by_seed, output)
    print(json.dumps(result, indent=2))


def plot(by_seed, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
    for seed in SEEDS:
        ax.plot(
            range(3),
            [by_seed[seed, c]["setup"] for c in CONDITIONS],
            marker="o",
            lw=1.2,
            alpha=0.8,
            label=f"Seed {seed}",
        )
    ax.plot(
        range(3),
        [mean(by_seed[s, c]["setup"] for s in SEEDS) for c in CONDITIONS],
        color="black",
        marker="D",
        lw=2.6,
        label="Mean",
    )
    ax.set(
        xticks=range(3),
        xticklabels=["Damage bot", "Switching bot", "50/50 mix"],
        xlabel="Training opponent",
        ylabel="Win rate against the setup bot (%)",
        ylim=(0, 100),
        title="Held-out setup bot at 262,144 learner decisions",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=5, loc="lower center", fontsize=9)
    fig.savefig(output / "setup-win-rates.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    torch.set_num_threads(1)
    analyze(args.runs, args.output)
