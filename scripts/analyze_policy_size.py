"""Measure transfer and type-sensitive targeting of wider PPO models at a fixed budget."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import torch
from analyze_long_rewards import load, settings, write_csv
from analyze_rewards import records
from analyze_roster_finetuning import identical_weights, interval, score_checkpoint, weights
from analyze_roster_transfer import OPPONENT_TEAMS, same_runtime
from analyze_transformer_rosters import ARCHIVE, summarize_rosters
from opponent_diversity import ROOT, SEEDS
from probe_target_typing import probe_cases, score_targets
from roster_transfer import BATTLES, STYLES, split
from transformer_rosters import LARGE, STEPS, WIDTH

from pokemon_rl.history import make_policy

HISTORY = ROOT / "runs/sketch5-transformer-v1"
PUBLISHED = ROOT / "results/sketch5-transformer-rosters"
MODELS = {"feedforward": (128, None), "chunks-1": (WIDTH, 1), "chunks-8": (WIDTH, 8), **LARGE}
PAIRS = [
    ("wide-feedforward", "feedforward"),
    ("wide-transformer", "chunks-1"),
    ("wide-transformer", "wide-feedforward"),
    ("wide-transformer-8", "chunks-8"),
    ("wide-transformer-8", "wide-transformer"),
    ("wide-transformer-8", "wide-feedforward"),
    ("wide-transformer-full", "wide-transformer-8"),
    ("wide-transformer-full", "wide-transformer"),
    ("wide-transformer-full", "wide-feedforward"),
]


def locations(root, seed, model):
    if model == "feedforward":
        run = ARCHIVE / f"seed-{seed}"
        return run / f"scratch/train/checkpoint-{STEPS}.pt", run / f"eval/scratch-{STEPS}"
    run = (root if model in LARGE else HISTORY) / f"seed-{seed}"
    return run / model / "train/checkpoint.pt", run / "eval" / model


def training_record(cp, seed, model, manifest, ref, original_episodes):
    args = manifest["args"]
    same_runtime(manifest, ref)
    skip = {"run_dir", "hidden_size", "transformer_chunks", "save_at"}
    assert settings(args, skip) == settings(ref["args"], skip)
    assert args["learner_teams"] == split()["train"] and args["opponent_teams"] == OPPONENT_TEAMS
    assert args["opponent"] == "mixture" and args["reward_mode"] == "win-loss"
    assert args["init_checkpoint"] is None and args["steps"] == STEPS
    width, chunks = MODELS[model]
    assert (args["hidden_size"], args.get("transformer_chunks")) == (width, chunks)
    s = weights(cp)
    assert (s["hidden_size"], s.get("transformer_chunks"), s["seed"], s["steps"]) == (
        width,
        chunks,
        seed,
        STEPS,
    )
    if model in LARGE:
        initial = weights(cp.parent / "checkpoint-0.pt")
        torch.manual_seed(seed)
        fresh = make_policy(s["obs_size"], width, chunks).state_dict()
        assert initial["steps"] == 0 and all(
            torch.equal(t, fresh[k]) for k, t in initial["policy"].items()
        )
        assert any(not torch.equal(t, fresh[k]) for k, t in s["policy"].items())
    episodes = records(cp.parent / "episodes.jsonl")
    summary = load(cp.parent / "summary.json")
    assert summary["steps"] == STEPS and summary["episodes"] == len(episodes)
    assert {r["learner_team"] for r in episodes} == set(args["learner_teams"])
    for old, new in zip(original_episodes, episodes):
        assert all(
            old[k] == new[k]
            for k in (
                "episode",
                "learner_team",
                "opponent",
                "opponent_team_index",
                "simulator_seed",
            )
        )
    return {
        "seed": seed,
        "model": model,
        "hidden_width": width,
        "chunks": chunks,
        "parameters": sum(t.numel() for t in s["policy"].values()),
        "training_steps": s["steps"],
        "training_episodes": len(episodes),
        "sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
        "training_commit": manifest["git_commit"],
    }


def analyze(root, out):
    torch.set_num_threads(1)
    rosters = split()["test"]
    assert len(rosters) == 7 and len(split()["train"]) == 15
    with (PUBLISHED / "checkpoints.csv").open() as f:
        hashes = {(int(r["seed"]), r["model"]): r["sha256"] for r in csv.DictReader(f)}
    schedule, scores, games, provenance, probe, coverage = {}, [], [], [], [], []
    cases = probe_cases()
    for seed in SEEDS:
        assert identical_weights(
            *(
                root / f"seed-{seed}" / name / "train/checkpoint-0.pt"
                for name in ("wide-transformer", "wide-transformer-8")
            )
        )
        assert identical_weights(
            root / f"seed-{seed}/wide-transformer-8/train/checkpoint-0.pt",
            root / f"seed-{seed}/wide-transformer-full/train/checkpoint-0.pt",
        )
        base, _ = locations(root, seed, "feedforward")
        ref = load(base.parent / "manifest.json")
        old_episodes = records(base.parent / "episodes.jsonl")
        for model in MODELS:
            cp, eval_dir = locations(root, seed, model)
            manifest = load(cp.parent / "manifest.json")
            row = training_record(cp, seed, model, manifest, ref, old_episodes)
            if model not in LARGE:
                assert row["sha256"] == hashes[seed, model]
            provenance.append(row)
            condition = {"seed": seed, "model": model}
            eval_lengths = []
            for roster in rosters:
                dest = eval_dir / Path(roster).stem / "games"
                cells, played = score_checkpoint(dest, cp, roster, manifest, schedule, condition)
                scores.extend(cells)
                games.extend(played)
                if model == "wide-transformer-full":
                    for style in STYLES:
                        eval_lengths.extend(
                            r["learner_decisions"] for r in records(dest / f"{style}.jsonl")
                        )
            if model == "wide-transformer-full":
                episodes = records(cp.parent / "episodes.jsonl")
                sizes = [r["learner_decisions"] for r in episodes]
                end = 0
                for r in episodes:
                    end += r["learner_decisions"]
                    assert end == r["step"]
                if end < STEPS:
                    sizes.append(STEPS - end)
                assert (
                    sum(sizes) == STEPS
                    and len(eval_lengths) == len(rosters) * len(STYLES) * BATTLES
                )
                for phase, lengths in (("train", sizes), ("held_out", eval_lengths)):
                    assert min(lengths) >= 1
                    coverage.append(
                        {
                            "seed": seed,
                            "phase": phase,
                            "battle_prefixes": len(lengths),
                            "battles_past_eight": sum(n > 8 for n in lengths),
                            "decisions": sum(lengths),
                            "decisions_past_eight": sum(max(n - 8, 0) for n in lengths),
                            "longest_prefix": max(lengths),
                        }
                    )
            probe.extend(score_targets(cp, cases, condition))
    assert len(games) == len(SEEDS) * len(MODELS) * len(rosters) * len(STYLES) * BATTLES
    key = ("seed", "model", "roster", "opponent")
    with (PUBLISHED / "per-roster-opponent.csv").open() as f:
        archive_wins = {
            tuple(r[k] for k in key): int(r["win"])
            for r in csv.DictReader(f)
            if r["model"] in MODELS
        }
    assert {
        tuple(str(r[k]) for k in key): r["win"] for r in scores if r["model"] not in LARGE
    } == archive_wins
    summary, by_seed, by_roster, diffs = summarize_rosters(scores, games, rosters, MODELS, PAIRS)
    probed = [
        {
            "seed": s,
            **{
                m: mean(r["type_probe_pct"] for r in probe if r["seed"] == s and r["model"] == m)
                for m in MODELS
            },
        }
        for s in SEEDS
    ]
    probe_diffs = [
        {"left": left, "right": right, **interval([r[left] - r[right] for r in probed])}
        for left, right in PAIRS
    ]
    for row in summary:
        model = row["model"]
        row["type_probe_pct"] = mean(r[model] for r in probed)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("summary.csv", summary),
        ("per-seed.csv", by_seed),
        ("per-roster.csv", by_roster),
        ("paired-differences.csv", diffs),
        ("per-roster-opponent.csv", scores),
        ("evaluation-games.csv.gz", games),
        ("checkpoints.csv", provenance),
        ("type-probe.csv", probe),
        ("type-probe-per-seed.csv", probed),
        ("type-probe-differences.csv", probe_diffs),
        ("full-history-lengths.csv", coverage),
    ]:
        write_csv(out / name, rows)
    plot(out, by_seed, probed)
    print(
        json.dumps(
            {
                "win_rates_and_probe": summary,
                "battle_differences": diffs,
                "type_differences": probe_diffs,
            },
            indent=2,
        )
    )


def plot(out, by_seed, probed):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, rows, title, ylabel in zip(
        axes,
        (by_seed, probed),
        ("Seven held-out teams", "Type-sensitive targeting"),
        ("Win rate (%)", "Probability on higher-multiplier target (%)"),
        strict=True,
    ):
        for small, large, label, color in [
            ("feedforward", "wide-feedforward", "Feedforward", "#346d82"),
            ("chunks-1", "wide-transformer", "1-chunk Transformer", "#b35a32"),
            ("chunks-8", "wide-transformer-8", "8-chunk Transformer", "#4d864b"),
        ]:
            for r in rows:
                ax.plot([0, 1], [r[small], r[large]], color=color, alpha=0.2, lw=1)
            ax.plot(
                [0, 1],
                [mean(r[m] for r in rows) for m in (small, large)],
                marker="o",
                color=color,
                lw=2.5,
                label=label,
            )
        full = [r["wide-transformer-full"] for r in rows]
        ax.scatter([1.07] * len(full), full, color="#5e477b", alpha=0.35, s=12)
        ax.scatter(
            [1.07], [mean(full)], marker="D", color="#5e477b", s=46, label="Full-battle Transformer"
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Original", "Larger"],
            title=title,
            ylabel=ylabel,
            xlim=(-0.15, 1.15),
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(50, color="#666666", ls="--", lw=1)
    axes[0].legend(frameon=False)
    fig.savefig(out / "policy-size.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.runs.resolve(), args.output.resolve())
