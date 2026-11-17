"""Audit held-out games and compare decision histories across paired training seeds."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from analyze_long_rewards import load, settings, write_csv
from analyze_rewards import records
from analyze_roster_finetuning import identical_weights, interval, score_checkpoint, weights
from analyze_roster_transfer import OPPONENT_TEAMS, OUTCOMES, same_runtime
from opponent_diversity import ROOT, SEEDS
from roster_transfer import BATTLES, STYLES, split
from transformer_rosters import STEPS, WIDTH

MODELS = ("feedforward", "chunks-1", "chunks-8")
ARCHIVE = ROOT / "runs/sketch5-finetune-v1"
PREVIOUS = ROOT / "results/sketch5-roster-finetuning"


def archived_rows(filename):
    with (PREVIOUS / filename).open() as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["mode"] == "scratch" and int(r["tournament_steps"]) == STEPS
        ]


def audit(root, seed, teams, archive_hash):
    archive = ARCHIVE / f"seed-{seed}/scratch/train"
    ref = load(archive / "manifest.json")
    previous = records(archive / "episodes.jsonl")
    check = root / f"seed-{seed}"
    assert identical_weights(*(check / m / "train/checkpoint-0.pt" for m in MODELS[1:]))
    entries = []
    for model, chunks in zip(MODELS, (None, 1, 8), strict=True):
        train = check / model / "train" if chunks else archive
        checkpoint = train / ("checkpoint.pt" if chunks else f"checkpoint-{STEPS}.pt")
        saved = weights(checkpoint)
        assert saved["steps"] == STEPS and saved["seed"] == seed
        assert saved.get("transformer_chunks") == chunks
        assert saved["hidden_size"] == (WIDTH if chunks else 128)
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if chunks is None:
            assert sha == archive_hash
        manifest = load(train / "manifest.json")
        same_runtime(manifest, ref)
        args = manifest["args"]
        skip = {"run_dir", "save_at", "hidden_size", "transformer_chunks"}
        assert settings(args, skip) == settings(ref["args"], skip)
        assert args["learner_teams"] == teams["train"] and args["opponent_teams"] == OPPONENT_TEAMS
        assert args["opponent"] == "mixture" and args["reward_mode"] == "win-loss"
        assert args["init_checkpoint"] is None and args["steps"] == STEPS
        summary = load(train / "summary.json")
        episodes = records(train / "episodes.jsonl")
        assert summary["steps"] == STEPS and summary["episodes"] == len(episodes)
        assert set(x["learner_team"] for x in episodes) == set(teams["train"])
        for old, new in zip(previous, episodes):
            for k in (
                "episode",
                "simulator_seed",
                "opponent",
                "learner_team",
                "opponent_team_index",
            ):
                assert old[k] == new[k]
        entries.append(
            {
                "seed": seed,
                "model": model,
                "sha256": sha,
                "source": manifest["git_commit"],
                "parameters": sum(t.numel() for t in saved["policy"].values()),
                "steps": summary["steps"],
                "training_episodes": len(episodes),
            }
        )
    assert entries[1]["parameters"] == entries[2]["parameters"]
    return ref, entries


def analyze(root, out):
    torch.set_num_threads(1)
    teams = split()
    assert (len(teams["train"]), len(teams["test"])) == (15, 7)
    prior_hash = {int(r["seed"]): r["sha256"] for r in archived_rows("checkpoints.csv")}
    schedule, scores, games, metadata = {}, [], [], []
    for seed in SEEDS:
        ref, entries = audit(root, seed, teams, prior_hash[seed])
        metadata.extend(entries)
        for model in MODELS:
            run = root / f"seed-{seed}"
            label = model
            cp = run / model / "train/checkpoint.pt"
            if model == "feedforward":
                run = ARCHIVE / f"seed-{seed}"
                label = f"scratch-{STEPS}"
                cp = run / f"scratch/train/checkpoint-{STEPS}.pt"
            for roster in teams["test"]:
                dest = run / "eval" / label / Path(roster).stem / "games"
                rows, played = score_checkpoint(
                    dest, cp, roster, ref, schedule, {"seed": seed, "model": model}
                )
                scores.extend(rows)
                games.extend(played)
    assert len(games) == len(SEEDS) * len(MODELS) * 7 * len(STYLES) * BATTLES
    previous = {
        (int(r["seed"]), r["roster"], r["opponent"]): int(r["win"])
        for r in archived_rows("per-roster-opponent.csv")
        if r["group"] == "held_out"
    }
    assert {
        (r["seed"], r["roster"], r["opponent"]): r["win"]
        for r in scores
        if r["model"] == "feedforward"
    } == previous

    def win_pct(model, **condition):
        return mean(
            r["win_pct"]
            for r in scores
            if r["model"] == model and all(r[k] == v for k, v in condition.items())
        )

    by_seed = [{"seed": s, **{m: win_pct(m, seed=s) for m in MODELS}} for s in SEEDS]
    by_roster = [
        {"roster": Path(t).stem, **{m: win_pct(m, roster=Path(t).stem) for m in MODELS}}
        for t in teams["test"]
    ]
    overall = []
    for model in MODELS:
        outcomes = Counter(r["outcome"] for r in games if r["model"] == model)
        overall.append(
            {
                "model": model,
                "mean_win_pct": win_pct(model),
                "battles": sum(outcomes.values()),
                **{o: outcomes[o] for o in OUTCOMES},
                **{f"{s}_win_pct": win_pct(model, opponent=s) for s in STYLES},
            }
        )
    contrasts = []
    for left, right in [
        ("chunks-8", "chunks-1"),
        ("chunks-8", "feedforward"),
        ("chunks-1", "feedforward"),
    ]:
        diffs = [r[left] - r[right] for r in by_seed]
        contrasts.append(
            {
                "left": left,
                "right": right,
                **interval(diffs),
                **{f"seed_{s}": d for s, d in zip(SEEDS, diffs, strict=True)},
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("summary.csv", overall),
        ("per-seed.csv", by_seed),
        ("per-roster.csv", by_roster),
        ("paired-differences.csv", contrasts),
        ("per-roster-opponent.csv", scores),
        ("evaluation-games.csv.gz", games),
        ("checkpoints.csv", metadata),
    ]:
        write_csv(out / name, rows)
    plot(out, by_seed, by_roster)
    print(json.dumps({"results": overall, "contrasts": contrasts}, indent=2))


def plot(out, by_seed, by_roster):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for row in by_seed:
        left.plot(range(3), [row[m] for m in MODELS], marker="o", color="#708398", alpha=0.5, lw=1)
    left.plot(
        range(3), [mean(r[m] for r in by_seed) for m in MODELS], marker="o", lw=3, color="#173e54"
    )
    left.set(
        xticks=range(3),
        xticklabels=["Feedforward", "1 chunk", "8 chunks"],
        ylabel="Held-out win rate (%)",
        title="Four training seeds and their mean",
    )
    deltas = sorted((r["chunks-8"] - r["chunks-1"], r["roster"]) for r in by_roster)
    for i, (delta, _) in enumerate(deltas):
        right.barh(i, delta, color="#37788b" if delta >= 0 else "#b96638")
    right.axvline(0, color="#41464b", lw=1)
    right.set(
        yticks=range(7),
        yticklabels=[name for _, name in deltas],
        xlabel="8 chunks minus 1 chunk (percentage points)",
        title="Effect of history by held-out team",
    )
    for ax in (left, right):
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(out / "held-out-comparison.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    analyze(args.runs.resolve(), args.output.resolve())
