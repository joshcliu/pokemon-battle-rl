"""Audit matched PPO runs and quantify adaptation, transfer and retention on Sun."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import torch
from analyze_long_rewards import load, settings, write_csv
from analyze_rewards import records
from analyze_roster_transfer import OPPONENT_TEAMS, OUTCOMES, same_runtime
from finetune_rosters import (
    MODES,
    SCRATCH_STUDY,
    STAGES,
    SUN,
    SUN_STUDY,
    checkpoint,
    evaluation,
    previous_policy,
)
from opponent_diversity import ROOT, SEEDS, STEPS, verified_mixture
from roster_transfer import BATTLES, EVAL_SEED, STYLES, split

GROUPS = ("practice", "held_out", "sun")
PRIOR = ROOT / "results/sketch5-roster-transfer"


def weights(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def identical_weights(a, b):
    x, y = weights(a)["policy"], weights(b)["policy"]
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def training_audit(root, seed, sun):
    archive = SCRATCH_STUDY / f"seed-{seed}/train"
    ref = load(archive / "manifest.json")
    past_eps = records(archive / "episodes.jsonl")
    provenance, coverage = [], []
    with (PRIOR / "checkpoints.csv").open() as f:
        old_hash = {r["policy"]: r["sha256"] for r in csv.DictReader(f) if int(r["seed"]) == seed}
    for tag, p in [
        ("sun", sun),
        ("untrained", archive / "checkpoint-0.pt"),
        ("tournament", archive / "checkpoint.pt"),
    ]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == old_hash[tag]
    for mode in MODES:
        directory = root / f"seed-{seed}" / mode / "train"
        manifest = load(directory / "manifest.json")
        same_runtime(manifest, ref)
        a = manifest["args"]
        skip = {"run_dir", "init_checkpoint", "save_at", "steps"}
        assert settings(a, skip) == settings(ref["args"], skip)
        assert a["learner_teams"] == split()["train"] and a["opponent_teams"] == OPPONENT_TEAMS
        assert a["init_checkpoint"] == (str(sun) if mode == "warm" else None)
        assert identical_weights(
            directory / "checkpoint-0.pt", sun if mode == "warm" else archive / "checkpoint-0.pt"
        )
        end = STEPS if mode == "warm" else STAGES[1]
        summary = load(directory / "summary.json")
        assert a["steps"] == summary["steps"] == end
        current = records(directory / "episodes.jsonl")
        assert len(current) == summary["episodes"]
        for old, new in zip(past_eps, current):
            for k in (
                "episode",
                "opponent",
                "learner_team",
                "opponent_team_index",
                "simulator_seed",
            ):
                assert old[k] == new[k]
        if mode == "scratch":
            assert current == past_eps[: len(current)]
            orig = records(archive / "metrics.jsonl")
            rep = records(directory / "metrics.jsonl")
            assert rep[-1]["steps"] == end
            assert len(rep) == end // a["rollout_steps"]
            for old, new in zip(orig, rep):
                assert settings(old, {"steps_per_second"}) == settings(new, {"steps_per_second"})
        for stage in (0, *STAGES):
            p = checkpoint(root, seed, mode, stage)
            saved = weights(p)
            assert saved["steps"] == stage and saved["seed"] == seed
            provenance.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "tournament_steps": stage,
                    "lifetime_steps": stage + (STEPS if mode == "warm" else 0),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "training_commit": ref["git_commit"]
                    if mode == "scratch" and stage == STEPS
                    else manifest["git_commit"],
                }
            )
        full = current if mode == "warm" else past_eps
        for stage in STAGES:
            sampled = Counter(r["learner_team"] for r in full if r["step"] <= stage)
            assert set(sampled) == set(split()["train"])
            coverage.extend(
                {
                    "seed": seed,
                    "mode": mode,
                    "tournament_steps": stage,
                    "roster": Path(p).stem,
                    "episodes": n,
                }
                for p, n in sorted(sampled.items())
            )
    return provenance, coverage


def score_run(root, seed, mode, step, roster, group, manifest, schedule, sun):
    dest = evaluation(root, seed, mode, step, roster)
    src = checkpoint(root, seed, mode, step)
    if previous_policy(mode, step) and group != "practice" and step == 0:
        src = sun if mode == "warm" else SCRATCH_STUDY / f"seed-{seed}/train/checkpoint-0.pt"
    condition = {"seed": seed, "mode": mode, "tournament_steps": step, "group": group}
    return score_checkpoint(dest, src, roster, manifest, schedule, condition)


def score_checkpoint(dest, src, roster, manifest, schedule, condition):
    """Check evaluation provenance and paired battle draws before counting wins."""
    ev = load(dest / "manifest.json")
    same_runtime(ev, manifest)
    a = ev["args"]
    assert (
        a["team"] == roster
        and a["opponent_teams"] == OPPONENT_TEAMS
        and a["opponents"] == list(STYLES)
    )
    assert (a["battles"], a["seed"], a["deterministic"]) == (BATTLES, EVAL_SEED, False)
    assert Path(a["checkpoint"]).resolve() == src
    assert not (dest / "errors.jsonl").exists()
    scores, games = [], []
    summary = load(dest / "summary.json")
    for style in STYLES:
        rows = records(dest / f"{style}.jsonl")
        assert [r["seed"] for r in rows] == list(range(EVAL_SEED, EVAL_SEED + BATTLES))
        counts = Counter(r["outcome"] for r in rows)
        assert counts == Counter(summary[style]["outcomes"]) and set(counts) <= set(OUTCOMES)
        key = condition | {"roster": Path(roster).stem, "opponent": style}
        scores.append(
            key
            | {o: counts[o] for o in OUTCOMES}
            | {"battles": len(rows), "win_pct": 100 * counts["win"] / len(rows)}
        )
        for r in rows:
            assert r["learner_team"] == roster and r["opponent"] == style
            pair = r["simulator_seed"], r["opponent_team_index"]
            assert schedule.setdefault(r["seed"], pair) == pair
            games.append(
                key
                | {
                    "eval_seed": r["seed"],
                    "simulator_seed": pair[0],
                    "opponent_team": Path(OPPONENT_TEAMS[pair[1]]).stem,
                    "outcome": r["outcome"],
                    "turns": r["turn"],
                }
            )
    return scores, games


def interval(values):
    # Pointwise paired t interval for the mean across training seeds, df = 3.
    assert len(values) == 4
    center = mean(values)
    radius = 3.182446305284263 * stdev(values) / sqrt(len(values))
    return {"mean_pp": center, "low95_pp": center - radius, "high95_pp": center + radius}


def analyze(root, out):
    torch.set_num_threads(1)
    prior = verified_mixture(SUN_STUDY)
    teams = split()
    pools = {"practice": teams["train"], "held_out": teams["test"], "sun": [SUN]}
    all_scores, games, provenance, coverage = [], [], [], []
    values, schedule = {}, {}
    for seed in SEEDS:
        info, draws = training_audit(root, seed, prior[seed])
        provenance.extend(info)
        coverage.extend(draws)
        ref = load(SCRATCH_STUDY / f"seed-{seed}/train/manifest.json")
        for step in (0, *STAGES):
            for mode in MODES:
                for group, rs in pools.items():
                    scores = []
                    for roster in rs:
                        scored, played = score_run(
                            root, seed, mode, step, roster, group, ref, schedule, prior[seed]
                        )
                        scores.extend(scored)
                        games.extend(played)
                    all_scores.extend(scores)
                    values[seed, step, mode, group] = mean(r["win_pct"] for r in scores)
    with (PRIOR / "per-seed.csv").open() as f:
        archived = list(csv.DictReader(f))
    for r in archived:
        mode, step = {
            "sun": ("warm", 0),
            "untrained": ("scratch", 0),
            "tournament": ("scratch", STEPS),
        }[r["policy"]]
        for group in ("sun", "held_out"):
            assert abs(values[int(r["seed"]), step, mode, group] - float(r[group])) < 1e-10
    means, paired, seed_rows = [], [], []
    for step in (0, *STAGES):
        for group in GROUPS:
            for mode in MODES:
                ys = [values[s, step, mode, group] for s in SEEDS]
                means.append(
                    {
                        "tournament_steps": step,
                        "mode": mode,
                        "group": group,
                        "mean_win_pct": mean(ys),
                        "seed_sd_pp": stdev(ys),
                    }
                )
                seed_rows.extend(
                    {
                        "seed": s,
                        "tournament_steps": step,
                        "mode": mode,
                        "group": group,
                        "win_pct": v,
                    }
                    for s, v in zip(SEEDS, ys, strict=True)
                )
            gaps = [
                values[s, step, "warm", group] - values[s, step, "scratch", group] for s in SEEDS
            ]
            diffs = [values[s, step, "warm", group] - values[s, 0, "warm", group] for s in SEEDS]
            for label, vals in [("warm_minus_scratch", gaps), ("warm_minus_own_start", diffs)]:
                paired.append(
                    {
                        "tournament_steps": step,
                        "group": group,
                        "comparison": label,
                        **{f"seed_{s}": v for s, v in zip(SEEDS, vals, strict=True)},
                        **interval(vals),
                    }
                )
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("learning-curve.csv", means),
        ("paired-differences.csv", paired),
        ("per-seed.csv", seed_rows),
        ("per-roster-opponent.csv", all_scores),
        ("evaluation-games.csv.gz", games),
        ("training-draws.csv", coverage),
        ("checkpoints.csv", provenance),
    ]:
        write_csv(out / name, rows)
    total = Counter(r["outcome"] for r in games)
    (out / "outcomes.json").write_text(json.dumps(total, indent=2) + "\n")
    plot(values, out)
    for row in means:
        print(row, flush=True)
    print(total, flush=True)


def plot(values, out):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, group, title in zip(
        axes, GROUPS, ["15 practice rosters", "7 held-out rosters", "Sun retention"], strict=True
    ):
        for mode, label, color in [
            ("warm", "Sun checkpoint", "#90451e"),
            ("scratch", "From scratch", "#306681"),
        ]:
            for s in SEEDS:
                yy = [values[s, t, mode, group] for t in (0, *STAGES)]
                ax.plot(range(4), yy, color=color, alpha=0.19, lw=0.8)
            avg = [mean(values[s, t, mode, group] for s in SEEDS) for t in (0, *STAGES)]
            ax.plot(range(4), avg, color=color, marker="o", lw=2, label=label)
        ax.set(title=title, xticks=range(4), xticklabels=["0", "16k", "65k", "262k"], ylim=(0, 100))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Win rate (%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.supxlabel("Additional tournament-training decisions (checkpoint)", fontsize=10)
    fig.savefig(out / "finetuning-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analyze(args.runs.resolve(), args.output.resolve())
