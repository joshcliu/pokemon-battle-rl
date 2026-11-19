"""Audit all matched M-B games and summarize the win-rate change across training seeds."""

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from analyze_challenger_transfer import key, paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_vgcbench_mb import CHAMPION, FAMILIES, OUT, SEED_BASE, STAGES, N, candidates
from opponent_diversity import SEEDS
from train_vgcbench_mb import FORMAT, RUN, TEAM


def collect(spec):
    records = defaultdict(list)
    previews = defaultdict(set)
    scopes = {}
    for step in STAGES:
        models = candidates(step)
        locked = read(RUN / f"checkpoints-{step}.json")
        assert set(locked) == set(models)
        for label, path in models.items():
            assert digest(path) == locked[label]["sha256"]
            scopes[label] = set()
            for i, team in enumerate(spec["teams"]):
                for j, family in enumerate(FAMILIES):
                    base = SEED_BASE + (4 * i + j) * 1000
                    for start in range(0, N, 8):
                        with (
                            RUN / "crossplay" / label / Path(team).stem / f"{family}-{start}.csv"
                        ).open() as stream:
                            batch = list(csv.DictReader(stream))
                        assert len(batch) == 8
                        for k, r in enumerate(batch, start):
                            side = "p1" if k % 2 == 0 else "p2"
                            assert (
                                r["candidate"],
                                r["opponent_team"],
                                r["roster"],
                                r["family"],
                                int(r["seed"]),
                                r["side"],
                            ) == (label, team, Path(team).stem, family, base + k, side)
                            assert (
                                r["outcome"] in ("win", "loss", "tie", "cutoff")
                                and r["lineup"] == "1234"
                            )
                            code = r["opponent_lineup"]
                            assert len(set(code)) == len(code) == 4 and set(code) <= set("123456")
                            previews[key(r)].add(code)
                            records[label].append(r)
                            scopes[label].add(key(r))
    assert all(len(v) == 1 for v in previews.values())
    assert all(
        v == scopes[CHAMPION] and len(v) == len(spec["teams"]) * len(FAMILIES) * N
        for v in scopes.values()
    )
    return records


def averaged_effect(pairs):
    """Average four model differences per battle before estimating stratified variance."""
    by_match = defaultdict(list)
    for first, second in pairs:
        ref = {key(r): r["outcome"] == "win" for r in second}
        for r in first:
            by_match[key(r)].append(int(r["outcome"] == "win") - int(ref[key(r)]))
    strata = defaultdict(list)
    for k, changes in by_match.items():
        assert len(changes) == len(SEEDS)
        strata[k[:3] + (k[-1],)].append(sum(changes) / len(changes))
    n = len(by_match)
    delta = 100 * sum(map(sum, strata.values())) / n
    variance = sum(
        (len(v) * sum(x * x for x in v) - sum(v) ** 2) / (len(v) - 1) for v in strata.values()
    )
    margin = 1.959963984540054 * 100 * math.sqrt(variance) / n
    return dict(
        paired_battle_seeds=n, delta_pp=delta, low95_pp=delta - margin, high95_pp=delta + margin
    )


def plot_crossplay(scores, average):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(r["candidate"], r["family"]): r for r in scores if r["subset"] == "external-12"}
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]}
    )
    colors = ["#156a9c", "#bd5d29", "#278061", "#7a5791"]
    for seed, color in zip(SEEDS, colors, strict=True):
        values = [lookup[f"seed-{seed}-step-{step}", "bc-learned"]["win_pct"] for step in STAGES]
        left.plot(STAGES, values, color=color, marker="o", label=f"Seed {seed}", linewidth=1.8)
    left.axhline(
        lookup[CHAMPION, "bc-learned"]["win_pct"],
        color="#687078",
        linestyle=":",
        label="M-C champion on variant",
    )
    left.set(
        title="Against VGC-Bench behavior cloning",
        ylabel="Wins (%)",
        xlabel="M-B fine-tuning decisions",
        xticks=STAGES,
        xticklabels=["0", "16,384", "65,536"],
        ylim=(0, 100),
    )
    left.grid(axis="y", alpha=0.18)
    left.legend(fontsize=8, loc="best")
    matchups = sorted(
        (
            r
            for r in average
            if r["family"] == "bc-learned" and r["step"] == 65536 and r["subset"] != "external-12"
        ),
        key=lambda r: r["delta_pp"],
    )
    for pos, r in enumerate(matchups):
        right.plot([r["low95_pp"], r["high95_pp"]], [pos, pos], color="#6b8998", linewidth=1.5)
        right.plot(r["delta_pp"], pos, "o", color="#15536f", markersize=4)
    right.axvline(0, color="#687078", linestyle=":", linewidth=1)
    right.set(
        yticks=range(len(matchups)),
        yticklabels=[r["subset"] for r in matchups],
        xlabel="Win-rate change (percentage points)",
        title="Paired change from 0 to 65,536 decisions",
    )
    for panel in (left, right):
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "external-crossplay.png", dpi=175)
    plt.close(fig)


def export_policy(root, out, seed):
    cp = root / "checkpoint-65536.pt"
    exported = out / f"weights/seed-{seed}.pt"
    checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
    del checkpoint["optimizer"]
    torch.save(checkpoint, exported)
    restored = torch.load(exported, map_location="cpu", weights_only=True)
    assert all(torch.equal(v, restored["policy"][k]) for k, v in checkpoint["policy"].items())
    return dict(
        seed=seed,
        source_sha256=digest(cp),
        weights=str(exported.relative_to(out)),
        sha256=digest(exported),
    )


def report():
    spec = read(RUN / "protocol.json")
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    records = collect(spec)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUN / "protocol.json", OUT / "protocol.json")
    for step in STAGES:
        shutil.copyfile(RUN / f"checkpoints-{step}.json", OUT / f"checkpoints-{step}.json")
        shutil.copyfile(RUN / f"reproduced-{step}.csv", OUT / f"reproduced-{step}.csv")
    all_games = [r for g in records.values() for r in g]
    write_csv(OUT / "games.csv.gz", all_games)
    scores, changes, average = [], [], []
    for team in [None, *spec["teams"]]:
        scope = Path(team).stem if team else "external-12"
        for family in FAMILIES:
            g = {
                label: [
                    r
                    for r in pool
                    if r["family"] == family and (team is None or r["opponent_team"] == team)
                ]
                for label, pool in records.items()
            }
            for label, rows in g.items():
                scores.append(dict(subset=scope, family=family, candidate=label, **score(rows)))
            for stage in STAGES[1:]:
                pairs = []
                for s in SEEDS:
                    label = f"seed-{s}-step-{stage}"
                    ref = f"seed-{s}-step-0"
                    changes.append(
                        dict(
                            subset=scope,
                            family=family,
                            candidate=label,
                            reference=ref,
                            **paired(g[label], g[ref]),
                        )
                    )
                    pairs.append((g[label], g[ref]))
                average.append(
                    dict(subset=scope, family=family, step=stage, **averaged_effect(pairs))
                )
    write_csv(OUT / "win-rates.csv", scores)
    write_csv(OUT / "paired-changes.csv", changes)
    write_csv(OUT / "mean-changes.csv", average)
    plot_crossplay(scores, average)
    training = []
    training_windows = []
    simulator_seeds = {int(r["seed"]) for r in all_games}
    overlaps = {}
    exports = []
    (OUT / "weights").mkdir(exist_ok=True)
    for seed in SEEDS:
        root = RUN / f"seed-{seed}/train"
        args = read(root / "manifest.json")["args"]
        assert (
            args["format"],
            args["team"],
            args["steps"],
            args["seed"],
            args["open_team_sheets"],
            args["transformer_chunks"],
        ) == (FORMAT, TEAM, 65536, seed + 110, True, "full")
        assert read(root / "summary.json")["steps"] == 65536
        exports.append(export_policy(root, OUT, seed))
        source = torch.load(args["init_checkpoint"], map_location="cpu", weights_only=True)
        initial = torch.load(root / "checkpoint-0.pt", map_location="cpu", weights_only=True)
        assert all(
            torch.equal(value, initial["policy"][key]) for key, value in source["policy"].items()
        )
        with (root / "episodes.jsonl").open() as stream:
            episodes = [json.loads(line) for line in stream]
        assert len(episodes) > 1024
        for window, group in (("first-512", episodes[:512]), ("last-512", episodes[-512:])):
            wins = sum(r["outcome"] == "win" for r in group)
            training_windows.append(
                dict(
                    seed=seed,
                    window=window,
                    episodes=len(group),
                    wins=wins,
                    win_pct=100 * wins / len(group),
                )
            )
        parent_log = Path(args["init_checkpoint"]).parent / "episodes.jsonl"
        with parent_log.open() as stream:
            parent_seeds = {json.loads(line)["simulator_seed"] for line in stream}
        seen = parent_seeds | {r["simulator_seed"] for r in episodes}
        overlaps[str(seed)] = sorted(seen & simulator_seeds)
        with (root / "metrics.jsonl").open() as stream:
            for line in stream:
                metric = json.loads(line)
                training.append(
                    dict(
                        seed=seed,
                        steps=metric["steps"],
                        episodes=metric["episodes"],
                        wins=metric["outcomes"].get("win", 0),
                        losses=metric["outcomes"].get("loss", 0),
                    )
                )
    write_csv(OUT / "training-curves.csv", training)
    write_csv(OUT / "weights.csv", exports)
    write_csv(OUT / "training-windows.csv", training_windows)
    audit = dict(
        games=len(all_games),
        outcomes=Counter(r["outcome"] for r in all_games),
        native_defaults=sum(int(r["opponent_defaults"]) for r in all_games),
        replicated_games=3 * 8,
        matched_opponent_previews=True,
        evaluation_seed_coincidences_with_training=overlaps,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    report()
