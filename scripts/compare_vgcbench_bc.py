"""Compare equal-budget BC and league fine-tuning on the locked M-B benchmark."""

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from analyze_challenger_transfer import key, paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from evaluate_vgcbench_mb import FAMILIES, SEED_BASE, N
from opponent_diversity import ROOT, SEEDS
from report_vgcbench_mb import averaged_effect, export_policy
from train_vgcbench_mb import BC_RUN, RUN, TEAM

PRIOR = ROOT / "results/sketch5-vgcbench-mb"
OUT = ROOT / "results/sketch5-vgcbench-bc"
STEPS = (16384, 65536)


def load_games(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return list(csv.DictReader(stream))


def evaluate(seed, steps):
    cp = BC_RUN / f"seed-{seed}/train/checkpoint-{steps}.pt"
    dest = BC_RUN / f"eval-{steps}/seed-{seed}"
    cmd = [
        sys.executable,
        "scripts/evaluate_vgcbench_mb.py",
        "--checkpoint",
        str(cp),
        "--team",
        TEAM,
        "--out",
        str(dest),
        "--jobs",
        "6",
    ]
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "driver.log").open("a") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            [*cmd, "--verify"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def training_audit(seed, spec):
    root = BC_RUN / f"seed-{seed}/train"
    old = RUN / f"seed-{seed}/train"
    args = read(root / "manifest.json")["args"].copy()
    previous = read(old / "manifest.json")["args"].copy()
    assert args.pop("opponent") == "vgcbench-bc" and previous.pop("opponent") == "mixture"
    assert previous.pop("opponent_pool").endswith("training-pool.json")
    for x in (args, previous):
        x.pop("run_dir")
    assert args == previous and list(args["opponent_teams"]) == spec["training_teams"]
    assert spec["learner_team"] == args["team"] == TEAM
    assert read(root / "summary.json")["steps"] == 65536
    a, b = [
        torch.load(p / "checkpoint-0.pt", map_location="cpu", weights_only=True)["policy"]
        for p in (root, old)
    ]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    with (root / "episodes.jsonl").open() as stream:
        episodes = [json.loads(line) for line in stream]
    with (old / "episodes.jsonl").open() as stream:
        baseline = [json.loads(line) for line in stream]

    def schedule(r):
        return r["simulator_seed"], r["opponent_team_index"]

    shared = min(len(episodes), len(baseline))
    assert list(map(schedule, episodes[:shared])) == list(map(schedule, baseline[:shared]))
    assert {r["opponent"] for r in episodes} == {"vgcbench-bc"}
    assert {r["learner_team"] for r in episodes} == {TEAM}
    assert all(
        len(set(r["opponent_lineup"])) == len(r["opponent_lineup"]) == 4
        and set(r["opponent_lineup"]) <= set("123456")
        for r in episodes
    )
    sim_seeds = {r["simulator_seed"] for r in episodes}
    heldout_seeds = {
        SEED_BASE + (i * len(FAMILIES) + j) * 1000 + k
        for i in range(len(spec["teams"]))
        for j in range(len(FAMILIES))
        for k in range(N)
    }
    counts = Counter(r["opponent_team_index"] for r in episodes)
    windows = [
        dict(seed=seed, window=label, games=len(g), wins=sum(r["outcome"] == "win" for r in g))
        for label, g in (("first-512", episodes[:512]), ("last-512", episodes[-512:]))
    ]
    return dict(
        seed=seed,
        episodes=len(episodes),
        matched_training_schedule=shared,
        training_team_episodes={t: counts[i] for i, t in enumerate(spec["training_teams"])},
        distinct_ordered_bc_lineups={
            t: len({r["opponent_lineup"] for r in episodes if r["opponent_team_index"] == i})
            for i, t in enumerate(spec["training_teams"])
        },
        evaluation_seed_coincidences=sorted(sim_seeds & heldout_seeds),
    ), windows


def collect(steps, spec, prior):
    groups = {}
    new_games = []
    reproduced = []
    for seed in SEEDS:
        dest = BC_RUN / f"eval-{steps}/seed-{seed}"
        locked = read(dest / "checkpoints-0.json")["supplied-policy"]
        assert digest(BC_RUN / f"seed-{seed}/train/checkpoint-{steps}.pt") == locked["sha256"]
        settings = read(dest / "protocol.json")
        for field in (
            "format",
            "open_team_sheets",
            "teams",
            "families",
            "battles_per_cell",
            "seed_base",
            "evaluation_simulator",
            "model_sha256",
            "lineup",
        ):
            assert settings[field] == spec[field]
        label = f"seed-{seed}-step-{steps}"
        old = [r for r in prior if r["candidate"] == label]
        new = [dict(r, candidate=label) for r in load_games(dest / "games.csv.gz")]
        old_idx = {key(r): r for r in old}
        expected = len(spec["teams"]) * len(FAMILIES) * N
        assert len(old_idx) == len(old) == len(new) == len({key(r) for r in new}) == expected
        for r in new:
            ref = old_idx[key(r)]
            assert (
                r["lineup"] == ref["lineup"] == "1234"
                and r["opponent_lineup"] == ref["opponent_lineup"]
            )
            assert r["outcome"] in ("win", "loss", "tie", "cutoff")
        verified = load_games(dest / "reproduced-0.csv")
        assert len(verified) == 8
        reproduced.extend(dict(r, candidate=label) for r in verified)
        new_games.extend(new)
        groups[seed] = (new, old)
    return groups, new_games, reproduced


def compare(steps, spec, prior):
    groups, games, replays = collect(steps, spec, prior)
    effects, means = [], []
    for team in [None, *spec["teams"]]:
        scope = Path(team).stem if team else "external-12"
        for family in FAMILIES:
            pairs = []
            for seed, (new, old) in groups.items():
                a, b = (
                    [
                        r
                        for r in rows
                        if r["family"] == family and (not team or r["opponent_team"] == team)
                    ]
                    for rows in (new, old)
                )
                pairs.append((a, b))
                effects.append(
                    dict(
                        step=steps,
                        subset=scope,
                        family=family,
                        seed=seed,
                        bc_wins=score(a)["wins"],
                        league_wins=score(b)["wins"],
                        **paired(a, b),
                    )
                )
            denom = sum(len(a) for a, _ in pairs)
            means.append(
                dict(
                    step=steps,
                    subset=scope,
                    family=family,
                    bc_win_pct=100
                    * sum(r["outcome"] == "win" for a, _ in pairs for r in a)
                    / denom,
                    league_win_pct=100
                    * sum(r["outcome"] == "win" for _, b in pairs for r in b)
                    / denom,
                    **averaged_effect(pairs),
                )
            )
    return games, replays, effects, means


def plot_final(effects, means):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]}
    )
    rows = [r for r in effects if r["subset"] == "external-12" and r["family"] == "bc-learned"]
    for r, color in zip(rows, ("#155d8b", "#b95e27", "#267e61", "#80589a"), strict=True):
        rates = [100 * r[field] / r["battles_each"] for field in ("league_wins", "bc_wins")]
        left.plot([0, 1], rates, marker="o", color=color, linewidth=2, label=f"Seed {r['seed']}")
    left.set(
        xticks=[0, 1],
        xticklabels=["League", "Behavior cloning"],
        xlim=(-0.18, 1.18),
        xlabel="Training opponent",
        ylabel="Win rate against BC (%)",
        title="All 12 held-out teams",
    )
    left.set_ylim(0, 100)
    left.grid(axis="y", alpha=0.2)
    left.legend(fontsize=8, loc="best")
    teams = sorted(
        (r for r in means if r["family"] == "bc-learned" and r["subset"] != "external-12"),
        key=lambda r: r["delta_pp"],
    )
    right.axvline(0, linestyle=":", color="#777", linewidth=1)
    for i, r in enumerate(teams):
        right.plot([r["low95_pp"], r["high95_pp"]], [i, i], color="#6e8b9b", linewidth=1.6)
        right.plot(r["delta_pp"], i, "o", color="#12556f", markersize=4)
    right.set(
        yticks=range(len(teams)),
        yticklabels=[r["subset"] for r in teams],
        title="Effect of BC training by opposing team",
        xlabel="Paired win-rate change (percentage points)",
    )
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "paired-bc-crossplay.png", dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["evaluate", "report"])
    p.add_argument("--steps", choices=STEPS, type=int, default=65536)
    args = p.parse_args()
    if args.stage == "evaluate":
        with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
            list(pool.map(evaluate, SEEDS, [args.steps] * len(SEEDS)))
        return
    spec = read(BC_RUN / "protocol.json")
    assert not set(spec["training_teams"]) & set(spec["teams"])
    assert all(digest(path) == h for path, h in spec["sha256"].items())
    prior = load_games(PRIOR / "games.csv.gz")
    OUT.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(BC_RUN / "protocol.json", OUT / "protocol.json")
    (OUT / "weights").mkdir(exist_ok=True)
    data = [compare(step, spec, prior) for step in STEPS]
    plot_final(data[-1][2], data[-1][3])
    for name, parts in zip(
        ("games.csv.gz", "reproduced.csv", "paired-by-seed.csv", "mean-changes.csv"),
        zip(*data, strict=True),
        strict=True,
    ):
        write_csv(OUT / name, [r for part in parts for r in part])
    audit = []
    windows = []
    weights = []
    for seed in SEEDS:
        row, segments = training_audit(seed, spec)
        audit.append(row)
        windows.extend(segments)
        weights.append(export_policy(BC_RUN / f"seed-{seed}/train", OUT, seed))
    write_csv(OUT / "training-windows.csv", windows)
    write_csv(OUT / "weights.csv", weights)
    all_games = [r for part in data for r in part[0]]
    (OUT / "audit.json").write_text(
        json.dumps(
            dict(
                new_benchmark_games=len(all_games),
                baseline_games_sha256=digest(PRIOR / "games.csv.gz"),
                outcomes=Counter(r["outcome"] for r in all_games),
                matched_opposing_lineups=True,
                replayed_records=sum(len(part[1]) for part in data),
                native_defaults=sum(int(r["opponent_defaults"]) for r in all_games),
                training=audit,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
