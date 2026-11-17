"""Audit locked league games, report stratified uncertainty and export inference weights."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from fixed_league import FAMILIES
from screen_league import lock
from train_fixed_league import ARCHIVE, ROOT, ROSTERS

from pokemon_rl.experiments import wilson


def games(directory):
    with gzip.open(directory / "games.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def interval(rows):
    # Equal cell and side counts by design. Resample games within those fixed strata.
    strata = defaultdict(list)
    for r in rows:
        strata[r["roster"], r["family"], r["side"]].append(r["outcome"] == "win")
    rng = np.random.default_rng(37899)
    draws = np.zeros(20000)
    for _, xs in sorted(strata.items()):
        draws += rng.binomial(len(xs), np.mean(xs), len(draws))
    n = len(rows)
    w = sum(r["outcome"] == "win" for r in rows)
    lo, hi = 100 * np.quantile(draws / n, [0.025, 0.975])
    return {
        "wins": w,
        "games": n,
        "win_pct": 100 * w / n,
        "low95_pct": lo,
        "high95_pct": hi,
        "pooled_wilson_low95_pct": 100 * wilson(w, n)[0],
        "pooled_wilson_high95_pct": 100 * wilson(w, n)[1],
    }


def export_weights(src, out):
    raw = torch.load(ROOT / src, weights_only=True, map_location="cpu")
    saved = {
        k: raw[k]
        for k in (
            "policy",
            "obs_size",
            "hidden_size",
            "transformer_chunks",
            "steps",
            "seed",
            "format",
            "feature_schema",
        )
    }
    torch.save(saved, out)
    copy = torch.load(out, weights_only=True, map_location="cpu")
    assert raw["policy"].keys() == copy["policy"].keys() and all(
        torch.equal(x, copy["policy"][k]) for k, x in raw["policy"].items()
    )
    return {
        "source": src,
        "source_sha256": hashlib.sha256((ROOT / src).read_bytes()).hexdigest(),
        "export": out.name,
        "export_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "parameters": sum(x.numel() for x in raw["policy"].values()),
        "width": raw["hidden_size"],
        "history": raw["transformer_chunks"] or "current",
        "checkpoint_steps": raw["steps"],
    }


def report(root, valid, final, out, public):
    spec = json.loads((valid / "locked.json").read_text())
    assert spec == lock(
        root,
        spec["model"],
        spec["label"],
        spec["team"],
        spec["validation"],
        spec["confirmation_base"],
    )
    rows = games(final)
    n = spec["games_per_cell"]
    expected = {}
    for ri, t in enumerate(ROSTERS):
        for fi, family in enumerate(FAMILIES):
            for i in range(n):
                es = spec["confirmation_base"] + (ri * len(FAMILIES) + fi) * 10000 + i
                expected[es] = (
                    Path(t).stem,
                    family,
                    "p1" if i % 2 == 0 else "p2",
                    t,
                    spec["pool"][t][fi],
                )
    single = json.loads((valid / "locked-single.json").read_text())
    assert single == lock(
        root,
        single["model"],
        single["label"],
        spec["team"],
        spec["validation"],
        spec["confirmation_base"],
    )
    single_dir = final.with_name(final.name + "-single")
    single_rows = games(single_dir)
    for matchups, entry in ((rows, spec), (single_rows, single)):
        assert len(matchups) == len(expected) == len({int(r["seed"]) for r in matchups})
        assert all(
            expected[int(r["seed"])]
            == tuple(r[k] for k in ("roster", "family", "side", "opponent_team", "opponent"))
            for r in matchups
        )
        assert all(
            r["candidate"] == entry["label"]
            and r["learner"] == entry["model"]
            and r["team"] == entry["team"]
            for r in matchups
        )
        assert all(r["outcome"] in ("win", "loss", "cutoff", "tie") for r in matchups)
    for directory in [root / "selection", root / "calibration", valid, *root.glob("screen-*")]:
        if (directory / "games.csv.gz").exists():
            assert set(expected).isdisjoint(int(r["seed"]) for r in games(directory))
    # Confirm physical simulator seeds were unused by training or any development game.
    actual_seeds = set(expected)
    logs = [
        *root.glob("*/*/train/episodes.jsonl"),
        *ARCHIVE.glob("seed-*/fixed/train/episodes.jsonl"),
    ]
    for log in logs:
        with log.open() as f:
            assert actual_seeds.isdisjoint(
                json.loads(line)["simulator_seed"] for line in f if line.endswith("\n")
            )
    out.mkdir(parents=True, exist_ok=False)
    assets = out / "weights"
    assets.mkdir()
    summaries = [{"group": "all", **interval(rows)}]
    summaries.append({"group": "single-network", **interval(single_rows)})
    for field in ["family", "roster"]:
        for val in sorted({r[field] for r in rows}):
            summaries.append({"group": val, **interval([r for r in rows if r[field] == val])})
    write_csv(out / "confirmation-summary.csv", summaries)
    for directory, stem in [
        (final, "confirmation"),
        (single_dir, "single"),
        (valid, "validation"),
        (root / "calibration", "calibration"),
        (root / "selection", "rival-selection"),
        (root / "screen-archive", "baseline"),
    ]:
        shutil.copyfile(directory / "scores.csv", out / (stem + "-scores.csv"))
        shutil.copyfile(directory / "games.csv.gz", out / (stem + "-games.csv.gz"))
    shutil.copyfile(valid / "locked.json", out / "locked.json")
    shutil.copyfile(valid / "locked-single.json", out / "locked-single.json")
    shutil.copyfile(valid / "models.json", out / "validation-models.json")
    all_screen = []
    for directory in sorted(root.glob("screen-*")):
        if not (directory / "scores.csv").is_file():
            continue
        with (directory / "scores.csv").open() as f:
            all_screen.extend({"group": directory.name, **r} for r in csv.DictReader(f))
    write_csv(out / "development-scores.csv", all_screen)
    model = spec["model"]
    paths = [model] if model.endswith(".pt") else []
    if not paths:
        router = json.loads((ROOT / model).read_text())
        paths = sorted(set([*router["routes"].values(), router["default"]]))
    exported = {}
    provenance = []
    for i, p in enumerate(paths):
        dest = f"expert-{i + 1}.pt"
        provenance.append(export_weights(p, assets / dest))
        exported[p] = str(public / "weights" / dest)
    if model.endswith(".json"):
        converted = {
            "routes": {k: exported[v] for k, v in router["routes"].items()},
            "default": exported[router["default"]],
        }
        (out / "policy.json").write_text(json.dumps(converted, indent=2) + "\n")
        learner = str(public / "policy.json")
    else:
        learner = exported[model]
    if single["model"] not in exported:
        provenance.append(export_weights(single["model"], assets / "single.pt"))
        exported[single["model"]] = str(public / "weights/single.pt")
    (out / "single.json").write_text(
        json.dumps({single["label"]: exported[single["model"]]}, indent=2) + "\n"
    )
    league = {}
    for t, entries in spec["pool"].items():
        dest = f"rival-{Path(t).stem}.pt"
        provenance.append(export_weights(entries[-1], assets / dest))
        league[t] = [*entries[:-1], str(public / "weights" / dest)]
    (out / "pool.json").write_text(json.dumps(league, indent=2) + "\n")
    (out / "candidate.json").write_text(json.dumps({spec["label"]: learner}, indent=2) + "\n")
    write_csv(out / "weights.csv", provenance)
    plot(out, rows, games(root / "calibration"))
    print(json.dumps(summaries, indent=2))


def plot(out, rows, calibration):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
    teams = [Path(t).stem for t in ROSTERS]
    for ax, data, win, title in [
        (axes[0], rows, "win", "Trained policy vs. league"),
        (
            axes[1],
            [r for r in calibration if r["candidate"] == "damage"],
            "loss",
            "League vs. damage bot, mirror teams",
        ),
    ]:
        mat = np.array(
            [
                [
                    100
                    * np.mean(
                        [
                            r["outcome"] == win
                            for r in data
                            if r["roster"] == team and r["family"] == family
                        ]
                    )
                    for family in FAMILIES
                ]
                for team in teams
            ]
        )
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(
                j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 55 else "#16283b"
            )
        ax.set(
            xticks=range(3),
            xticklabels=["Damage", "Tactical", "PPO"],
            yticks=range(4),
            yticklabels=teams,
            title=title,
            xlabel="League opponent family",
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(out / "league.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--final", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--project-path", type=Path, default=Path("results/sketch5-fixed-league"))
    a = p.parse_args()
    report(
        a.runs.resolve(),
        a.validation.resolve(),
        a.final.resolve(),
        a.output.resolve(),
        a.project_path,
    )
