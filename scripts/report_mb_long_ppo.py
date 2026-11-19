"""Summarize performance, size, active count and expert identity over matched PPO budgets."""

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import CELLS, EARLY, FINAL, jobspecs, observations, path_for, schedule
from mb_roster_ppo import LONG_ARMS, LONG_OUT, LONG_RUN, SEEDS, TEAMS, tag, verify_assets
from vgc_posttrain import atomic_rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_training():
    models, logs = [], []
    with gzip.open(LONG_OUT / "training-games.csv.gz", "wt", newline="") as handle:
        writer = None
        for arm in LONG_ARMS:
            for seed in SEEDS:
                dest = LONG_RUN / tag(arm, seed)
                ck = torch.load(dest / "policy.pt", weights_only=True, map_location="cpu")
                assert (ck["arm"], ck["seed"], ck["steps"]) == (arm, seed, FINAL)
                assert ck["team_decisions"] == {Path(t).stem: FINAL // 4 for t in TEAMS}
                with (dest / "training.csv").open() as f:
                    updates = list(csv.DictReader(f))
                assert [int(x["steps"]) for x in updates] == list(range(256, FINAL + 1, 256))
                for row in updates:
                    expert = [int(v) for k, v in row.items() if k.startswith("expert_")]
                    if arm != "shared":
                        assert sum(expert) == 256 * (1 if arm == "moe1" else 2)
                    logs.append(
                        dict(
                            arm=arm,
                            training_seed=seed,
                            steps=row["steps"],
                            games=row["games"],
                            wins_in_block=row["wins"],
                            loss=row["loss"],
                            expert_counts=",".join(map(str, expert)),
                        )
                    )
                with (dest / "episodes.csv").open() as f:
                    completed = list(csv.DictReader(f))
                assert len(completed) == int(updates[-1]["games"])
                assert sum(int(x["wins"]) for x in updates) == sum(
                    x["reward"] == "1" for x in completed
                )
                for row in completed:
                    entry = dict(arm=arm, training_seed=seed, **row)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(entry), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(entry)
                ident = ""
                if arm in ("shared", "moe"):
                    ref = json.loads((dest / "reference.json").read_text())
                    assert ref["identical_tensors"] == len(ck["state"])
                    ident = ref["reference_sha256"]
                recovery_path = dest / "progress.pt"
                recovery = torch.load(recovery_path, weights_only=True, map_location="cpu")
                assert recovery["logs"][-1]["steps"] == FINAL and len(recovery["streams"]) == 4
                assert recovery["optimizer"]["state"]
                assert all(torch.equal(v, recovery["model"][k]) for k, v in ck["state"].items())
                recovery_hash = digest(recovery_path)
                del recovery
                for budget in (EARLY, FINAL):
                    models.append(
                        dict(
                            arm=arm,
                            training_seed=seed,
                            steps=budget,
                            parameters=ck["parameters"],
                            checkpoint_sha256=digest(path_for(arm, seed, budget)),
                            resume_sha256=recovery_hash if budget == FINAL else "",
                            original_65k_sha256=ident,
                        )
                    )
    atomic_rows(LONG_OUT / "training.csv", logs)
    atomic_rows(LONG_OUT / "checkpoints.csv", models)


def evaluate():
    arrays = defaultdict(list)
    seen_previews = {}
    per_seed, per_pair, usage = [], [], []
    n = 0
    with gzip.open(LONG_OUT / "games.csv.gz", "wt", newline="") as handle:
        writer = None
        for arm, seed, steps, variant in jobspecs():
            original = observations(arm, seed, steps, variant)
            expected = {
                (cell, sim): (Path(own).stem, Path(foe).stem, side)
                for _, _, own, foe, battles, _, cell in schedule(arm, seed, steps, variant)
                for sim, side in battles
            }
            assert len(original) == len(expected) and len(
                {(r["cell"], r["seed"]) for r in original}
            ) == len(original)
            meta = dict(arm=arm, training_seed=seed, steps=steps, variant=variant)
            groups = defaultdict(list)
            for r in original:
                cell, sim = r["cell"], int(r["seed"])
                assert expected[cell, sim] == (r["own"], r["foe"], r["side"])
                assert (
                    r["mode"] == "sample"
                    and r["selector"] == "native"
                    and r["reward"] in ("-1", "0", "1")
                )
                preview = r["lineup"], r["opponent_lineup"]
                assert preview == seen_previews.setdefault((cell, sim), preview)
                groups[cell].append(r)
                routes = r.get("routes", "")
                if not routes and arm == "moe" and variant == "normal" and steps == EARLY:
                    routes = ",".join(r[f"expert_{i}"] for i in range(4))
                if variant == "normal" and arm != "shared":
                    counts = np.array([int(x) for x in routes.split(",")])
                    assert counts.size == (8 if arm == "moe8" else 4) and counts.sum() > 0
                    assert counts.sum() % (1 if arm == "moe1" else 2) == 0
                entry = dict(
                    **meta,
                    **{
                        k: r[k]
                        for k in (
                            "cell",
                            "seed",
                            "own",
                            "foe",
                            "side",
                            "lineup",
                            "opponent_lineup",
                            "reward",
                            "turns",
                        )
                    },
                    routes=routes,
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(entry), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(entry)
                n += 1
            for cell, rs in groups.items():
                rs.sort(key=lambda row: int(row["seed"]))
                wins = np.array([r["reward"] == "1" for r in rs], dtype=float).reshape(4, 8, 16)
                arrays[arm, steps, variant, cell].append(wins)
                per_seed.append(dict(**meta, cell=cell, wins=int(wins.sum()), games=wins.size))
                for pos in range(0, len(rs), 16):
                    group = rs[pos : pos + 16]
                    assert len({(x["own"], x["foe"]) for x in group}) == 1
                    per_pair.append(
                        dict(
                            **meta,
                            cell=cell,
                            own=group[0]["own"],
                            foe=group[0]["foe"],
                            wins=sum(x["reward"] == "1" for x in group),
                            games=16,
                        )
                    )
                if variant == "normal" and arm != "shared":
                    for own in sorted({r["own"] for r in rs}):
                        rows = [r for r in rs if r["own"] == own]
                        cols = [
                            r["routes"].split(",")
                            if "routes" in r
                            else [r[f"expert_{i}"] for i in range(4)]
                            for r in rows
                        ]
                        totals = np.array(cols, dtype=int).sum(axis=0)
                        usage.append(
                            dict(
                                **meta,
                                cell=cell,
                                team=own,
                                decisions=int(totals.sum()) // (1 if arm == "moe1" else 2),
                                counts=",".join(map(str, totals)),
                            )
                        )
    assert n == 114688, n
    stacks = {k: np.stack(v) for k, v in arrays.items()}
    for arm in LONG_ARMS:
        for steps in (EARLY, FINAL):
            for variant in ("normal",) if arm == "shared" else ("normal", "rotate", "zero"):
                key = (arm, steps, variant)
                stacks[*key, "heldout"] = (stacks[*key, "forward"] + stacks[*key, "reverse"]) / 2
    atomic_rows(LONG_OUT / "per-seed.csv", per_seed)
    atomic_rows(LONG_OUT / "per-pair.csv", per_pair)
    atomic_rows(LONG_OUT / "expert-usage.csv", usage)
    return stacks


def bootstrap(table):
    # Identical index draws to the previous crossed bootstrap. Multiplicity weights allow
    # all contrasts to reuse one resample without copying every battle outcome each time.
    rng = np.random.default_rng(141592)
    draws = 6000
    s = rng.integers(4, size=(draws, 4))
    a = rng.integers(4, size=(draws, 4))
    b = rng.integers(8, size=(draws, 8))
    k = np.arange(16) % 2 + 2 * rng.integers(8, size=(draws, 4, 8, 16))
    mat = np.stack([row.ravel() for row in table.values()], axis=1)
    samples = np.empty((draws, len(table)))
    for i in range(draws):
        pos = (a[i, :, None, None] * 8 + b[i, None, :, None]) * 16 + k[i]
        weight = np.bincount(s[i], minlength=4)[:, None] * np.bincount(pos.ravel(), minlength=512)
        samples[i] = weight.ravel() @ mat * (100 / 2048)
    intervals = np.percentile(samples, [2.5, 97.5], axis=0).T
    return dict(zip(table, intervals, strict=True))


def summarize(stacks):
    deltas = {}
    for cell in (*CELLS, "heldout"):
        for arm in LONG_ARMS:

            def normal(t):
                return stacks[arm, t, "normal", cell]

            deltas[arm, "training gain", cell] = normal(FINAL) - normal(EARLY)
            if arm != "shared":
                for t in (EARLY, FINAL):
                    deltas[arm, f"vs shared at {t}", cell] = (
                        normal(t) - stacks["shared", t, "normal", cell]
                    )
                    if arm != "moe":
                        deltas[arm, f"vs 4/top2 at {t}", cell] = (
                            normal(t) - stacks["moe", t, "normal", cell]
                        )
                    if cell in ("forward", "reverse", "heldout"):
                        for v, desc in (("rotate", "identity"), ("zero", "expert contribution")):
                            deltas[arm, f"{desc} at {t}", cell] = (
                                normal(t) - stacks[arm, t, v, cell]
                            )
                if cell in ("forward", "reverse", "heldout"):
                    deltas[arm, "identity effect change", cell] = (
                        deltas[arm, f"identity at {FINAL}", cell]
                        - deltas[arm, f"identity at {EARLY}", cell]
                    )
                if arm != "moe" and cell in ("forward", "reverse", "heldout"):
                    for t in (EARLY, FINAL):
                        deltas[arm, f"identity vs 4/top2 at {t}", cell] = deltas[
                            arm, f"identity at {t}", cell
                        ] - (stacks["moe", t, "normal", cell] - stacks["moe", t, "rotate", cell])
    intervals = bootstrap({**stacks, **deltas})
    scores = []
    for k, arr in stacks.items():
        arm, steps, variant, cell = k
        games = arr.size * (2 if cell == "heldout" else 1)
        lo, hi = intervals[k]
        scores.append(
            dict(
                arm=arm,
                steps=steps,
                variant=variant,
                cell=cell,
                wins=round(arr.mean() * games),
                games=games,
                win_pct=100 * arr.mean(),
                lo95=lo,
                hi95=hi,
            )
        )
    atomic_rows(LONG_OUT / "scores.csv", scores)
    effects = []
    for k, arr in deltas.items():
        arm, effect, cell = k
        lo, hi = intervals[k]
        effects.append(
            dict(
                arm=arm,
                effect=effect,
                cell=cell,
                delta_pp=100 * arr.mean(),
                lo95=lo,
                hi95=hi,
                positive_seeds=int((arr.mean(axis=(1, 2, 3)) > 0).sum()),
            )
        )
    atomic_rows(LONG_OUT / "effects.csv", effects)


if __name__ == "__main__":
    verify_assets()
    audit_training()
    summarize(evaluate())
    assert len((LONG_OUT / "replayed.csv").read_text().splitlines()) == 161
    print(
        "Audited 32 checkpoint/budget combinations, 114688 scores and 160 replay checks.",
        flush=True,
    )
