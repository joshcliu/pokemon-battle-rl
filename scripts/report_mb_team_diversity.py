"""Paired held-out effects of learner team diversity and frozen expert-routing interventions."""

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import FINAL, jobspecs, observations, schedule
from mb_roster_ppo import (
    DIVERSE_OUT as OUT,
)
from mb_roster_ppo import (
    DIVERSE_RUN as RUN,
)
from mb_roster_ppo import (
    LONG_OUT,
    SEEDS,
    diversity_protocol,
    roster_batch,
    verify_assets,
)
from report_mb_long_ppo import bootstrap, digest
from vgc_posttrain import atomic_rows


def audit_training(spec):
    checkpoints, exposure, updates = [], [], []
    teams = [Path(t).stem for t in spec["own_train"]]
    foes = {Path(t).stem for t in spec["foe_train"]}
    for seed in SEEDS:
        for arm in spec["arms"]:
            dest = RUN / f"{arm}-{seed}"
            model = torch.load(dest / "policy.pt", map_location="cpu", weights_only=True)
            assert (model["arm"], model["seed"], model["steps"]) == (arm, seed, FINAL)
            ck = torch.load(dest / "progress.pt", map_location="cpu", weights_only=True)
            assert len(ck["streams"]) == 16 and ck["optimizer"]["state"]
            assert all(torch.equal(t, ck["model"][k]) for k, t in model["state"].items())
            del ck
            counts = Counter()
            with (dest / "training.csv").open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == FINAL // 256
            for i, row in enumerate(rows):
                idx = list(map(int, row["roster_indices"].split(",")))
                assert idx == roster_batch(seed, i, 16).tolist()
                assert int(row["steps"]) == (i + 1) * 256 and np.isfinite(float(row["loss"]))
                if arm == "moe":
                    assert sum(int(row[f"expert_{e}"]) for e in range(4)) == 512
                counts.update(teams[j] for j in idx)
                updates.append(
                    dict(
                        arm=arm,
                        training_seed=seed,
                        steps=row["steps"],
                        games=row["games"],
                        wins=row["wins"],
                        loss=row["loss"],
                        roster_indices=row["roster_indices"],
                        expert_counts=",".join(row.get(f"expert_{e}", "") for e in range(4))
                        if arm == "moe"
                        else "",
                    )
                )
            assert (
                model["team_decisions"]
                == {t: counts[t] * 64 for t in teams}
                == dict.fromkeys(teams, 16384)
            )
            with (dest / "episodes.csv").open() as f:
                episodes = list(csv.DictReader(f))
            assert len(episodes) == int(rows[-1]["games"])
            assert sum(r["reward"] == "1" for r in episodes) == sum(int(r["wins"]) for r in rows)
            assert len({r["seed"] for r in episodes}) == len(episodes)
            cells = Counter((r["own"], r["foe"]) for r in episodes)
            assert set(cells) == {(team, foe) for team in teams for foe in foes}
            for (team, foe), n in sorted(cells.items()):
                exposure.append(
                    dict(arm=arm, training_seed=seed, own=team, foe=foe, completed_games=n)
                )
            checkpoints.append(
                dict(
                    arm=arm,
                    training_seed=seed,
                    steps=FINAL,
                    decisions_per_roster=16384,
                    parameters=model["parameters"],
                    games=len(episodes),
                    checkpoint_sha256=digest(dest / "policy.pt"),
                    resume_sha256=digest(dest / "progress.pt"),
                )
            )
    atomic_rows(OUT / "checkpoints.csv", checkpoints)
    atomic_rows(OUT / "training.csv", updates)
    atomic_rows(OUT / "training-matchups.csv", exposure)


def evaluate():
    policy_jobs = list(jobspecs(diverse=True))
    baseline = defaultdict(list)
    with gzip.open(LONG_OUT / "games.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if (
                r["steps"] == str(FINAL)
                and r["arm"] in ("shared", "moe")
                and r["cell"] in ("forward", "reverse")
            ):
                baseline[r["arm"], int(r["training_seed"]), r["variant"]].append(r)
    boards = {}
    table, usage, per_pair, per_seed = {}, [], [], []
    with gzip.open(OUT / "games.csv.gz", "wt", newline="") as f:
        writer = None
        for arm, seed, steps, variant in policy_jobs:
            expected = {
                (cell, sim): (Path(own).stem, Path(foe).stem, side)
                for _, _, own, foe, sims, _, cell in schedule(
                    arm, seed, steps, variant, diverse=True
                )
                for sim, side in sims
            }
            for count, rows in (
                (4, baseline[arm, seed, variant]),
                (16, observations(arm, seed, steps, variant, diverse=True)),
            ):
                assert len(rows) == len(expected) == 1024
                assert {(r["cell"], int(r["seed"])) for r in rows} == set(expected)
                for r in rows:
                    key = r["cell"], int(r["seed"])
                    assert (r["own"], r["foe"], r["side"]) == expected[key]
                    pair = r["lineup"], r["opponent_lineup"]
                    assert boards.setdefault(key, pair) == pair
                    assert r["reward"] in ("-1", "0", "1")
                    if count == 16:
                        assert r["mode"] == "sample" and r["selector"] == "native"
                        assert r["policy"] == f"diverse/{arm}-{seed}-{FINAL}-{variant}"
                    entry = dict(
                        training_teams=count,
                        arm=arm,
                        training_seed=seed,
                        variant=variant,
                        **{
                            k: r[k]
                            for k in (
                                "cell",
                                "seed",
                                "side",
                                "own",
                                "foe",
                                "lineup",
                                "opponent_lineup",
                                "reward",
                                "turns",
                                "routes",
                            )
                        },
                    )
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(entry), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(entry)
                for cell in ("forward", "reverse"):
                    games = sorted(
                        (r for r in rows if r["cell"] == cell), key=lambda r: int(r["seed"])
                    )
                    wins = np.array([r["reward"] == "1" for r in games], dtype=float).reshape(
                        4, 8, 16
                    )
                    table.setdefault((count, arm, variant, cell), np.zeros((4, 4, 8, 16)))[
                        SEEDS.index(seed)
                    ] = wins
                    per_seed.append(
                        dict(
                            training_teams=count,
                            arm=arm,
                            variant=variant,
                            training_seed=seed,
                            cell=cell,
                            wins=int(wins.sum()),
                            games=512,
                        )
                    )
                    for pos in range(0, 512, 16):
                        part = games[pos : pos + 16]
                        assert len({(r["own"], r["foe"]) for r in part}) == 1
                        per_pair.append(
                            dict(
                                training_teams=count,
                                arm=arm,
                                variant=variant,
                                training_seed=seed,
                                cell=cell,
                                own=part[0]["own"],
                                foe=part[0]["foe"],
                                wins=sum(r["reward"] == "1" for r in part),
                                games=16,
                            )
                        )
                    if arm == "moe" and variant == "normal":
                        for team in sorted({r["own"] for r in games}):
                            counts = np.array(
                                [
                                    list(map(int, r["routes"].split(",")))
                                    for r in games
                                    if r["own"] == team
                                ]
                            )
                            assert (
                                counts.shape[1] == 4
                                and (counts.sum(axis=1) % 2 == 0).all()
                                and (counts.sum(axis=1) > 0).all()
                            )
                            total = counts.sum(axis=0)
                            usage.append(
                                dict(
                                    training_teams=count,
                                    training_seed=seed,
                                    cell=cell,
                                    own=team,
                                    decisions=int(total.sum()) // 2,
                                    **{f"expert_{i}": int(n) for i, n in enumerate(total)},
                                )
                            )
    for k in list(table):
        count, arm, variant, cell = k
        if cell == "forward":
            table[count, arm, variant, "heldout"] = (
                table[k] + table[count, arm, variant, "reverse"]
            ) / 2
    atomic_rows(OUT / "per-seed.csv", per_seed)
    atomic_rows(OUT / "per-pair.csv", per_pair)
    atomic_rows(OUT / "expert-usage.csv", usage)
    return table


def summarize(table):
    effects = {}
    for c in ("forward", "reverse", "heldout"):
        for n in (4, 16):
            normal = table[n, "moe", "normal", c]
            effects[f"MoE minus shared ({n} teams)", c] = normal - table[n, "shared", "normal", c]
            effects[f"Normal minus rotated ({n} teams)", c] = normal - table[n, "moe", "rotate", c]
            effects[f"Normal minus backbone ({n} teams)", c] = normal - table[n, "moe", "zero", c]
        for arm in ("moe", "shared"):
            effects[f"16 minus 4 teams ({arm})", c] = (
                table[16, arm, "normal", c] - table[4, arm, "normal", c]
            )
        for effect in ("Normal minus rotated", "Normal minus backbone", "MoE minus shared"):
            effects[f"Change in {effect.lower()} with diversity", c] = (
                effects[f"{effect} (16 teams)", c] - effects[f"{effect} (4 teams)", c]
            )
    intervals = bootstrap({**table, **effects})
    scores = []
    for k, value in table.items():
        n, arm, variant, cell = k
        games = value.size * (2 if cell == "heldout" else 1)
        lo, hi = intervals[k]
        scores.append(
            dict(
                training_teams=n,
                arm=arm,
                variant=variant,
                cell=cell,
                wins=round(value.mean() * games),
                games=games,
                win_pct=100 * value.mean(),
                lo95=lo,
                hi95=hi,
            )
        )
    effects_rows = []
    for k, value in effects.items():
        lo, hi = intervals[k]
        effects_rows.append(
            dict(
                effect=k[0],
                cell=k[1],
                delta_pp=100 * value.mean(),
                lo95=lo,
                hi95=hi,
                **{
                    f"seed_{s}_pp": 100 * v
                    for s, v in zip(SEEDS, value.mean(axis=(1, 2, 3)), strict=True)
                },
            )
        )
    atomic_rows(OUT / "scores.csv", scores)
    atomic_rows(OUT / "effects.csv", effects_rows)
    plot_effects(effects_rows)


def plot_effects(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "16 minus 4 teams (shared)": "16 vs 4 teams · shared PPO",
        "16 minus 4 teams (moe)": "16 vs 4 teams · MoE",
        "MoE minus shared (16 teams)": "MoE vs shared PPO · 16 teams",
        "Normal minus rotated (4 teams)": "Learned vs rotated routing · 4 teams",
        "Normal minus rotated (16 teams)": "Learned vs rotated routing · 16 teams",
        "Change in normal minus rotated with diversity": "Change in routing benefit · 16 vs 4",
    }
    lookup = {r["effect"]: r for r in rows if r["cell"] == "heldout"}
    fig, ax = plt.subplots(figsize=(8.1, 3.5), layout="constrained")
    for j, effect in enumerate(labels):
        row = lookup[effect]
        color = "#087b73" if j >= 3 else "#404a57"
        ax.plot([row["lo95"], row["hi95"]], [j, j], color=color, lw=2)
        ax.scatter(row["delta_pp"], j, s=35, color=color, zorder=3)
    ax.axvline(0, color="#777", ls="--", lw=1, zorder=0)
    ax.set_yticks(range(len(labels)), list(labels.values()))
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.set_xlabel("Paired change in held-out win rate (percentage points)")
    ax.set_title("Team diversity at 262k decisions · 95% intervals", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "heldout-effects.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    verify_assets()
    spec = diversity_protocol()
    audit_training(spec)
    summarize(evaluate())
    assert len((OUT / "replayed.csv").read_text().splitlines()) == 33
    print(
        "Verified eight final checkpoints and equal exposure on 16 teams. Scored 16,384 new held-out games, paired with 16,384 archived games; verified 32 replays.",
        flush=True,
    )
