"""Offline contextual-bandit policy improvement from the archived matched training quartets."""

import argparse
import csv
import functools
import gzip
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from mb_policy_iteration_eval import GAMES, OUT, RUN, SEEDS, SPEC, decide, schedule, verify
from mb_residual_preview import OUT as ARCHIVE
from mb_residual_preview import ROSTERS, pilot
from vgc_posttrain import Match, atomic_rows, batch_obs

from pokemon_rl.league import stat

LINES = [
    (a, b, c, d)
    for a, b in combinations(range(6), 2)
    for c, d in combinations([i for i in range(6) if i not in (a, b)], 2)
]
LINE_ID = {line: k for k, line in enumerate(LINES)}
SLOTS = np.array(LINES)
ROLES = [
    {"fakeout"},
    {"followme", "ragepowder"},
    {"tailwind"},
    {"trickroom"},
    {"protect", "detect"},
    {"spore", "sleeppowder", "yawn"},
    {"icywind", "electroweb"},
    {"instruct", "helpinghand"},
    {"partingshot", "uturn", "voltswitch", "flipturn"},
    {"swordsdance", "dragondance", "calmmind", "nastyplot"},
    {"wideguard", "quickguard"},
    {"taunt", "encore"},
    {"lightscreen", "reflect", "auroraveil"},
]


def matchup(view):
    own, foes = list(view.team.values()), list(view.opponent_team.values())
    assert len(own) == len(foes) == 6

    def pressure(a, d):
        hits = []
        for move in a.moves.values():
            if not move.base_power or not move.type:
                continue
            physical = move.category.name == "PHYSICAL"
            factor = d.damage_multiplier(move.type)
            if (move.type.name == "GROUND" and d.ability == "levitate") or (
                move.type.name == "FIRE" and d.ability in ("flashfire", "wellbakedbody")
            ):
                factor = 0
            power = move.base_power * move.expected_hits * float(move.accuracy)
            power *= 1.5 if move.type in a.types else 1
            hits.append(
                np.clip(
                    (
                        0.44
                        * power
                        * stat(a, "atk" if physical else "spa")
                        / stat(d, "def" if physical else "spd")
                        + 2
                    )
                    * factor
                    / stat(d, "hp"),
                    0,
                    2,
                )
            )
        return max(hits, default=0.0)

    attack = np.array([[pressure(a, f) for f in foes] for a in own])
    danger = np.array([[pressure(f, a) for f in foes] for a in own])
    speed = np.array([[stat(a, "spe") > stat(f, "spe") for f in foes] for a in own], dtype=float)

    def traits(m):
        moves = set(m.moves)
        return [bool(moves & role) for role in ROLES] + [
            m.ability in ("drought", "drizzle", "snowwarning", "sandstream"),
            m.ability in ("chlorophyll", "swiftswim", "slushrush", "sandrush"),
            m.ability == "intimidate",
            m.ability == "prankster",
            bool(m.item and m.item.endswith("ite")),
            stat(m, "spe") / 150,
            (stat(m, "def") + stat(m, "spd")) / 250,
            max(stat(m, "atk"), stat(m, "spa")) / 160,
        ]

    own_traits = np.array([traits(m) for m in own])
    theirs = np.array([traits(m) for m in foes]).mean(0)
    features = []
    for slots in (SLOTS[:, :2], SLOTS[:, 2:], SLOTS):
        a, d, s = attack[slots], danger[slots], speed[slots]
        # Opponent slots are exchangeable. Order statistics retain the worst matchups.
        for matrix in (
            a.max(1),
            a.mean(1),
            d.min(1),
            d.mean(1),
            s.mean(1),
            (a / (0.4 + d)).max(1),
            (a * s).max(1),
        ):
            features.append(np.sort(matrix, axis=1))
        role = own_traits[slots]
        group = role.mean(1)
        features.extend((group, role.max(1), group * theirs))
        features.extend(
            (
                group[:, 3:4] * (1 - s.mean((1, 2)))[:, None],
                group[:, 2:3] * s.mean((1, 2))[:, None],
                role[:, :, 13].max(1)[:, None] * role[:, :, 14].max(1)[:, None],
            )
        )
    return np.concatenate(features, axis=1).astype(np.float32)


def preview_features(own, foe):
    match = Match()
    features = []

    def pick(view):
        features.append(matchup(view))
        return "1234"

    match.reset(own, foe, 557001001, "p1", preview_picker=pick)
    assert len(features) == 1
    return features[0]


def closure(x, y, lam):
    scales = np.sqrt(np.mean(x * x, axis=0)).clip(0.05)
    z = x / scales
    reg = z.T @ z + lam * np.eye(z.shape[1])
    return np.linalg.solve(reg, z.T @ y) / scales


def fit():
    own, foe = ROSTERS["own_train"], ROSTERS["foe_train"]
    assert len(own) == 16 and len(foe) == 8
    assert set(own) <= set(SPEC["own_train"]) and set(foe) <= set(SPEC["foe_train"])
    xn = {(Path(x).stem, Path(y).stem): preview_features(x, y) for x in own for y in foe}
    rows_by_seed = {s: [] for s in SEEDS}
    with gzip.open(ARCHIVE / "training-games.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            rows_by_seed[int(row["training_seed"])].append(row)
    dest = RUN / "lineup-value"
    dest.mkdir(parents=True, exist_ok=True)
    for seed, rows in rows_by_seed.items():
        rows.sort(key=lambda r: (int(r["group"]), int(r["sample"])))
        assert len(rows) == 16384
        features = []
        labels = []
        pairs = []
        for g in range(len(rows) // 4):
            group = rows[4 * g : 4 * g + 4]
            assert (
                len(
                    {
                        (r["own"], r["foe"], r["seed"], r["side"], r["opponent_lineup"])
                        for r in group
                    }
                )
                == 1
            )
            assert [int(r["sample"]) for r in group] == list(range(4))
            pair = group[0]["own"], group[0]["foe"]
            expanded = []
            for r in group:
                seq = [int(c) - 1 for c in r["lineup"]]
                line = tuple(sorted(seq[:2]) + sorted(seq[2:]))
                expanded.append(xn[pair][LINE_ID[line]])
            x = np.stack(expanded)
            y = np.array([float(r["reward"]) for r in group])
            features.append(x - x.mean(0))
            labels.append(y - y.mean())
            pairs.extend([pair] * 4)
        X = np.concatenate(features).astype(float)
        Y = np.concatenate(labels)
        ids = np.array(
            [
                ([Path(t).stem for t in own].index(x), [Path(t).stem for t in foe].index(y))
                for x, y in pairs
            ]
        )
        errors = {penalty: [] for penalty in (10.0, 100.0, 1000.0, 10000.0)}
        for fold in range(4):
            a, b = ids[:, 0] % 4 == fold, ids[:, 1] % 4 == fold
            train, test = ~a & ~b, a & b
            for lam, losses in errors.items():
                w = closure(X[train], Y[train], lam)
                losses.append(float(np.mean((X[test] @ w - Y[test]) ** 2)))
        chosen = min(errors, key=lambda penalty: np.mean(errors[penalty]))
        weights = closure(X, Y, chosen)
        np.savez(dest / f"seed-{seed}.npz", weights=weights, seed=seed, penalty=chosen)
        atomic_rows(
            OUT / "training" / f"lineup-value-{seed}.csv",
            [
                dict(
                    training_seed=seed,
                    penalty=penalty,
                    fold_mse=np.mean(losses),
                    selected=penalty == chosen,
                    training_groups=len(rows) // 4,
                )
                for penalty, losses in errors.items()
            ],
        )
        print("fit lineup value", seed, len(rows), chosen, flush=True)


@functools.lru_cache(maxsize=4)
def weights(seed):
    with np.load(RUN / "lineup-value" / f"seed-{seed}.npz") as state:
        assert int(state["seed"]) == seed
        return state["weights"]


@torch.no_grad()
def score(task):
    seed, cell = task
    i, j, d, own, foe, start = cell
    model, match = pilot(seed), Match()
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]

        def pick(view):
            scores = matchup(view) @ weights(seed)
            return "".join(str(x + 1) for x in LINES[int(scores.argmax())])

        match.reset(own, foe, sim, side, preview_picker=pick)
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            a = decide(model, batch_obs(match.obs()), "map", seed)
            done, r = match.step(a)
        rows.append(
            dict(
                method="lineup-q-map",
                training_seed=seed,
                i=i,
                j=j,
                direction=d,
                k=k,
                own=Path(own).stem,
                foe=Path(foe).stem,
                sim=sim,
                side=side,
                lineup=match.lineup,
                opp_lineup=match.opponent.lineup,
                reward=r,
                turns=match.battle.turn,
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "validation", "custom-dev"))
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    verify()
    if args.stage == "fit":
        fit()
    else:
        cells = list(schedule(args.stage))
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for s in SEEDS:
                dest = OUT / "scores" / f"{args.stage}-lineup-q-map-{s}.csv"
                if dest.exists():
                    continue
                rows = [r for group in pool.map(score, [(s, c) for c in cells]) for r in group]
                atomic_rows(dest, rows)
                print(args.stage, s, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
