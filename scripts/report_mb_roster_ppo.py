"""Audit decisions and preview pairing, then score specialist and shared PPO policies."""

import csv
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from eval_mb_roster_ppo import CELLS, assignments, candidates, read_scores
from mb_roster_ppo import ARMS, OUT, ROSTERS, RUN, SEEDS, SPEC, TEAMS, tag, verify_assets
from report_mb_posttraining import interval
from report_mb_rewards import archive
from vgc_posttrain import atomic_rows


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def audit():
    checkpoints, curves, training_games = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            spent = Counter()
            for i in range(4) if arm == "specialist" else [None]:
                root = RUN / tag(arm, seed, i)
                policy = root / "policy.pt"
                ck = torch.load(policy, map_location="cpu", weights_only=True)
                assert (ck["arm"], ck["seed"], ck["index"]) == (arm, seed, i)
                spent.update(ck["team_decisions"])
                assert sum(ck["team_decisions"].values()) == ck["steps"]
                prefix = dict(arm=arm, training_seed=seed, checkpoint=root.name)
                checkpoints.append(
                    dict(
                        **prefix,
                        steps=ck["steps"],
                        parameters=ck["parameters"],
                        sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                    )
                )
                logs = read(root / "training.csv")
                assert [int(r["steps"]) for r in logs] == list(range(256, ck["steps"] + 1, 256))
                games = read(root / "episodes.csv")
                assert int(logs[-1]["games"]) == len(games)
                assert sum(int(r["wins"]) for r in logs) == sum(r["reward"] == "1" for r in games)
                assert set(ck["team_decisions"]) == {r["own"] for r in games}
                for own in ck["team_decisions"]:
                    found = {r["lineup"] for r in games if r["own"] == own}
                    assert len(found) > 1 and any("5" in x or "6" in x for x in found)
                assert {r["foe"] for r in games} == {Path(p).stem for p in ROSTERS["foe_train"]}
                for r in logs:
                    if arm == "moe":
                        assert sum(int(r[f"expert_{j}"]) for j in range(4)) == 512
                    curves.append({**prefix, **{f"expert_{j}": "" for j in range(4)}, **r})
                training_games.extend(dict(**prefix, **r) for r in games)
            assert spent == {Path(t).stem: SPEC["steps_per_practiced_roster"] for t in TEAMS}
    assert len(checkpoints) == 24
    atomic_rows(OUT / "checkpoints.csv", checkpoints)
    atomic_rows(OUT / "training.csv", curves)
    archive(OUT / "training-games.csv.gz", training_games)


def score_all():
    arrays = defaultdict(list)
    paired = {}
    seeds, pairs, games, routes = [], [], [], []
    for arm, seed in candidates():
        rows = read_scores(arm, seed)
        expected = {
            (t[-1], simseed): (t[0], Path(t[2]).stem, Path(t[3]).stem, side)
            for t in assignments(arm, seed)
            for simseed, side in t[4]
        }
        assert len(rows) == len(expected) == 2048
        observed = set()
        for r in rows:
            k = r["cell"], int(r["seed"])
            assert k not in observed
            observed.add(k)
            assert expected[k] == (r["policy"], r["own"], r["foe"], r["side"])
            assert r["selector"] == "native" and r["mode"] == "sample"
            assert r["reward"] in ("-1", "0", "1")
            lineups = (r["lineup"], r["opponent_lineup"])
            assert lineups == paired.setdefault(k, lineups)
            assert all(len(x) == len(set(x)) == 4 and set(x) <= set("123456") for x in lineups)
        games.extend(dict(arm=arm, training_seed=seed, **r) for r in rows)
        for cell in CELLS:
            group = sorted((r for r in rows if r["cell"] == cell), key=lambda r: int(r["seed"]))
            k = SPEC["evaluation"]["games_per_pair"]
            wins = np.array([r["reward"] == "1" for r in group], dtype=float).reshape(4, 8, k)
            arrays[arm, cell].append(wins)
            meta = dict(arm=arm, training_seed=seed, cell=cell)
            seeds.append(dict(**meta, wins=int(wins.sum()), games=wins.size))
            for start in range(0, len(group), k):
                block = group[start : start + k]
                own, foe = block[0]["own"], block[0]["foe"]
                assert all(r["own"] == own and r["foe"] == foe for r in block)
                pairs.append(
                    dict(
                        **meta,
                        own=own,
                        foe=foe,
                        wins=sum(r["reward"] == "1" for r in block),
                        games=k,
                    )
                )
            if arm == "moe":
                for own in sorted({r["own"] for r in group}):
                    counts = np.array(
                        [
                            [int(r[f"expert_{i}"]) for i in range(4)]
                            for r in group
                            if r["own"] == own
                        ]
                    )
                    assert (counts.sum(axis=1) > 0).all() and (counts.sum(axis=1) % 2 == 0).all()
                    totals = counts.sum(axis=0)
                    routes.append(
                        dict(
                            **meta,
                            own=own,
                            decisions=int(totals.sum() // 2),
                            **{f"expert_{i}": int(n) for i, n in enumerate(totals)},
                        )
                    )
    arrays = {k: np.stack(rows) for k, rows in arrays.items()}
    for arm in ("bc", *ARMS):
        arrays[arm, "balanced"] = (arrays[arm, "forward"] + arrays[arm, "reverse"]) / 2
    scores, effects = [], []
    for cell in (*CELLS, "balanced"):
        for arm in ("bc", *ARMS):
            v = arrays[arm, cell]
            lo, hi = interval(v)
            mult = 2 if cell == "balanced" else 1
            scores.append(
                dict(
                    arm=arm,
                    cell=cell,
                    wins=int(v.sum() * mult),
                    games=v.size * mult,
                    win_pct=100 * v.mean(),
                    lo95=lo,
                    hi95=hi,
                )
            )
        for ctrl, alt in combinations(("bc", *ARMS), 2):
            delta = arrays[alt, cell] - arrays[ctrl, cell]
            lo, hi = interval(delta)
            effects.append(
                dict(
                    arm=alt,
                    baseline=ctrl,
                    cell=cell,
                    delta_pp=100 * delta.mean(),
                    lo95=lo,
                    hi95=hi,
                    seeds_positive=int((delta.mean(axis=(1, 2, 3)) > 0).sum()),
                )
            )
    for arm, seed in candidates():
        rs = [
            r
            for r in seeds
            if r["arm"] == arm
            and r["training_seed"] == seed
            and r["cell"] in ("forward", "reverse")
        ]
        seeds.append(
            dict(
                arm=arm,
                training_seed=seed,
                cell="balanced",
                wins=sum(r["wins"] for r in rs),
                games=sum(r["games"] for r in rs),
            )
        )
    for file, items in [
        ("scores", scores),
        ("effects", effects),
        ("per-seed", seeds),
        ("per-pair", pairs),
        ("expert-usage", routes),
    ]:
        atomic_rows(OUT / (file + ".csv"), items)
    archive(OUT / "games.csv.gz", games)
    assert len(games) == 26624
    print("Audited 24 checkpoints and", len(games), "paired-schedule battle records.", flush=True)


if __name__ == "__main__":
    verify_assets()
    audit()
    score_all()
