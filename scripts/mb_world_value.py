"""Fit a terminal-return critic on fully specified training worlds for use inside public simulations."""

import argparse
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_mechanics_es import action, fitted
from mb_midgame_plan import choose, features, matchup_hit, projection, valid, value
from mb_opponent_belief import OUT, RUN
from mb_policy_iteration_eval import SEEDS, verify
from mb_residual_preview import pilot
from mb_rollout_branches import pairing
from torch import nn
from vgc_posttrain import Match, atomic_rows, batch_obs

BLOCKS = 40
GAMES = 28


def world_features(pos):
    old = features(pos)
    active = [[m for m in side if m] for side in (pos["own"], pos["foe"])]
    bench = pos["bench"]
    bucket = []
    for i in (0, 1):
        mons = active[i] + bench[i]
        other = 1 - i
        scalars = []
        for attackers, targets in (
            (active[i], active[other]),
            (active[i], bench[other]),
            (bench[i], active[other]),
            (bench[i], bench[other]),
        ):
            pressure = []
            for f in targets:
                hits = [
                    max(
                        (
                            matchup_hit(m["species"], mv, f["species"], f["ability"])
                            for mv in m["moves"]
                        ),
                        default=0.0,
                    )
                    for m in attackers
                ]
                pressure.append(max(hits, default=0))
            scalars.extend(
                [
                    sum(pressure),
                    max(pressure, default=0),
                    sum(h >= f["hp"] for h, f in zip(pressure, targets, strict=True)),
                ]
            )
        for subset in (active[i], bench[i]):
            h = sorted((m["hp"] for m in subset), reverse=True)
            scalars.extend((h + [0, 0])[:2])
            scalars.extend(
                [
                    sum(m["hp"] < 0.34 for m in subset),
                    sum(m["status"] in ("slp", "frz", "par") for m in subset),
                    sum("protect" in m["moves"] or "detect" in m["moves"] for m in subset),
                    sum(
                        bool(
                            set(m["moves"])
                            & {"followme", "ragepowder", "tailwind", "trickroom", "wideguard"}
                        )
                        for m in subset
                    ),
                ]
            )
        scalars.extend(
            [sum(m["hp"] ** 2 for m in mons), len(mons), sum(m["baseSpeed"] / 100 for m in mons)]
        )
        bucket.append(np.array(scalars))
    x, y = bucket
    return np.concatenate((old, x - y, (x + y) / 2))


def training_world(match):
    # This critic is fitted on hypothetical-world truth. Only the training data
    # collector can see both players. Deployment calls world_features on modeled leaves.
    a = projection(match.view)
    b = projection(match.battle.players[match.foe])
    if not valid(a) or not valid(b):
        return None
    a["foe"] = b["own"]
    a["bench"][1] = b["bench"][0]
    a["vacancies"] = 0
    a["unknown"] = []
    return a


@torch.no_grad()
def collect(task):
    seed, block = task
    dest = RUN / "critic" / f"s{seed}" / f"{block:03}.npz"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    match, model = Match(), pilot(seed)
    xs, ys, gids, own_ids, foe_ids = [], [], [], [], []
    for g in range(block * GAMES, (block + 1) * GAMES):
        own, foe, seat = pairing(seed, g)
        if g % 2:
            own, foe = foe, own
        sim = 120_000_000 + 1_000_000 * seed + g
        match.reset(own, foe, sim, seat, preview="native")
        torch.manual_seed(sim + 700_000_000)
        frames = []
        done = False
        while not done:
            pos = training_world(match)
            if pos is not None:
                frames.append(world_features(pos))
            ob = batch_obs(match.obs())
            if g % 4 == 0:
                a = choose(model, ob, match.view, seed, False)[0]
            elif g % 4 == 1:
                a = model(ob)[0][0].tolist()
            else:
                a = action(model, ob, match.view, fitted(seed))
            done, r = match.step(a)
        xs.extend(frames)
        ys.extend([(r + 1) / 2] * len(frames))
        gids.extend([g] * len(frames))
        own_ids.extend([own] * len(frames))
        foe_ids.extend([foe] * len(frames))
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        game=gids,
        own=own_ids,
        foe=foe_ids,
    )
    tmp.replace(dest)


def network(d):
    return nn.Sequential(
        nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)
    )


def fit_part(seed, x, y, g):
    torch.manual_seed(15042 + seed)
    center = x.mean(0)
    scale = x.std(0).clip(0.12, None)
    data = torch.from_numpy(((x - center) / scale).clip(-8, 8))
    old = torch.from_numpy((x[:, :23] @ value(seed)).astype(np.float32))
    label = torch.from_numpy(y)
    wt = torch.from_numpy((1 / np.bincount(g)[g]).astype(np.float32))
    wt /= wt.mean()
    model = network(x.shape[1])
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.004)
    for _ in range(16):
        for ix in torch.randperm(len(g)).split(256):
            adjustment = model(data[ix])[:, 0]
            pred = old[ix] + adjustment
            losses = nn.functional.binary_cross_entropy_with_logits(
                pred, label[ix], reduction="none"
            )
            loss = ((losses + 0.025 * adjustment.square()) * wt[ix]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), center, scale


def fit(seed):
    cols = {k: [] for k in ("x", "y", "game", "own", "foe")}
    for b in range(BLOCKS):
        with np.load(RUN / "critic" / f"s{seed}" / f"{b:03}.npz") as ds:
            for c in cols:
                cols[c].append(ds[c])
    x, y, g, own, foe = (np.concatenate(a) for a in cols.values())
    assert len(np.unique(g)) == BLOCKS * GAMES
    rosters = sorted(set(own) | set(foe))
    held = np.random.default_rng(52001).permutation(rosters)[:12]
    test = np.isin(own, held) & np.isin(foe, held)
    train = ~np.isin(own, held) & ~np.isin(foe, held)
    assert len(set(g[test])) >= 25
    model, mu, sd = fit_part(seed, x[train], y[train], g[train])
    old = x[test, :23] @ value(seed)
    with torch.no_grad():
        residual = model(torch.from_numpy(((x[test] - mu) / sd).clip(-8, 8)))[:, 0].numpy()
    scores = []
    for k, pred in (("original", old), ("world critic", old + residual)):
        prob = 1 / (1 + np.exp(-pred.clip(-30, 30)))
        scores.append(
            dict(
                seed=seed,
                model=k,
                training_games=len(set(g[train])),
                unseen_pair_games=len(set(g[test])),
                brier=np.average((prob - y[test]) ** 2, weights=1 / np.bincount(g[test])[g[test]]),
            )
        )
    atomic_rows(OUT / "training" / f"critic-{seed}.csv", scores)
    model, mu, sd = fit_part(seed, x, y, g)
    folder = OUT / "weights"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(weights=model.state_dict(), center=torch.from_numpy(mu), scale=torch.from_numpy(sd)),
        folder / f"critic-{seed}.pt",
    )
    print(seed, len(x), scores, flush=True)


@functools.lru_cache(maxsize=4)
def load_critic(seed):
    ck = torch.load(OUT / "weights" / f"critic-{seed}.pt", map_location="cpu", weights_only=True)
    model = network(len(ck["center"]))
    model.load_state_dict(ck["weights"])
    return model.eval(), ck["center"], ck["scale"]


@torch.no_grad()
def leaf_values(worlds, seed, depth):
    shape = (len(worlds), len(worlds[0]), depth)
    leaves = [
        leaf for world in worlds for trace in world for leaf in (trace if depth > 1 else [trace])
    ]
    out = np.zeros(len(leaves))
    ids = []
    states = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, dict):
            ids.append(i)
            states.append(world_features(leaf))
        else:
            out[i] = leaf
    if ids:
        x = torch.from_numpy(np.asarray(states, dtype=np.float32))
        model, mu, sd = load_critic(seed)
        old = x[:, :23] @ torch.tensor(value(seed), dtype=torch.float32)
        out[ids] = torch.sigmoid(old + model(((x - mu) / sd).clamp(-8, 8))[:, 0]).numpy()
    return out.reshape(shape).mean(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "train"))
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    verify()
    if args.stage == "smoke":
        collect((2, 0))
        print("Verified training-world features and returns", flush=True)
    else:
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, _ in enumerate(
                pool.map(collect, [(s, b) for b in range(BLOCKS) for s in SEEDS])
            ):
                if (i + 1) % 32 == 0:
                    print("completed shards", i + 1, flush=True)
        for s in SEEDS:
            fit(s)
