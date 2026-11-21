"""Learn how far to trust a public simulator rollout from matched training-game returns."""

import argparse
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_mechanics_es import action, fitted
from mb_midgame_plan import choose, estimates, projection, valid
from mb_policy_iteration_eval import OUT, RUN, SEEDS, SPEC, verify
from mb_residual_preview import pilot
from mb_rollout_branches import pairing, restore_root
from vgc_posttrain import Match, atomic_rows, batch_obs

BLOCKS = 16
GAMES = 14
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def mixture(lam):
    return np.array([1 - lam, lam * (1 - lam), lam * lam])


def path(seed):
    return RUN / "multistep" / f"seed-{seed}"


@torch.no_grad()
def collect(task):
    seed, block = task
    dest = path(seed) / f"{block:02}.npz"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    match, model, weights = Match(), pilot(seed), fitted(seed)
    records = []
    for g in range(block * GAMES, (block + 1) * GAMES):
        own, foe, seat = pairing(seed, g)
        if g % 2:
            own, foe = foe, own
        sim = 941_000_000 + 1_000_000 * seed + g
        setup = own, foe, sim, seat
        match.reset(*setup, preview="native")
        rng = np.random.default_rng([8195, seed, g])
        history, root, n = [], None, 0
        done = False
        while not done:
            raw = match.obs()
            if not any(match.view.force_switch) and valid(projection(match.view)):
                n += 1
                if rng.integers(n) == 0:
                    root = (list(history), raw)
            a = action(model, batch_obs(raw), match.view, weights)
            done, _ = match.step(a)
            history.append(a)
        if root is None:
            continue
        prefix, raw = root
        restore_root(match, setup, prefix, raw)
        acts, values, prior, worlds = estimates(model, batch_obs(raw), match.view, seed, False, 3)
        if len(acts) < 2:
            continue
        assert 2 <= worlds <= 4
        targets = []
        for candidate in acts:
            outcomes = []
            for rep in range(2):
                restore_root(match, setup, prefix, raw)
                match.opponent.generator.manual_seed(sim + 400_000_000 + 100_000_000 * rep)
                done, r = match.step(candidate)
                while not done:
                    a = action(model, batch_obs(match.obs()), match.view, weights)
                    done, r = match.step(a)
                outcomes.append((r + 1) / 2)
            targets.append(np.mean(outcomes))
        for values_at_horizons, policy_prior, y in zip(values, prior, targets, strict=True):
            records.append((g, Path(own).stem, values_at_horizons, policy_prior, y))
    # Some positions have only one available action. Preserve empty shard shape.
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        game=[r[0] for r in records],
        team=np.array([r[1] for r in records], dtype=str),
        forecasts=np.array([r[2] for r in records]).reshape(-1, 3),
        prior=[r[3] for r in records],
        y=[r[4] for r in records],
    )
    tmp.replace(dest)


def fit(seed):
    parts = {k: [] for k in ("game", "team", "forecasts", "prior", "y")}
    for block in range(BLOCKS):
        with np.load(path(seed) / f"{block:02}.npz") as data:
            for k in parts:
                parts[k].append(data[k])
    game, team, forecast, prior, y = (np.concatenate(xs) for xs in parts.values())
    ids = np.unique(game)
    rosters = sorted({Path(t).stem for t in SPEC["own_train"] + SPEC["foe_train"]})
    folds = np.random.default_rng(83218 + seed).permutation(len(rosters)) % 5
    errors, gains, splits = [], [], []
    for g in ids:
        idx = game == g
        assert len(set(team[idx])) == 1 and idx.sum() >= 2
        splits.append(folds[rosters.index(team[idx][0])])
        pred = np.column_stack([forecast[idx] @ mixture(lam) for lam in LAMBDAS])
        diffs = pred - y[idx, None]
        errors.append(np.mean((diffs - diffs.mean(0)) ** 2, axis=0))
        selected = (pred + prior[idx, None]).argmax(0)
        gains.append(y[idx][selected] - y[idx][selected[0]])
    assert len(ids) >= 100 and set(splits) == set(range(5))
    errors, gains, splits = map(np.array, (errors, gains, splits))
    k = errors.mean(0).argmin()
    output = []
    for fold in range(5):
        train = splits != fold
        held = ~train
        j = errors[train].mean(0).argmin()
        output.append(
            dict(
                seed=seed,
                fold=fold,
                states=held.sum(),
                selected_lambda=LAMBDAS[j],
                first_step_mse=errors[held, 0].mean(),
                selected_mse=errors[held, j].mean(),
                selected_minus_one_step=gains[held, j].mean(),
            )
        )
    atomic_rows(OUT / "training" / f"multistep-{seed}.csv", output)
    np.savez(path(seed) / "horizon.npz", weights=mixture(LAMBDAS[k]), seed=seed, roots=len(ids))
    print("fit multistep", seed, len(ids), LAMBDAS[k], flush=True)


@functools.lru_cache(maxsize=4)
def weights(seed):
    with np.load(path(seed) / "horizon.npz") as data:
        assert int(data["seed"]) == seed
        return tuple(data["weights"])


def deploy(model, obs, view, seed):
    w = weights(seed)
    if w[0] == 1:
        return choose(model, obs, view, seed, False)[0]
    selected, n, _ = choose(model, obs, view, seed, False, weights=w)
    if n < 2 and not any(view.force_switch):
        return choose(model, obs, view, seed, False)[0]
    return selected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "train"))
    p.add_argument("--jobs", type=int, default=5)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        collect((2, 0))
        with np.load(path(2) / "00.npz") as d:
            assert len(np.unique(d["game"])) >= 6 and np.isfinite(d["forecasts"]).all()
            print("training-only depth-three states", len(set(d["game"])), flush=True)
    else:
        tasks = [(s, b) for b in range(BLOCKS) for s in SEEDS]
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for j, _ in enumerate(pool.map(collect, tasks)):
                if (j + 1) % 16 == 0:
                    print("multistep shards", j + 1, flush=True)
        for s in SEEDS:
            fit(s)
