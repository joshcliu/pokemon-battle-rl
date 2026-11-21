"""Compare fitted three-step Bellman targets with terminal returns on the same branch actions."""

import argparse
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_mechanics_es import FEATURES, mechanics
from mb_policy_iteration_eval import OUT, RUN, SEEDS, verify
from mb_rollout_branches import CANDIDATES, PER_SHARD, REPEATS, restore_root, sample_root
from vgc_posttrain import atomic_rows, batch_obs


@torch.no_grad()
def measure(seed, idx, archive):
    match, model, setup, history, raw, choices = sample_root(seed, idx)
    k = len(choices)
    sim = setup[2]
    assert np.array_equal(raw["observation"], archive["obs"])
    assert np.array_equal(choices, archive["actions"][:k])
    restore_root(match, setup, history, raw)
    phi = np.zeros((CANDIDATES, len(FEATURES)))
    phi[:k] = mechanics(match.view, torch.from_numpy(choices))
    v = np.zeros((CANDIDATES, REPEATS))
    for a, choice in enumerate(choices):
        for rep in range(REPEATS):
            restore_root(match, setup, history, raw)
            torch.manual_seed(sim + 83_000_000 + 500_000_000 * rep)
            match.opponent.generator.manual_seed(sim + 95_000_000 + 500_000_000 * rep)
            act = choice.tolist()
            estimates = []
            for step in range(1, 4):
                done, r = match.step(act)
                if done:
                    assert r == archive["returns"][a, rep]
                    target = 0.99 ** (step - 1) * r
                    estimates.append(target)
                    break
                action, value, _ = model(batch_obs(match.obs()))
                if step in (1, 3):
                    estimates.append(0.99**step * np.clip(value.item(), -1, 1))
                act = action[0].tolist()
            v[a, rep] = np.mean(estimates)
    return phi, v


def shard(task):
    seed, block = task
    dest = RUN / "afterstates" / f"seed-{seed}" / f"{block:04}.npz"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with np.load(RUN / "branches" / f"seed-{seed}" / f"{block:04}.npz") as data:
        fields = {k: data[k] for k in ("obs", "actions", "valid", "returns")}
        rows = [
            measure(seed, block * PER_SHARD + j, {k: v[j] for k, v in fields.items()})
            for j in range(PER_SHARD)
        ]
        features, td = map(np.stack, zip(*rows, strict=True))
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, features=features, td=td, mc=fields["returns"], valid=fields["valid"]
        )
        tmp.replace(dest)


def fit(seed):
    columns = {k: [] for k in ("features", "td", "mc", "valid")}
    for b in range(40):
        with np.load(RUN / "afterstates" / f"seed-{seed}" / f"{b:04}.npz") as data:
            for k in columns:
                columns[k].append(data[k])
    phi, td, mc, valid = (np.concatenate(rows) for rows in columns.values())

    def center(x):
        mask = valid.reshape(valid.shape + (1,) * (x.ndim - 2))
        return x - (x * mask).sum(1, keepdims=True) / mask.sum(1, keepdims=True)

    x = center(phi)
    ys = {"td": center(0.7 * td.mean(2) + 0.3 * mc.mean(2)), "terminal": center(mc.mean(2))}
    ids = np.random.default_rng(4209 + seed).permutation(len(valid))
    fold = ids % 5
    logs = []
    for name, y in ys.items():

        def solve(mask, penalty):
            X, Y = x[mask], y[mask]
            scale = np.maximum(np.sqrt((X**2).mean(0)), 0.05)
            z = X / scale
            return np.linalg.solve(z.T @ z + penalty * np.eye(len(scale)), z.T @ Y) / scale

        losses = {}
        for lam in (10.0, 100.0, 500.0, 2000.0):
            held = []
            for f in range(5):
                test = valid & (fold[:, None] == f)
                train = valid & (fold[:, None] != f)
                pred = x[test] @ solve(train, lam)
                held.extend((pred - y[test]) ** 2)
            losses[lam] = np.mean(held)
        lam = min(losses, key=losses.get)
        w = solve(valid, lam)
        dest = RUN / "afterstates" / f"seed-{seed}" / f"{name}.npz"
        np.savez(dest, weights=w, seed=seed)
        for feature, value in zip(FEATURES, w, strict=True):
            logs.append(
                dict(
                    seed=seed,
                    target=name,
                    feature=feature,
                    weight=value,
                    penalty=lam,
                    cv_mse=losses[lam],
                    constant_mse=np.mean(y[valid] ** 2),
                )
            )
    atomic_rows(OUT / "training" / f"afterstate-weights-{seed}.csv", logs)


@functools.lru_cache(maxsize=8)
def weights(seed, target):
    assert target in ("td", "terminal")
    with np.load(RUN / "afterstates" / f"seed-{seed}" / f"{target}.npz") as d:
        assert int(d["seed"]) == seed
        return d["weights"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "train"))
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        shard((2, 0))
    else:
        tasks = [(s, b) for b in range(40) for s in SEEDS]
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, _ in enumerate(pool.map(shard, tasks)):
                if (i + 1) % 20 == 0:
                    print("afterstate shards", i + 1, flush=True)
        for s in SEEDS:
            fit(s)
