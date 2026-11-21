"""Branch from training games to measure terminal returns of different legal joint actions."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_policy_iteration_eval import RUN, SEEDS, SPEC, distribution, verify
from mb_residual_preview import pilot
from vgc_posttrain import Match, batch_obs

PER_SHARD = 14
CANDIDATES = 4
REPEATS = 2


def pairing(seed, index):
    pairs = [(x, y) for x in SPEC["own_train"] for y in SPEC["foe_train"]]
    epoch, pos = divmod(index, len(pairs))
    order = np.random.default_rng([71933, seed, epoch]).permutation(len(pairs))
    pair_id = int(order[pos])
    return *pairs[pair_id], ("p1", "p2")[(pair_id + epoch + seed) % 2]


@torch.no_grad()
def sample_root(seed, index):
    own, foe, side = pairing(seed, index)
    model = pilot(seed)
    sim = 410_000_000 + seed * 1_000_000 + index
    match = Match()
    match.reset(own, foe, sim, side, preview="native")
    torch.manual_seed(sim + 70_000_000)
    # Keep the last multi-action state if a short game ends before the sampled depth.
    rng = np.random.default_rng([91823, seed, index])
    horizon = int(rng.choice([0, 1, 2, 3, 4, 5, 7, 9, 12]))
    root, history = None, []
    for t in range(horizon + 1):
        raw = match.obs()
        obs = batch_obs(raw)
        actions, logp = distribution(model, obs)
        if len(actions) >= 2:
            root = (list(history), raw, actions, logp)
        if t == horizon:
            break
        action = model(obs)[0][0].tolist()
        done, _ = match.step(action)
        history.append(action)
        if done:
            break
    if root is None:
        raise ValueError(f"No multi-action state in {sim}")
    history, raw, actions, logp = root
    order = torch.argsort(logp, descending=True)
    k = min(len(actions), CANDIDATES)
    best = order[:1].tolist()
    rest = order[1:]
    # Give each legal alternative a chance and preserve the prior mode as a control.
    w = (logp[rest] * 0.65).softmax(0).numpy().astype(float)
    w = 0.9 * w / w.sum() + 0.1 / len(w)
    indices = best + rng.choice(rest.numpy(), k - 1, replace=False, p=w / w.sum()).tolist()
    choices = actions[indices].numpy()
    return match, model, (own, foe, sim, side), history, raw, choices


def restore_root(match, setup, history, raw):
    match.reset(*setup, preview="native")
    for old in history:
        assert not match.step(old)[0]
    replay = match.obs()
    assert all(np.array_equal(raw[key], replay[key]) for key in raw)


@torch.no_grad()
def one(seed, index, *, audit=False):
    match, model, setup, history, raw, choices = sample_root(seed, index)
    own, foe, sim, side = setup
    k = len(choices)
    returns = np.zeros((CANDIDATES, REPEATS), dtype=np.int8)

    def rollout(action, rep):
        restore_root(match, setup, history, raw)
        # Same hidden world within this training root. Independent learner
        # and opponent continuation randomness per repeat, shared across choices.
        torch.manual_seed(sim + 83_000_000 + 500_000_000 * rep)
        match.opponent.generator.manual_seed(sim + 95_000_000 + 500_000_000 * rep)
        done, r = match.step(action.tolist())
        while not done:
            act = model(batch_obs(match.obs()))[0][0].tolist()
            done, r = match.step(act)
        return r, match.battle.turn

    for a, choice in enumerate(choices):
        for rep in range(REPEATS):
            value, turns = rollout(choice, rep)
            returns[a, rep] = value
            if audit and a == rep == 0:
                assert rollout(choice, rep) == (value, turns)
    # Fixed-width batches keep explicit valid bits for states with 2 or 3 actions.
    pad = np.pad(choices, ((0, CANDIDATES - k), (0, 0)), mode="edge")
    return dict(
        obs=raw["observation"],
        actions=pad,
        valid=np.arange(CANDIDATES) < k,
        returns=returns,
        depth=len(history),
        sim=sim,
        own=own,
        foe=foe,
        side=side,
    )


def shard(task):
    seed, block = task
    folder = RUN / "branches" / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{block:04}.npz"
    if dest.exists():
        return seed, block
    rows = [one(seed, i, audit=i == 0) for i in range(block * PER_SHARD, (block + 1) * PER_SHARD)]
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
    tmp.replace(dest)
    return seed, block


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    verify()
    tasks = [(s, b) for b in range(args.blocks) for s in SEEDS]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for s, b in pool.map(shard, tasks):
            print("verified branch shard", s, b, flush=True)
