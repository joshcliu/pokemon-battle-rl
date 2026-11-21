"""Fill the frozen policy's public-preview cache independently of scored battles."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from confirm_mb_policy_iteration import LOCK, cells, freeze
from mb_policy_iteration_eval import SEEDS
from mb_public_policy_search import optimize
from mb_public_preview_search import public_fixture, read_plan


def precompute(task):
    mode, seed, own, foe = task
    public = public_fixture(own, foe)
    plan = (
        optimize(own, public, seed)
        if mode.startswith("adaptive-")
        else read_plan(own, public, seed)
    )
    assert plan["seed"] == seed and len(set(plan["selected"])) == 4


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    mode = json.loads(LOCK.read_text())["method"]
    assert mode.startswith(("adaptive-", "search-"))
    freeze(mode)
    tasks = [
        (mode, s, c[3], c[4])
        for split in ("primary", "custom")
        for s in reversed(SEEDS)
        for c, block in cells(split)
        if block == 0
    ]
    # The scorer visits seeds in ascending order, while this allocation starts
    # at the opposite end. Both paths use the same deterministic atomic caches.
    assert len(tasks) == len(set(tasks)) == 576
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for n, _ in enumerate(pool.map(precompute, tasks), 1):
            if n % 32 == 0:
                print(n, "of", len(tasks), "public plans cached", flush=True)
