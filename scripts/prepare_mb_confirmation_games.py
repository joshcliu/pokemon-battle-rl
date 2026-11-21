"""Compute independent frozen secondary-test shards in otherwise released worker slots."""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import confirm_mb_policy_iteration as confirm
from mb_policy_iteration_eval import RUN


def work(args):
    split, mode, seed, cell, block = args
    path = f"games/{split}/{mode}/s{seed}-{cell[0]}-{cell[1]}-{cell[2]}-{block}.csv"
    final = confirm.OUT / "confirmation" / path
    if final.exists():
        return
    # The primary auditor may reach this cell concurrently. Publish only complete
    # bytes. The independent writer cannot overwrite a scored shard.
    confirm.DEST = RUN / "confirmation-staging"
    confirm.play(args)
    temp = confirm.DEST / path
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temp, final)
    except FileExistsError:
        assert temp.read_bytes() == final.read_bytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    frozen = json.loads(confirm.LOCK.read_text())
    mode = frozen["method"]
    confirm.freeze(mode)
    assert set(args.seeds) <= set(frozen["seeds"])
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in ("custom", "transfer"):
            for method in ("native", mode):
                tasks = [
                    (split, method, s, cell, block)
                    for s in args.seeds
                    for cell, block in confirm.cells(split)
                ]
                for _ in pool.map(work, tasks):
                    pass
                print(split, method, args.seeds, len(tasks) * 8, "games prepared", flush=True)
