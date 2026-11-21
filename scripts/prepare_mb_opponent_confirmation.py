"""Reuse available workers on the same frozen matchup schedules with atomic publication."""

import argparse
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import confirm_mb_opponent_search as confirm
from mb_opponent_belief import RUN
from run_mb_itemless_confirmation import install, lock


def work(task):
    split, mode, seed, cell = task
    i, j, d, *_ = cell
    for m in ("native", confirm.OLD, mode):
        final = confirm.shard(split, m, seed, i, j, d)
        if final.exists():
            continue
        with tempfile.TemporaryDirectory(dir=RUN) as temp:
            original = confirm.RUN
            try:
                confirm.RUN = Path(temp)
                produced = confirm.play((split, m, seed, cell))
            finally:
                confirm.RUN = original
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(produced, final)
            except FileExistsError:
                assert produced.read_bytes() == final.read_bytes()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=confirm.SPLITS, required=True)
    p.add_argument("--seeds", nargs="+", type=int, choices=confirm.SEEDS, required=True)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    mode = json.loads(confirm.LOCK.read_text())["method"]
    confirm.freeze(mode)
    lock()
    tasks = [
        (args.split, mode, s, c)
        for s in args.seeds
        for c in reversed(list(confirm.cells(args.split)))
    ]
    with ProcessPoolExecutor(
        args.jobs, mp_context=multiprocessing.get_context("spawn"), initializer=install
    ) as pool:
        for k, _ in enumerate(pool.map(work, tasks)):
            if (k + 1) % 32 == 0:
                print(args.split, args.seeds, k + 1, "/", len(tasks), flush=True)
