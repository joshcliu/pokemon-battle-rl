"""Reuse a worker pool to evaluate all completed pilot checkpoints."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from evaluate_vgcbench_mb import check_simulator
from mb_matrix import matrix_tasks
from vgc_posttrain import REPORT, atomic_rows, eval_job

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    check_simulator()
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for label in args.labels:
            output = REPORT / f"matrix-{label}.csv"
            if output.exists():
                continue
            tasks = matrix_tasks(label)
            rows = [r for batch in pool.map(eval_job, tasks) for r in batch]
            assert len(rows) == sum(len(t[4]) for t in tasks)
            atomic_rows(output, rows)
            print(label, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
