"""Swap the assigned rosters in the unseen/unseen cell, keeping each roster in the same seat."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from evaluate_vgcbench_mb import check_simulator
from mb_matrix import matrix_tasks
from vgc_posttrain import REPORT, atomic_rows, eval_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="*",
        default=[
            "bc",
            "bc-native",
            *(
                f"{arm}-{seed}"
                for arm in ("native-preview", "preview-ppo", "preview-grpo")
                for seed in (2, 3, 4, 5)
            ),
        ],
    )
    args = parser.parse_args()
    check_simulator()
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        for label in args.labels:
            output = REPORT / f"reverse-{label}.csv"
            if output.exists():
                continue
            tasks = []
            for policy, selector, own, foe, seeds, mode, cell in matrix_tasks(label):
                if cell == "own_test_foe_test":
                    switched = [(s, "p1" if side == "p2" else "p2") for s, side in seeds]
                    tasks.append((policy, selector, foe, own, switched, mode, "reciprocal"))
            rows = [r for batch in pool.map(eval_job, tasks) for r in batch]
            assert len(rows) == 256
            atomic_rows(output, rows)
            print(label, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
