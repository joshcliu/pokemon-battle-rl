"""Replay the published 12-team MC461 schedule with each BC-initialized native PPO seed."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from evaluate_vgcbench_mb import check_simulator
from vgc_posttrain import REPORT, atomic_rows, development_tasks, eval_job

if __name__ == "__main__":
    check_simulator()
    with ProcessPoolExecutor(12, mp_context=multiprocessing.get_context("spawn")) as pool:
        for train_seed in (2, 3, 4, 5):
            label = f"native-{train_seed}"
            out = REPORT / f"dev-{label}.csv"
            if out.exists():
                continue
            tasks = development_tasks(label, "fixed")
            rows = [r for batch in pool.map(eval_job, tasks) for r in batch]
            assert len(rows) == 768
            atomic_rows(out, rows)
            print(label, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
