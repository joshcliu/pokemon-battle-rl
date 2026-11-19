"""Replay both seats of one held-out pairing per checkpoint and confirm archived outcomes."""

import csv

from evaluate_vgcbench_mb import check_simulator
from mb_matrix import matrix_tasks
from vgc_posttrain import REPORT, atomic_rows, eval_job

if __name__ == "__main__":
    check_simulator()
    labels = [
        "bc",
        "bc-native",
        *(
            f"{arm}-{seed}"
            for arm in (
                "history",
                "native",
                "flat",
                "structured",
                "native-preview",
                "preview-ppo",
                "preview-grpo",
            )
            for seed in (2, 3, 4, 5)
        ),
    ]
    checked = []
    for index, label in enumerate(labels):
        jobs = [t for t in matrix_tasks(label) if t[-1] == "own_test_foe_test"]
        policy, selector, own, foe, seeds, mode, cell = jobs[index % len(jobs)]
        a = index % 4 * 2
        replay = eval_job((policy, selector, own, foe, seeds[a : a + 2], mode, cell))
        with (REPORT / f"matrix-{label}.csv").open() as f:
            recorded = {r["seed"]: r for r in csv.DictReader(f)}
        for row in replay:
            assert all(str(val) == recorded[str(row["seed"])][key] for key, val in row.items())
        checked.extend(dict(label=label, **r) for r in replay)
    atomic_rows(REPORT / "reproduced.csv", checked)
    print("Reproduced", len(checked), "scored games.")
