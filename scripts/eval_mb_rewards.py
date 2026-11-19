"""Score the eight factorial conditions on paired familiar and reciprocal held-out schedules."""

import argparse
import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

from mb_matrix import matrix_tasks
from mb_policy import load_trained
from mb_reward_factorial import RUN, SPEC, verify_assets
from vgc_posttrain import atomic_rows, eval_job, players

from pokemon_rl.rewards import health_deficit, public_board


@lru_cache(maxsize=2)
def model_for(label):
    if label == "bc":
        return players()[0].policy
    return load_trained(RUN / label / "policy.pt")


def terminal_fields(match):
    foes = match.view.opponent_team.values()
    fainted = round(4 * public_board(match.view)[1])
    events = sum(line.startswith("|faint|" + match.foe) for line in match.battle.log)
    # No revival moves on these roster sheets. Reconcile the observable faint count to the replay.
    if fainted != events:
        raise ValueError(f"Faint tally differs for {match.seed}: view={fainted}, log={events}")
    return dict(foe_alive=4 - fainted, foe_hp_remaining=1 - sum(map(health_deficit, foes)) / 4)


def score(task):
    return eval_job(task, actor_loader=model_for, terminal_metrics=terminal_fields)


def schedule(label):
    tasks = []
    for _, _, own, foe, seeds, mode, cell in matrix_tasks("bc"):
        if cell not in ("own_train_foe_train", "own_test_foe_test"):
            continue
        seeds = [(seed + SPEC["evaluation_seed_offset"], side) for seed, side in seeds]
        tasks.append((label, "native", own, foe, seeds, mode, cell))
        if cell == "own_test_foe_test":
            reverse = [(seed, "p2" if side == "p1" else "p1") for seed, side in seeds]
            tasks.append((label, "native", foe, own, reverse, mode, "reciprocal"))
    return tasks


def verify(labels):
    verified = []
    for i, label in enumerate(labels):
        task = schedule(label)[(17 * i) % 96]
        chosen = list(task)
        chosen[4] = task[4][0:2]
        rows = score(chosen)
        with (RUN / "scores" / f"{label}.csv").open() as handle:
            recorded = {(r["cell"], r["seed"]): r for r in csv.DictReader(handle)}
        for r in rows:
            assert all(str(v) == recorded[r["cell"], str(r["seed"])][k] for k, v in r.items())
        verified.extend(rows)
    atomic_rows(RUN / "scores/reproduced.csv", verified)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    verify_assets()
    labels = [
        "bc",
        *(
            f"{a}-{r}-{s}"
            for a in SPEC["optimizers"]
            for r in SPEC["rewards"]
            for s in SPEC["training_seeds"]
        ),
    ]
    if args.verify:
        verify(labels)
    else:
        pending = [label for label in labels if not (RUN / "scores" / f"{label}.csv").exists()]
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            while pending:
                label = next(
                    (x for x in pending if x == "bc" or (RUN / x / "policy.pt").exists()), None
                )
                if label is None:
                    time.sleep(5)
                    continue
                tasks = schedule(label)
                rows = [r for sub in pool.map(score, tasks) for r in sub]
                assert len(rows) == sum(len(t[4]) for t in tasks)
                atomic_rows(RUN / "scores" / f"{label}.csv", rows)
                pending.remove(label)
                print(label, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
