"""Frozen expert-identity control using the same held-out battle seeds as the main comparison."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from eval_mb_roster_ppo import assignments, policy, read_scores
from mb_roster_ppo import OUT, SEEDS, verify_assets
from report_mb_posttraining import interval
from report_mb_rewards import archive
from vgc_posttrain import atomic_rows, eval_job


@lru_cache(maxsize=4)
def permuted(label):
    # This script runs in a separate process from the main evaluation. Move the expert modules
    # once. Each original router slot i now dispatches to expert (i+1) mod 4, with the same gate.
    model = policy(label)
    original = list(model.experts)
    model.experts = torch.nn.ModuleList(original[1:] + original[:1])
    model.collect_routes = False
    return model


def run_task(task):
    return eval_job(task, actor_loader=permuted)


def main(jobs):
    raw, deltas, scores = [], [], []
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for seed in SEEDS:
            tasks = [t for t in assignments("moe", seed) if t[-1] in ("forward", "reverse")]
            ev = [r for group in pool.map(run_task, tasks) for r in group]
            lookup = {(r["cell"], int(r["seed"])): r for r in read_scores("moe", seed)}
            assert len(ev) == len({(r["cell"], r["seed"]) for r in ev}) == 1024
            changes = []
            for cell in ("forward", "reverse"):
                rs = sorted((x for x in ev if x["cell"] == cell), key=lambda x: x["seed"])
                before = []
                after = []
                for r in rs:
                    ref = lookup[cell, r["seed"]]
                    assert all(
                        ref[k] == str(r[k])
                        for k in (
                            "policy",
                            "selector",
                            "mode",
                            "own",
                            "foe",
                            "side",
                            "lineup",
                            "opponent_lineup",
                        )
                    )
                    before.append(ref["reward"] == "1")
                    after.append(r["reward"] == 1)
                    raw.append(dict(training_seed=seed, original_reward=ref["reward"], **r))
                changes.append((np.array(after, dtype=float) - before).reshape(4, 8, 16))
                scores.append(
                    dict(
                        training_seed=seed,
                        cell=cell,
                        original_wins=sum(before),
                        permuted_wins=sum(after),
                        games=len(rs),
                    )
                )
            deltas.append((changes[0] + changes[1]) / 2)
            print("Frozen routing control", seed, flush=True)
    arr = np.stack(deltas)
    lo, hi = interval(arr)
    atomic_rows(
        OUT / "routing-control-effect.csv", [dict(delta_pp=100 * arr.mean(), lo95=lo, hi95=hi)]
    )
    atomic_rows(OUT / "routing-control-scores.csv", scores)
    archive(OUT / "routing-control-games.csv.gz", raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=20)
    args = p.parse_args()
    verify_assets()
    main(args.jobs)
