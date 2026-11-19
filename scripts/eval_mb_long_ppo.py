"""Score long PPO policies and frozen routing controls on the original matched battles."""

import argparse
import csv
import gzip
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import torch
from eval_mb_roster_ppo import assignments
from mb_policy import FineTune
from mb_roster_ppo import (
    DIVERSE_OUT,
    DIVERSE_RUN,
    LONG_ARMS,
    LONG_OUT,
    LONG_RUN,
    OUT,
    SEEDS,
    create_model,
    diversity_protocol,
    verify_assets,
)
from vgc_posttrain import atomic_rows, eval_job

EARLY = 65_536
FINAL = 262_144
CELLS = ("familiar", "familiar_vs_unseen", "forward", "reverse")


def label(arm, seed, steps, variant):
    return f"{arm}-{seed}-{steps}-{variant}"


def path_for(arm, seed, steps, run=LONG_RUN):
    return run / f"{arm}-{seed}" / ("65k.pt" if steps == EARLY else "policy.pt")


def jobspecs(diverse=False):
    for steps in (FINAL,) if diverse else (EARLY, FINAL):
        for arm in ("shared", "moe") if diverse else LONG_ARMS:
            for variant in ("normal",) if arm == "shared" else ("normal", "rotate", "zero"):
                for seed in SEEDS:
                    yield arm, seed, steps, variant


def is_archived(arm, steps, variant):
    return steps == EARLY and arm in ("shared", "moe") and variant in ("normal", "rotate")


def schedule(arm, seed, steps, variant, *, diverse=False):
    key = label(arm, seed, steps, variant)
    if diverse:
        key = "diverse/" + key
    for _, selector, own, foe, sims, mode, cell in assignments("moe", seed):
        if cell in ("forward", "reverse") or (variant == "normal" and not diverse):
            yield key, selector, own, foe, sims, mode, cell


@lru_cache(maxsize=3)
def actor(key):
    run = DIVERSE_RUN if key.startswith("diverse/") else LONG_RUN
    key = key.removeprefix("diverse/")
    arm, seed, steps, variant = key.split("-")
    ck = torch.load(
        path_for(arm, int(seed), int(steps), run), weights_only=True, map_location="cpu"
    )
    state = ck if int(steps) == EARLY else ck["state"]
    if variant == "zero":
        model = FineTune("native")
        model.load_state_dict({k: v for k, v in state.items() if k.startswith("base.")})
    else:
        model = create_model(arm)
        model.load_state_dict(state)
        if variant == "rotate":
            experts = list(model.experts)
            model.experts = torch.nn.ModuleList(experts[1:] + experts[:1])
        if arm != "shared" and variant == "normal":
            model.collect_routes = True
    return model.eval()


def score(task):
    model = actor(task[0])

    def usage(_match):
        counts = ""
        if getattr(model, "collect_routes", False):
            counts = ",".join(map(str, model.route_counts.tolist()))
            model.route_counts.zero_()
        return {"routes": counts}

    return eval_job(task, actor_loader=actor, terminal_metrics=usage)


def read(file):
    with gzip.open(file, "rt") if file.suffix == ".gz" else file.open() as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=2)
def archived_groups(variant):
    groups = {}
    path = OUT / ("games.csv.gz" if variant == "normal" else "routing-control-games.csv.gz")
    for row in read(path):
        arm = row.get("arm", "moe")
        if arm not in ("shared", "moe"):
            continue
        seed = int(row["training_seed"])
        groups.setdefault((arm, seed), []).append(row)
    return groups


def observations(arm, seed, steps, variant, *, diverse=False):
    if diverse:
        return read(DIVERSE_RUN / "scores" / (label(arm, seed, steps, variant) + ".csv"))
    if is_archived(arm, steps, variant):
        return archived_groups(variant)[arm, seed]
    return read(LONG_RUN / "scores" / (label(arm, seed, steps, variant) + ".csv"))


def evaluate(workers, *, diverse=False):
    run = DIVERSE_RUN if diverse else LONG_RUN
    todo = [
        j
        for j in jobspecs(diverse)
        if not is_archived(j[0], j[2], j[3])
        and not (run / "scores" / (label(*j) + ".csv")).is_file()
    ]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while todo:
            ready = next((j for j in todo if path_for(*j[:3], run=run).is_file()), None)
            if ready is None:
                time.sleep(15)
                continue
            tasks = list(schedule(*ready, diverse=diverse))
            rows = [r for group in pool.map(score, tasks) for r in group]
            assert len(rows) == sum(len(t[4]) for t in tasks)
            atomic_rows(run / "scores" / (label(*ready) + ".csv"), rows)
            print(label(*ready), len(rows), flush=True)
            todo.remove(ready)


def verify(*, diverse=False):
    checks = []
    for n, job in enumerate(jobspecs(diverse)):
        tasks = list(schedule(*job, diverse=diverse))
        task = list(tasks[(n * 17) % len(tasks)])
        task[4] = task[4][:2]
        saved = {(r["cell"], int(r["seed"])): r for r in observations(*job, diverse=diverse)}
        for row in score(task):
            ref = saved[row["cell"], row["seed"]]
            for key in (
                "own",
                "foe",
                "side",
                "selector",
                "mode",
                "lineup",
                "opponent_lineup",
                "reward",
                "turns",
            ):
                assert str(row[key]) == ref[key], (job, row["seed"], key)
            if row["routes"]:
                counts = ref.get("routes") or ",".join(ref[f"expert_{i}"] for i in range(4))
                assert row["routes"] == counts
            checks.append(row)
    atomic_rows((DIVERSE_OUT if diverse else LONG_OUT) / "replayed.csv", checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--diverse", action="store_true")
    args = p.parse_args()
    verify_assets()
    if args.diverse:
        diversity_protocol()
    if args.verify:
        verify(diverse=args.diverse)
    else:
        evaluate(args.jobs, diverse=args.diverse)
