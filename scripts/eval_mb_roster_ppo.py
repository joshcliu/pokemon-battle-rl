"""Evaluate matched roster assignments, using a specialist chosen before scoring."""

import argparse
import csv
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch
from mb_roster_ppo import ARMS, OUT, ROSTERS, RUN, SEEDS, SPEC, create_model, tag, verify_assets
from vgc_posttrain import Match, atomic_rows, batch_obs, eval_job, players

CELLS = ("familiar", "familiar_vs_unseen", "forward", "reverse")


@lru_cache(maxsize=1)
def routing():
    return {r["team"]: r["index"] for r in json.loads((OUT / "specialists.json").read_text())}


def assignments(arm, seed):
    num_games = SPEC["evaluation"]["games_per_pair"]
    start = SPEC["evaluation"]["seed_base"]
    directions = [
        ("familiar", "own_train", "foe_train", 0),
        ("familiar_vs_unseen", "own_train", "foe_test", 1),
        ("forward", "own_test", "foe_test", 2),
        ("reverse", "own_test", "foe_test", 2),
    ]
    for cell, own_key, foe_key, section in directions:
        for i, a in enumerate(ROSTERS[own_key]):
            for j, b in enumerate(ROSTERS[foe_key]):
                own, foe = (b, a) if cell == "reverse" else (a, b)
                offset = start + section * 1_000_000 + i * 10_000 + j * 100
                seats = ("p2", "p1") if cell == "reverse" else ("p1", "p2")
                seeds = [(offset + k, seats[k % 2]) for k in range(num_games)]
                index = routing()[Path(own).stem] if arm == "specialist" else None
                label = "bc" if arm == "bc" else tag(arm, seed, index)
                yield label, "native", own, foe, seeds, "sample", cell


@lru_cache(maxsize=5)
def policy(label):
    if label == "bc":
        return players()[0].policy
    ck = torch.load(RUN / label / "policy.pt", map_location="cpu", weights_only=True)
    model = create_model(ck["arm"])
    model.load_state_dict(ck["state"])
    if ck["arm"] == "moe":
        model.collect_routes = True
    return model.eval()


def score(task):
    model = policy(task[0])

    def metrics(_match):
        if getattr(model, "collect_routes", False):
            counts = model.route_counts.tolist()
            model.route_counts.zero_()
        else:
            counts = [""] * 4
        return {f"expert_{i}": n for i, n in enumerate(counts)}

    return eval_job(task, actor_loader=policy, terminal_metrics=metrics)


def candidates():
    yield "bc", None
    for arm in ARMS:
        for seed in SEEDS:
            yield arm, seed


def name(arm, seed):
    return "bc" if arm == "bc" else tag(arm, seed)


def verify():
    checks = []
    for i, (arm, seed) in enumerate(candidates()):
        tasks = list(assignments(arm, seed))
        saved = read_scores(arm, seed)
        for pos in (i % 32, 64 + (3 * i % 32), 96 + (3 * i % 32)):
            t = list(tasks[pos])
            t[4] = t[4][:2]
            original = {(r["cell"], r["seed"]): r for r in saved}
            for r in score(t):
                assert all(str(v) == original[r["cell"], str(r["seed"])][k] for k, v in r.items())
                checks.append(dict(arm=arm, training_seed=seed, **r))
    atomic_rows(OUT / "replayed.csv", checks)
    verify_sparse_inference()


@torch.no_grad()
def verify_sparse_inference():
    match = Match()
    rows = []
    for seed in SEEDS:
        model = policy(tag("moe", seed))
        for task in list(assignments("moe", seed))[64:96]:
            _, _, own, foe, games, _, _ = task
            simseed, seat = games[0]
            match.reset(own, foe, simseed, seat, preview="native")
            obs = batch_obs(match.obs())
            called = []
            hooks = [
                expert.register_forward_hook(lambda _m, _x, _y, i=i: called.append(i))
                for i, expert in enumerate(model.experts)
            ]
            try:
                actual, _ = model.battle_logits(obs)
            finally:
                for hook in hooks:
                    hook.remove()
            x = model.base.mlp_extractor.forward_actor(model.base.pi_features_extractor(obs))
            logits, ids = model.router(x).topk(2, dim=1)
            weights = logits.softmax(1)
            indices = ids[0].tolist()
            reference = model.base.action_net(x) + sum(
                weights[0, j] * model.experts[i](x) for j, i in enumerate(indices)
            )
            assert sorted(called) == sorted(indices) and len(set(called)) == 2
            torch.testing.assert_close(actual, reference)
            model.route_counts.zero_()
            rows.append(
                dict(
                    training_seed=seed,
                    own=Path(own).stem,
                    foe=Path(foe).stem,
                    expert_1=indices[0],
                    expert_2=indices[1],
                    max_abs_error=(actual - reference).abs().max().item(),
                )
            )
    atomic_rows(OUT / "routing-check.csv", rows)


def read_scores(arm, seed):
    with (RUN / "scores" / (name(arm, seed) + ".csv")).open() as f:
        return list(csv.DictReader(f))


def evaluate(jobs):
    pending = [
        (a, s) for a, s in candidates() if not (RUN / "scores" / (name(a, s) + ".csv")).exists()
    ]
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        while pending:
            ready = None
            for a, s in pending:
                tasks = list(assignments(a, s))
                if a == "bc" or all(
                    (RUN / lb / "policy.pt").is_file() for lb in {t[0] for t in tasks}
                ):
                    ready = a, s
                    break
            if ready is None:
                time.sleep(10)
                continue
            rows = [r for group in pool.map(score, tasks) for r in group]
            assert len(rows) == sum(len(t[4]) for t in tasks)
            atomic_rows(RUN / "scores" / (name(*ready) + ".csv"), rows)
            print(name(*ready), len(rows), flush=True)
            pending.remove(ready)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    verify_assets()
    if args.verify:
        verify()
    else:
        evaluate(args.jobs)
