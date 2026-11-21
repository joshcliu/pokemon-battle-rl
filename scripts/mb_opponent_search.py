"""Model-based policy improvement using plausible opponent lineups and neural replies."""

import argparse
import atexit
import copy
import csv
import functools
import hashlib
import json
import multiprocessing
import os
import selectors
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_endgame_plan import Forecaster
from mb_mechanics_es import fitted, mechanics
from mb_midgame_plan import backed_up_values, projection, valid, value
from mb_opening_planner import text_choice
from mb_opponent_belief import OUT, ROOT, private_prior, public_beliefs
from mb_policy_iteration_eval import SEEDS, SPEC, distribution, schedule, verify
from mb_public_policy_search import optimize
from mb_public_preview_search import public_fixture, public_sets
from mb_residual_preview import pilot
from vgc_posttrain import Match, atomic_rows, batch_obs

from pokemon_rl.battle import node_binary
from pokemon_rl.vgcbench import CACHE


class ReplyForecaster(Forecaster):
    def __init__(self):
        self.process = subprocess.Popen(
            [
                node_binary(),
                str(ROOT / "simulator/public_reply_search.cjs"),
                str(CACHE / "showdown"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        atexit.register(self.close)

    def forecast(self, pos, orders, beliefs, depth, salt="", partial=False, operation="values"):
        key = hashlib.sha256(
            (salt + json.dumps(pos, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()
        packet = dict(
            position=pos,
            orders=orders,
            beliefs=beliefs,
            samples=len(beliefs),
            depth=depth,
            key=key,
            partial=partial,
            operation=operation,
        )
        self.process.stdin.write(json.dumps(packet) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(90):
            raise TimeoutError("Opponent model did not return forecasts")
        result = json.loads(self.process.stdout.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["worlds"]


@functools.lru_cache(maxsize=1)
def engine():
    return ReplyForecaster()


# Width, public-world samples and learned value lookahead.
CONFIGS = {
    "reply": (4, 4, (1.0,)),
    "reply-wide": (8, 8, (1.0,)),
    "reply-deep": (8, 8, (0.45, 0.30, 0.25)),
    "coordinate": (8, 12, (1.0,)),
    "coordinate-critic": (8, 12, (1.0,)),
    "reply-critic": (8, 8, (1.0,)),
}


@torch.no_grad()
def act(model, obs, view, seed, prior, mode):
    width, n, weights = CONFIGS[mode]
    pairs, logp = distribution(model, obs)
    phi = mechanics(view, pairs)
    scores = logp.numpy() + phi @ fitted(seed)
    ranked = scores.argsort()[::-1].tolist()
    base = pairs[ranked[0]].tolist()
    if len(pairs) < 2 or any(view.force_switch):
        return base, 0
    pos = projection(view)
    if not valid(pos):
        return base, 0
    if mode.startswith("coordinate"):
        from mb_coordinate_search import coordinate_search

        return coordinate_search(engine(), view, pos, seed, prior, pairs.numpy(), scores, mode)
    ids = list(dict.fromkeys([ranked[0], int(logp.argmax()), int(phi[:, 0].argmax()), *ranked]))[:4]
    acts = pairs.numpy()
    while len(ids) < min(width, len(ranked)):
        distance = (acts[:, None, :] != acts[ids][None, :, :]).sum(2).min(1)
        diverse = scores + 0.8 * distance
        diverse[ids] = -np.inf
        ids.append(int(diverse.argmax()))
    orders = [text_choice(view, acts[j].tolist()).removeprefix("/choose ") for j in ids]
    beliefs = public_beliefs(view, pos, prior, n)
    worlds = engine().forecast(pos, orders, beliefs, len(weights))
    if len(worlds) < 2:
        return base, len(worlds)
    if mode == "reply-critic":
        from mb_world_value import leaf_values

        values = leaf_values(worlds, seed, len(weights)) @ weights
    else:
        values = backed_up_values(worlds, value(seed), len(weights)) @ weights
    best = int((values + 0.025 * (scores[ids] - scores[ids[0]])).argmax())
    return acts[ids[best]].tolist(), len(worlds)


@torch.no_grad()
def score(task, *, transfer=False):
    seed, cell, mode, games = task
    i, j, d, own, foe, start = cell
    public = public_fixture(own, foe)
    prior = private_prior(own, public, seed)
    plan = optimize(own, public, seed)
    model, match = pilot(seed), Match()
    if transfer:
        from mb_policy_transfer import opponent
        from poke_env.environment import DoublesEnv

        match.opponent = copy.copy(match.opponent)
        match.opponent.generator = torch.Generator()
        other = opponent()

        def rival(view):
            raw = match.player.embed_battle(view, fake_rating=2000)
            mask = np.asarray(DoublesEnv.get_action_mask(view), dtype=np.int8)
            chosen = match.opponent.sample(
                other, batch_obs(dict(observation=raw, action_mask=mask))
            )[0][0]
            return DoublesEnv.action_to_order(
                chosen.numpy(), view, strict=True
            ).message.removeprefix("/choose ")

        match.opponent.order = rival

    def pick(view):
        assert public_sets(view) == public
        return plan["selected"]

    rows = []
    for k in range(games):
        sim, side = start + k, ("p1", "p2")[k % 2]
        match.reset(own, foe, sim, side, preview_picker=pick)
        torch.manual_seed(sim + 700_000_000)
        done = False
        total = 0
        while not done:
            action, n = act(model, batch_obs(match.obs()), match.view, seed, prior, mode)
            total += n
            done, r = match.step(action)
        rows.append(
            dict(
                method=mode,
                training_seed=seed,
                i=i,
                j=j,
                direction=d,
                k=k,
                own=Path(own).stem,
                foe=Path(foe).stem,
                sim=sim,
                side=side,
                lineup=match.lineup,
                opp_lineup=match.opponent.lineup,
                reward=r,
                turns=match.battle.turn,
                matched_worlds=total,
                opponent="shared-2-262144-normal" if transfer else "released-bc",
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "validation", "custom-dev"))
    p.add_argument("modes", nargs="+", choices=CONFIGS)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        cell = (0, 0, 0, SPEC["own_train"][0], SPEC["foe_train"][0], 667100000)
        for mode in args.modes:
            rows = score((2, cell, mode, 2))
            assert sum(r["matched_worlds"] for r in rows) > 8
            print(mode, [(r["reward"], r["matched_worlds"]) for r in rows], flush=True)
    else:
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for mode in args.modes:
                for seed in args.seeds:
                    dest = OUT / "scores" / f"{args.stage}-{mode}-{seed}.csv"
                    if dest.exists():
                        continue
                    cells = list(schedule(args.stage))
                    rows = [
                        r
                        for batch in pool.map(score, [(seed, c, mode, 8) for c in cells])
                        for r in batch
                    ]
                    assert len(rows) == len(cells) * 8
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
                        tmp = Path(staging) / dest.name
                        atomic_rows(tmp, rows)
                        try:
                            os.link(tmp, dest)
                        except FileExistsError:
                            with dest.open() as f:
                                existing = list(csv.DictReader(f))
                            assert len(existing) == len(rows)
                            assert all(
                                all(str(r[k]) == v for k, v in old.items())
                                for r, old in zip(rows, existing, strict=True)
                            )
                    print(
                        args.stage,
                        mode,
                        seed,
                        len(rows),
                        sum(r["reward"] == 1 for r in rows),
                        flush=True,
                    )
