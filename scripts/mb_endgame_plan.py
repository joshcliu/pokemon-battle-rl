"""Choose late-game joint commands by backing up a learned terminal value through Showdown."""

import argparse
import atexit
import functools
import hashlib
import json
import multiprocessing
import selectors
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_endgame_value import no_reserves, public_position, value_features
from mb_mechanics_es import fitted, mechanics
from mb_opening_planner import text_choice
from mb_policy_iteration_eval import GAMES, OUT, RUN, SEEDS, SPEC, distribution, schedule, verify
from mb_residual_preview import pilot
from vgc_posttrain import ROOT, Match, atomic_rows, batch_obs

from pokemon_rl.battle import node_binary
from pokemon_rl.vgcbench import CACHE


class Forecaster:
    def __init__(self):
        self.process = subprocess.Popen(
            [node_binary(), str(ROOT / "simulator/public_endgame.cjs"), str(CACHE / "showdown")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()

    def __call__(self, position, orders):
        body = json.dumps(position, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(body.encode()).hexdigest()
        self.process.stdin.write(json.dumps(dict(position=position, orders=orders, key=key)) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(90):
            raise TimeoutError("Endgame forecaster timed out")
        result = json.loads(self.process.stdout.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["worlds"]


@functools.lru_cache(maxsize=1)
def forecaster():
    return Forecaster()


@functools.lru_cache(maxsize=4)
def value_model(seed):
    return np.load(RUN / "endgame" / f"seed-{seed}" / "value.npy")


def decision(model, obs, view, seed, check=False):
    choices, logp = distribution(model, obs)
    phi = mechanics(view, choices)
    scores = logp.numpy() + phi @ fitted(seed)
    ranked = scores.argsort()[::-1].tolist()
    base = choices[ranked[0]].tolist()
    if not no_reserves(view) or len(choices) == 1:
        return base, False, 0, False
    ids = list(dict.fromkeys([ranked[0], int(logp.argmax()), int(phi[:, 0].argmax()), *ranked]))[:4]
    orders = [text_choice(view, choices[i].tolist()).removeprefix("/choose ") for i in ids]
    projected = public_position(view)
    # No private opponent stats, evaluation simulator state, true damage RNG or
    # unobserved reserves cross the boundary into the forecasting process.
    assert all(not m or "stats" not in m for m in projected["foe"])
    worlds = forecaster()(projected, orders)
    if len(worlds) < 2:
        return base, True, len(worlds), False
    # A unit-test value may precede collecting terminal labels; never score it
    # as an evaluation configuration.
    beta = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) if check else value_model(seed)

    def pwin(leaf):
        if isinstance(leaf, (int, float)):
            return leaf
        return 1 / (1 + np.exp(-np.clip(value_features(leaf) @ beta, -30, 30)))

    values = np.mean([[pwin(leaf) for leaf in group] for group in worlds], axis=0)
    improved = np.argmax(values + 0.025 * (scores[ids] - scores[ids[0]]))
    return choices[ids[improved]].tolist(), True, len(worlds), bool(improved)


@torch.no_grad()
def score(task):
    seed, cell, check = task
    i, j, d, own, foe, start = cell
    match, model = Match(), pilot(seed)
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]
        match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done = False
        eligible = worlds = changed = 0
        while not done:
            act, at_end, n, switch = decision(
                model, batch_obs(match.obs()), match.view, seed, check
            )
            eligible += at_end
            worlds += n
            changed += switch
            done, r = match.step(act)
        rows.append(
            dict(
                method="endgame-es",
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
                eligible_decisions=eligible,
                matched_worlds=worlds,
                changed_commands=changed,
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "validation", "custom-dev"))
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        x, y = SPEC["own_train"][0], SPEC["foe_train"][0]
        rows = score((2, (0, 0, 0, x, y, 666000000), True))
        assert sum(r["matched_worlds"] for r in rows) > 0
        print(
            [(r["eligible_decisions"], r["matched_worlds"], r["changed_commands"]) for r in rows],
            flush=True,
        )
    else:
        cells = list(schedule(args.stage))
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for s in SEEDS:
                dest = OUT / "scores" / f"{args.stage}-endgame-es-{s}.csv"
                if dest.exists():
                    continue
                rows = [
                    r for batch in pool.map(score, [(s, c, False) for c in cells]) for r in batch
                ]
                atomic_rows(dest, rows)
                print(args.stage, s, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
