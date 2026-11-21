"""Transfer the unaltered deployment policies to a separately trained shared PPO opponent."""

import argparse
import copy
import functools
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import FINAL, actor, path_for, read
from mb_policy_iteration_eval import GAMES, OUT, SEEDS, decide, schedule, verify
from mb_public_preview_search import public_fixture, public_sets, read_plan
from mb_residual_preview import pilot
from mb_roster_ppo import LONG_OUT
from poke_env.environment import DoublesEnv
from vgc_posttrain import Match, atomic_rows, batch_obs


@functools.lru_cache(maxsize=1)
def opponent():
    file = path_for("shared", 2, FINAL)
    expected = next(
        r["checkpoint_sha256"]
        for r in read(LONG_OUT / "checkpoints.csv")
        if r["arm"] == "shared" and int(r["training_seed"]) == 2 and int(r["steps"]) == FINAL
    )
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    return actor(f"shared-2-{FINAL}-normal").eval().requires_grad_(False)


@torch.no_grad()
def score(task):
    seed, mode, cell = task
    i, j, d, own, foe, start = cell
    public = (
        public_fixture(own, foe) if mode.startswith("search-") or mode == "adaptive-es" else None
    )
    if mode == "adaptive-es":
        from mb_mechanics_es import action
        from mb_public_policy_search import optimize

        plan = optimize(own, public, seed)
        adapted = np.array(plan["weights"])
    else:
        plan = read_plan(own, public, seed) if public is not None else None
    match = Match()
    match.opponent = copy.copy(match.opponent)
    match.opponent.generator = torch.Generator()
    model = pilot(seed)
    other = opponent()

    def rival(view):
        raw = match.player.embed_battle(view, fake_rating=2000)
        mask = np.asarray(DoublesEnv.get_action_mask(view), dtype=np.int8)
        a = match.opponent.sample(other, batch_obs(dict(observation=raw, action_mask=mask)))[0][0]
        return DoublesEnv.action_to_order(a.numpy(), view, strict=True).message.removeprefix(
            "/choose "
        )

    match.opponent.order = rival
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]
        if plan:

            def choose(view):
                assert public_sets(view) == public
                return plan["selected"]

            match.reset(own, foe, sim, side, preview_picker=choose)
        else:
            match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            obs = batch_obs(match.obs())
            if mode == "adaptive-es":
                a = action(model, obs, match.view, adapted)
            else:
                a = decide(model, obs, mode.removeprefix("search-"), seed, match.view)
            done, r = match.step(a)
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
                opponent="shared-2-262144-normal",
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "modes",
        nargs="+",
        choices=(
            "native",
            "map",
            "search-map",
            "es-trained",
            "search-es-trained",
            "context-ppo",
            "search-context-ppo",
            "endgame-es",
            "belief-es",
            "search-belief-es",
            "multistep-es",
            "adaptive-es",
        ),
    )
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    verify()
    cells = list(schedule("validation"))
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for mode in args.modes:
            for s in SEEDS:
                file = OUT / "scores" / f"transfer-{mode}-{s}.csv"
                if file.exists():
                    continue
                groups = pool.map(score, [(s, mode, c) for c in cells])
                rows = [r for group in groups for r in group]
                assert len(rows) == GAMES * len(cells)
                atomic_rows(file, rows)
                print(
                    "transfer", mode, s, len(rows), sum(r["reward"] == 1 for r in rows), flush=True
                )
