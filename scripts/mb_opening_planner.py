"""Simulate alternative first-turn battle commands from player-visible leads and team sheets."""

import argparse
import copy
import functools
import hashlib
import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_mechanics_es import START, mechanics
from mb_policy_iteration_eval import (
    GAMES,
    OUT,
    RUN,
    SEEDS,
    SPEC,
    decide,
    distribution,
    schedule,
    verify,
)
from mb_public_preview_search import inferred_sheet, public_fixture
from mb_residual_preview import pilot
from poke_env.environment import DoublesEnv
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text


def text_choice(view, action):
    return DoublesEnv.action_to_order(np.asarray(action), view, strict=True).message


@functools.lru_cache(maxsize=1)
def modeling_match():
    match = Match(independent=True)
    match.opponent = copy.copy(match.opponent)
    match.opponent.generator = torch.Generator()
    return match


def condition_opponent(match, leads):
    opponent = match.opponent
    # Build the reserves using a fresh BC draw on the inferred sheet.
    from pokemon_rl.vgcbench import TrainingBC

    def preview(view, seed):
        native = TrainingBC.preview(opponent, view, seed)
        mons = list(view.team.values())
        species = [m.species for m in mons]
        idx = [species.index(m) + 1 for m in leads]
        result = (idx + [i for i in native if i not in idx])[:4]
        assert len(set(result)) == 4
        for i, m in enumerate(mons, start=1):
            m._selected_in_teampreview = i in result
        opponent.lineup = "".join(map(str, result))
        return tuple(result)

    opponent.preview = preview


@torch.no_grad()
def search(own, public, own_line, view, obs, seed):
    policy = pilot(seed)
    actions, logp = distribution(policy, obs)
    native = int(logp.argmax())
    base = actions[native].tolist()
    leads = tuple(m.species if m else "" for m in view.opponent_active_pokemon)
    if len(set(leads)) != 2 or any(x not in {s[0] for s in public} for x in leads):
        return base, 0, False
    if len(actions) == 1:
        return base, 0, False
    # The planner never receives the evaluator seed, opponent decision or private roster.
    identity = json.dumps([team_text(own), public, own_line, leads, seed, "first-turn-v1"])
    rng = np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16))
    ranking = logp.argsort(descending=True).tolist()
    mech = mechanics(view, actions) @ START
    idx = list(dict.fromkeys([native, int((logp.numpy() + mech).argmax()), *ranking]))[:3]
    candidates = [actions[i].tolist() for i in idx]
    names = [text_choice(view, a) for a in candidates]
    match = modeling_match()
    condition_opponent(match, leads)
    counts = np.zeros(len(idx))
    returns = np.zeros(len(idx))
    with tempfile.TemporaryDirectory(dir=RUN) as tmp:
        for rep in range(3):
            path = Path(tmp) / f"guess-{rep}.txt"
            path.write_text(inferred_sheet(public, int(rng.integers(2**31))))
            sim = int(rng.integers(1_000_000, 180_000_000))
            for i, (candidate, name) in enumerate(zip(candidates, names, strict=True)):
                match.reset(own, str(path), sim, ("p1", "p2")[rep % 2], lineup=own_line)
                # Both outer and modeled battles label our commands in the same
                # order. If a modeled ability changes legality, discard that replicate.
                mask = batch_obs(match.obs())
                valid, _ = distribution(policy, mask)
                if not (valid == torch.tensor(candidate)).all(1).any():
                    continue
                if text_choice(match.view, candidate) != name:
                    continue
                assert tuple(m.species for m in match.view.opponent_active_pokemon) == leads
                torch.manual_seed(sim + 700_000_000)
                done, r = match.step(candidate)
                while not done:
                    a = decide(policy, batch_obs(match.obs()), "map", seed)
                    done, r = match.step(a)
                counts[i] += 1
                returns[i] += (1 + r) / 2
    # Require matched worlds across every candidate before overriding the pilot.
    if not (counts == 3).all():
        return base, int(counts.sum()), False
    prior = np.exp(logp[idx].numpy() - logp[native].item())
    chosen = int((returns + 0.75 * prior).argmax())
    return candidates[chosen], int(counts.sum()), chosen != 0


@torch.no_grad()
def score(task):
    seed, cell = task
    i, j, d, own, foe, start = cell
    public = public_fixture(own, foe)
    model, match = pilot(seed), Match()
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]
        match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        # The planning environment owns its process, view and opponent RNG. Preserve
        # the main torch stream too, so only a changed command can affect this game.
        rng_before = match.opponent.generator.get_state()
        with torch.random.fork_rng(devices=[]):
            act, n, changed = search(
                own, public, match.lineup, match.view, batch_obs(match.obs()), seed
            )
        assert torch.equal(rng_before, match.opponent.generator.get_state())
        done, r = match.step(act)
        while not done:
            act = decide(model, batch_obs(match.obs()), "map", seed)
            done, r = match.step(act)
        rows.append(
            dict(
                method="opening-map",
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
                modeled_games=n,
                changed=changed,
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "validation", "custom-dev"))
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        x, y = SPEC["own_train"][0], SPEC["foe_train"][0]
        rows = score((2, (0, 0, 0, x, y, 666000000)))
        assert (
            all(0 <= r["modeled_games"] <= 9 for r in rows)
            and sum(r["modeled_games"] for r in rows) > 36
        )
        print(
            "eight public-state opening searches",
            sum(r["modeled_games"] for r in rows),
            "model games",
            sum(r["changed"] for r in rows),
            "decisions changed",
            flush=True,
        )
    else:
        cells = list(schedule(args.stage))
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for s in SEEDS:
                dest = OUT / "scores" / f"{args.stage}-opening-map-{s}.csv"
                if dest.exists():
                    continue
                rows = [r for chunk in pool.map(score, [(s, c) for c in cells]) for r in chunk]
                atomic_rows(dest, rows)
                print(
                    args.stage,
                    s,
                    len(rows),
                    sum(r["reward"] == 1 for r in rows),
                    sum(r["modeled_games"] for r in rows),
                    flush=True,
                )
