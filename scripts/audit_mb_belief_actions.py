"""Replay one prespecified game per development pairing and intervene on its first changed command."""

import argparse
import csv
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_mechanics_es import action, fitted
from mb_midgame_plan import estimates
from mb_policy_iteration_eval import OUT, SEEDS, schedule, verify
from mb_residual_preview import pilot
from poke_env.battle import Move, Pokemon
from poke_env.environment import DoublesEnv
from vgc_posttrain import Match, atomic_rows, batch_obs


@functools.lru_cache(maxsize=8)
def reference(seed, mode):
    with (OUT / "scores" / f"validation-{mode}-{seed}.csv").open() as f:
        return {int(r["sim"]): r for r in csv.DictReader(f)}


def commands(view, pair):
    joint = DoublesEnv.action_to_order(np.array(pair, dtype=np.int64), view, strict=True)
    parts = []
    for order in (joint.first_order, joint.second_order):
        item = getattr(order, "order", None)
        if isinstance(item, Pokemon):
            parts.append(("switch", item.species, 0))
        elif isinstance(item, Move):
            kind = "attack" if item.category.name != "STATUS" else "status"
            parts.append((kind, item.id, int(order.move_target)))
        else:
            parts.append(("pass", "", 0))
    return joint.message.removeprefix("/choose "), parts


@torch.no_grad()
def replay(cell):
    i, j, d, own, foe, start = cell
    seed = SEEDS[(i + j + 2 * d) % 4]
    k = (2 * i + j + d) % 8
    sim, side = start + k, ("p1", "p2")[k % 2]
    model, match = pilot(seed), Match()
    metadata = dict(seed=seed, sim=sim, side=side, own=Path(own).stem, foe=Path(foe).stem)
    transitions = []
    first = None
    results = {}
    for mode in ("belief-es", "es-trained", "first-change"):
        match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done, step = False, 0
        while not done:
            obs = batch_obs(match.obs())
            if mode == "belief-es":
                actions, values, priors, worlds = estimates(model, obs, match.view, seed, False, 1)
                idx = int((values[:, 0] + priors).argmax())
                chosen = actions[idx]
                if idx:
                    before, prior = commands(match.view, actions[0])
                    after, selected = commands(match.view, chosen)
                    if first is None:
                        first = (step, chosen, actions[0])
                        metadata.update(
                            first_turn=match.view.turn,
                            original_command=before,
                            selected_command=after,
                        )
                    for slot, (old_action, new_action) in enumerate(
                        zip(prior, selected, strict=True)
                    ):
                        if actions[0][slot] == chosen[slot]:
                            continue
                        old_type, old_move, old_target = old_action
                        new_type, new_move, new_target = new_action
                        if old_move == new_move:
                            change = "retarget" if old_target != new_target else "move modifier"
                        else:
                            change = old_type + " to " + new_type
                        transitions.append(
                            dict(
                                **{
                                    key: metadata[key]
                                    for key in ("seed", "sim", "side", "own", "foe")
                                },
                                turn=match.view.turn,
                                decision=step,
                                slot=slot + 1,
                                old_action=before,
                                new_action=after,
                                kind=change,
                                old_move=old_move,
                                new_move=new_move,
                                old_target=old_target,
                                new_target=new_target,
                                modeled_win_delta=values[idx, 0] - values[0, 0],
                                matched_worlds=worlds,
                            )
                        )
            else:
                chosen = action(model, obs, match.view, fitted(seed))
                if mode == "first-change" and first and step == first[0]:
                    assert chosen == first[2]
                    chosen = first[1]
            done, r = match.step(chosen)
            step += 1
        results[mode] = r
        if mode != "first-change":
            ref = reference(seed, mode)[sim]
            assert (int(ref["reward"]), int(ref["turns"]), ref["lineup"], ref["opp_lineup"]) == (
                r,
                match.battle.turn,
                match.lineup,
                match.opponent.lineup,
            )
    metadata.setdefault("first_turn", "")
    metadata.setdefault("original_command", "")
    metadata.setdefault("selected_command", "")
    metadata.update(
        base_reward=results["es-trained"],
        planner_reward=results["belief-es"],
        intervention_reward=results["first-change"],
    )
    return metadata, transitions


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    verify()
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(replay, list(schedule("validation"))))
    games, changes = zip(*records, strict=True)
    assert len(games) == 64 and len({(g["own"], g["foe"]) for g in games}) == 64
    atomic_rows(OUT / "first-change-replays.csv", games)
    atomic_rows(OUT / "changed-commands.csv", [r for game in changes for r in game])
    print(
        "Verified",
        len(games),
        "matched game triples and",
        sum(map(len, changes)),
        "changed slot commands",
        flush=True,
    )
