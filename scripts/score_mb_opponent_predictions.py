"""Compare sampled public-model replies to the observed next action in training-only battles."""

import argparse
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_mechanics_es import action, fitted
from mb_midgame_plan import projection, valid
from mb_opponent_belief import OUT, possible_bench, private_prior, public_beliefs
from mb_opponent_search import engine
from mb_public_preview_search import public_fixture
from mb_residual_preview import pilot
from mb_rollout_branches import pairing
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text


def command(text):
    segments = []
    for part in text.removeprefix("/choose ").lower().strip().split(","):
        words = part.strip().split()
        if words[0] == "move":
            target = next((s for s in words[2:] if s in ("-2", "-1", "1", "2")), "0")
            segments.append(("move", to_id_str(words[1]), target, "mega" in words))
        elif words[0] == "switch":
            segments.append(("switch", to_id_str(" ".join(words[1:])), "0", False))
        else:
            segments.append((words[0], "", "0", False))
    return segments


@torch.no_grad()
def game(task):
    seed, index = task
    own, foe, side = pairing(seed, 2400 + index)
    if index % 2:
        own, foe = foe, own
    prior = private_prior(own, public_fixture(own, foe), seed)
    match = Match()
    sim = 135000000 + seed * 1000000 + index
    match.reset(own, foe, sim, side, preview="native")
    rows = []
    bench = []
    done = False
    while not done:
        v = match.view
        pos = projection(v)
        if valid(pos) and not any(v.force_switch) and match.battle.needs_action(match.foe):
            beliefs = public_beliefs(v, pos, prior, 8)
            preds = engine().forecast(pos, [], beliefs, 1, operation="replies")
            # Use the true view only after model forecasts have been written and
            # only for the outcome labels in this training-roster diagnostic.
            hidden = match.battle.players[match.foe]
            if v.turn == 1 and pos["vacancies"] == 2 and len(pos["unknown"]) == 4:
                enemies = list(v.opponent_team.values())
                choices, p = possible_bench(pos, enemies, prior["profiles"][0], prior)
                roster = Teambuilder.parse_showdown_team(team_text(foe))
                selected = {
                    re.sub(
                        r"mega[xy]?$",
                        "",
                        to_id_str(roster[int(k) - 1].species or roster[int(k) - 1].nickname),
                    )
                    for k in match.opponent.lineup
                }
                chosen_species = {
                    m.species for m in hidden.team.values() if m.selected_in_teampreview
                }
                assert len(selected) == 4 and selected == chosen_species, (selected, chosen_species)
                assert sum(m["species"] in selected for m in pos["unknown"]) == 2
                for m in pos["unknown"]:
                    n = m["name"]
                    truth = float(m["species"] in selected)
                    prob = sum(w for c, w in zip(choices, p, strict=True) if n in c)
                    bench.append(
                        dict(
                            seed=seed, game=index, name=n, target=truth, posterior=prob, uniform=0.5
                        )
                    )
            rng = match.opponent.generator.get_state()
            target = command(match.opponent.order(hidden))
            match.opponent.generator.set_state(rng)
            if len(target) == 2:
                for k, arm in enumerate(("neural", "heuristic")):
                    votes = [command(pair[k]) for pair in preds]
                    votes = [r for r in votes if len(r) == 2]
                    if votes:
                        rows.append(
                            dict(
                                seed=seed,
                                game=index,
                                turn=v.turn,
                                model=arm,
                                worlds=len(votes),
                                slot_match=np.mean(
                                    [
                                        float(x == y)
                                        for r in votes
                                        for x, y in zip(r, target, strict=True)
                                    ]
                                ),
                                move_or_switch_match=np.mean(
                                    [
                                        float(x[:2] == y[:2])
                                        for r in votes
                                        for x, y in zip(r, target, strict=True)
                                    ]
                                ),
                                joint_match=np.mean([r == target for r in votes]),
                            )
                        )
        a = action(pilot(seed), batch_obs(match.obs()), v, fitted(seed))
        done, _ = match.step(a)
    return rows, bench


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(game, [(s, i) for s in (2, 3, 4, 5) for i in range(8)]))
    rows = [r for a, _ in results for r in a]
    bench = [r for _, b in results for r in b]
    assert len(bench) >= 96 and len(rows) > 100 and len(rows) % 2 == 0
    atomic_rows(OUT / "reply-predictions.csv", rows)
    atomic_rows(OUT / "reserve-predictions.csv", bench)
    for arm in ("neural", "heuristic"):
        group = [r for r in rows if r["model"] == arm]
        print(
            arm,
            len(group),
            *(
                np.mean([r[c] for r in group])
                for c in ("slot_match", "move_or_switch_match", "joint_match")
            ),
            flush=True,
        )
    print(
        "reserve Brier",
        *(np.mean([(r[k] - r["target"]) ** 2 for r in bench]) for k in ("posterior", "uniform")),
        flush=True,
    )
