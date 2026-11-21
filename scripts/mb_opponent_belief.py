"""Predict the other player's reserves and replies from an open sheet and public battle state."""

import copy
import functools
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from mb_policy_iteration_eval import ROOT
from mb_public_policy_search import estimate
from mb_public_preview_search import key_for
from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder
from vgc_posttrain import Match, players

OUT = ROOT / "results/sketch5-mb-opponent-search"
RUN = ROOT / "runs/sketch5-mb-opponent-search"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def public_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def index_name(mon, profiles):
    for label in (mon.name, mon.species, mon.base_species):
        k = to_id_str(label)
        if k in profiles:
            return k
    raise ValueError(f"No model sheet for {mon.name}/{mon.species}")


def private_prior(own, public, seed):
    # Matchups use sorted open-sheet attributes. The actual opponent roster path,
    # game seed and submitted lineup never enter this interface.
    key = key_for(own, public, seed)
    file = RUN / "public-prior" / (key + ".json")
    if file.exists():
        result = json.loads(file.read_text())
        assert result["key"] == key
        return result
    rng = np.random.default_rng(public_seed("reply-v1" + key))
    profiles, draws = [], []
    file.parent.mkdir(parents=True, exist_ok=True)
    model = Match(independent=True)
    try:
        with tempfile.TemporaryDirectory(dir=RUN) as temp:
            for i in range(4):
                txt = estimate(public, rng)
                tmp = Path(temp) / f"synthetic-{i}.txt"
                tmp.write_text(txt)
                sets = Teambuilder.parse_showdown_team(txt)
                ids = [to_id_str(m.species or m.nickname) for m in sets]
                profiles.append(
                    {
                        name: dict(
                            nature=m.nature, evs=dict(zip(STATS, m.evs or [0] * 6, strict=True))
                        )
                        for name, m in zip(ids, sets, strict=True)
                    }
                )
                for _ in range(12):
                    sim = int(rng.integers(1_000_000, 180_000_000))
                    model.reset(own, str(tmp), sim, "p1")
                    draws.append([ids[int(k) - 1] for k in model.opponent.lineup])
    finally:
        model.battle.close()
    result = dict(key=key, profiles=profiles, lineups=draws)
    assert len(draws) == 48 and all(len(set(q)) == 4 for q in draws)
    with tempfile.NamedTemporaryFile("w", dir=file.parent, delete=False) as temp:
        json.dump(result, temp)
    Path(temp.name).replace(file)
    return result


@functools.lru_cache(maxsize=1)
def natures():
    return GenData.from_gen(9).natures


def estimated_stats(m, spread):
    modifiers = natures().get(spread["nature"].lower()) or natures().get(spread["nature"])
    assert modifiers is not None
    return {
        k: int(
            (m.base_stats[k] + spread["evs"][k] + (75 if k == "hp" else 20)) * modifiers.get(k, 1)
            + 1e-6
        )
        for k in STATS
    }


def possible_bench(pos, enemies, profiles, prior):
    import itertools

    unknown = [m["name"] for m in pos["unknown"]]
    choices = list(itertools.combinations(unknown, pos["vacancies"]))
    assert choices
    canonical = {m.name: index_name(m, profiles) for m in enemies}
    revealed = {canonical[m.name] for m in enemies if m.revealed}
    live = {canonical[m["name"]] for m in pos["foe"] if m}
    options = []
    for choice in choices:
        four = revealed | {canonical[n] for n in choice} | live
        # Condition on every known bring. Opening leads also inform probabilities.
        mass = 1.0
        for line in prior["lineups"]:
            if four.issubset(line):
                mass += 1 + 2 * (pos["turn"] == 1 and live == set(line[:2]))
        options.append(mass)
    return choices, np.array(options) / sum(options)


@torch.no_grad()
def public_beliefs(view, pos, prior, count):
    actor = players()[0]
    def canonical(m):
        return to_id_str(m.name)
    enemies = sorted(view.opponent_team.values(), key=canonical)
    friends = sorted(view.team.values(), key=canonical)
    assert len(enemies) == len(friends) == 6
    seed_text = json.dumps(pos, sort_keys=True, separators=(",", ":"))
    rng = np.random.default_rng(public_seed(prior["key"] + seed_text))
    choices, probs = possible_bench(pos, enemies, prior["profiles"][0], prior)
    glob = actor.embed_global(view)
    mine = actor.embed_side(view, 0).copy()
    mine[-5:-1] = [
        float(any(m and m.species != m.base_species and "mega" in m.species for m in friends)),
        0,
        0,
        0,
    ]
    theirs = actor.embed_side(view, 2000, opp=True).copy()
    theirs[-5:-1] = [
        float(
            bool(view.opponent_active_pokemon[0])
            and not view.opponent_used_mega_evolve
            and (view.opponent_active_pokemon[0].item or "").endswith(("ite", "itex", "itey"))
        ),
        0,
        0,
        0,
    ]

    def active_names(mons):
        return [m.name if m else None for m in mons]

    us = active_names(view.active_pokemon)
    them = active_names(view.opponent_active_pokemon)

    def embed(m, i, opp, side, active):
        p = actor.embed_pokemon(
            m, i, from_opponent=opp, active_a=m.name == active[0], active_b=m.name == active[1]
        )
        return np.concatenate((glob, side, p))

    ally = []
    for i, original in enumerate(friends):
        m = copy.copy(original)
        m._selected_in_teampreview = False
        ally.append(embed(m, i, True, mine, us))
    obs, records = [], []
    team_names = [m.name for m in enemies]
    moves = [list(m.moves)[:4] if m else [] for m in view.opponent_active_pokemon]
    for _ in range(count):
        profile = prior["profiles"][int(rng.integers(len(prior["profiles"])))]
        bench = choices[int(rng.choice(len(choices), p=probs))]
        imagined = []
        spreads = {}
        for i, original in enumerate(enemies):
            m = copy.copy(original)
            spread = profile[index_name(m, profile)]
            m.stats = estimated_stats(m, spread)
            m._selected_in_teampreview = m.revealed or m.name in bench or m.name in them
            imagined.append(embed(m, i, False, theirs, them))
            spreads[m.name] = spread
        obs.append(np.concatenate(imagined + ally))
        records.append(dict(bench=list(bench), spreads=spreads, roster=team_names, moves=moves))
    inputs = {
        "observation": torch.from_numpy(np.stack(obs)),
        "action_mask": torch.ones((count, 214), dtype=torch.int8),
    }
    logits = actor.policy.get_logits(inputs, actor_grad=False)[0].reshape(count, 2, 107).numpy()
    assert np.isfinite(logits).all()
    for record, values in zip(records, logits, strict=True):
        record["logits"] = values.tolist()
    return records
