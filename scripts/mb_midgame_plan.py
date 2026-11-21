"""Public-belief one-turn planning with a learned Monte Carlo value of the next position."""

import argparse
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_endgame_plan import forecaster
from mb_endgame_value import public_position, regress, value_features, visible_mon
from mb_mechanics_es import action, fitted, mechanics
from mb_opening_planner import text_choice
from mb_policy_iteration_eval import GAMES, OUT, RUN, SEEDS, SPEC, distribution, schedule, verify
from mb_residual_preview import pilot
from mb_rollout_branches import pairing
from poke_env.battle import Move, Pokemon
from vgc_posttrain import Match, atomic_rows, batch_obs


def projection(view):
    pos = public_position(view)
    pos["bench"] = [[], []]
    active = [view.active_pokemon, view.opponent_active_pokemon]
    for side, team in enumerate((view.team, view.opponent_team)):
        names = {m.name for m in active[side] if m and not m.fainted}
        for m in team.values():
            selected = m.selected_in_teampreview if side == 0 else m.revealed
            if selected and not m.fainted and m.name not in names:
                item = visible_mon(m, side == 0)
                item["name"] = m.name
                pos["bench"][side].append(item)
        for item, mon in zip((pos["own"], pos["foe"])[side], active[side], strict=True):
            if item:
                item["name"] = mon.name

    def unobserved(m):
        # Only open-sheet attributes are used for a potential unrevealed reserve.
        return dict(
            species=m.species,
            base=m.base_species,
            name=m.name,
            ability=m.ability or "",
            item="" if m.item in (None, "unknown_item") else m.item,
            moves=list(m.moves),
            pp=[mv.max_pp for mv in m.moves.values()],
            hp=1.0,
            status="",
            boosts={},
            baseSpeed=m.base_stats["spe"],
            first=True,
            protect=0,
            last="",
        )

    pos["unknown"] = [
        unobserved(m)
        for m in view.opponent_team.values()
        if not m.revealed and m.name not in {q.name for q in active[1] if q}
    ]
    pos["vacancies"] = (
        4
        - sum(m.fainted for m in view.opponent_team.values())
        - sum(bool(m) for m in pos["foe"])
        - len(pos["bench"][1])
    )
    return pos


@functools.lru_cache(maxsize=512)
def species(name):
    return Pokemon(gen=9, species=name)


@functools.lru_cache(maxsize=1024)
def move(name):
    return Move(name, gen=9)


@functools.lru_cache(maxsize=65536)
def matchup_hit(attacker, attack, defender, ability):
    a, f = species(attacker), species(defender)
    m = move(attack)
    if not m.type or not m.base_power:
        return 0.0
    kind = m.type.name
    if (
        (kind == "GROUND" and ability == "levitate")
        or (kind == "FIRE" and ability in ("flashfire", "wellbakedbody"))
        or (kind == "WATER" and ability in ("waterabsorb", "stormdrain", "dryskin"))
    ):
        return 0.0
    physical = m.category.name == "PHYSICAL"
    offense = a.base_stats["atk" if physical else "spa"] + 30
    defense = f.base_stats["def" if physical else "spd"] + 30
    power = 0.44 * m.base_power * offense / defense + 2
    return min(
        2.0,
        power
        * m.expected_hits
        * float(m.accuracy)
        * f.damage_multiplier(m.type)
        * (1.5 if m.type in a.types else 1.0)
        / (f.base_stats["hp"] + 80),
    )


def features(pos):
    vectors = []
    active = [[m for m in side if m] for side in (pos["own"], pos["foe"])]
    for j, mons in enumerate(active):
        bench = pos["bench"][j]
        vacancies = pos["vacancies"] if j else 0
        enemies = active[1 - j]
        hits = []
        kos = []
        fast = []
        for f in enemies:
            dmg = [
                max(
                    (
                        matchup_hit(m["species"], mv, f["species"], f["ability"])
                        for mv in m["moves"]
                    ),
                    default=0.0,
                )
                for m in mons
            ]
            hits.append(max(dmg, default=0.0))
            kos.append(sum(dmg) >= f["hp"] and bool(dmg))
            fast.append(
                max(
                    [v for v, m in zip(dmg, mons, strict=True) if m["baseSpeed"] > f["baseSpeed"]]
                    + [0.0]
                )
            )
        active_hp = sum(m["hp"] for m in mons)
        bench_hp = sum(m["hp"] for m in bench) + vacancies
        material = len(mons) + len(bench) + vacancies
        vectors.append(
            np.array(
                [
                    len(bench) + vacancies,
                    bench_hp,
                    material,
                    active_hp + bench_hp,
                    sum(hits),
                    sum(kos),
                    sum(fast),
                    max(hits, default=0.0),
                    sum(bool(m["status"]) for m in bench),
                ]
            )
        )
    return np.concatenate([value_features(pos), vectors[0] - vectors[1]])


def valid(pos):
    return 0 <= pos["vacancies"] <= len(pos["unknown"]) and all(
        len(pos["bench"][j]) + (pos["vacancies"] if j else 0) <= 2 for j in (0, 1)
    )


@torch.no_grad()
def collect(task):
    seed, block = task
    dest = RUN / "midgame" / f"seed-{seed}" / f"{block:03}.npz"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    model, w, match = pilot(seed), fitted(seed), Match()
    groups = []
    states = []
    labels = []
    for g in range(block * 14, block * 14 + 14):
        own, foe, side = pairing(seed, g)
        if g % 2:
            own, foe = foe, own
        sim = 912_000_000 + seed * 1_000_000 + g
        match.reset(own, foe, sim, side, preview="native")
        done = False
        frames = []
        while not done:
            pos = projection(match.view)
            if valid(pos):
                frames.append(features(pos))
            a = action(model, batch_obs(match.obs()), match.view, w)
            done, r = match.step(a)
        states.extend(frames)
        labels.extend([r] * len(frames))
        groups.extend([g] * len(frames))
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, x=states, y=labels, g=groups)
    tmp.replace(dest)


def fit(seed):
    folder = RUN / "midgame" / f"seed-{seed}"
    cols = {k: [] for k in ("x", "y", "g")}
    for b in range(40):
        with np.load(folder / f"{b:03}.npz") as part:
            for k in cols:
                cols[k].append(part[k])
    x, r, g = (np.concatenate(p) for p in cols.values())
    assert len(set(g)) == 560 and x.shape[1] == 23
    y = (r + 1) / 2
    w = 1 / np.bincount(g)[g]
    test = g % 5 == 0
    beta = regress(x[~test], y[~test], w[~test])
    pred = 1 / (1 + np.exp(-np.clip(x[test] @ beta, -25, 25)))
    prior = np.average(y[~test], weights=w[~test])
    atomic_rows(
        OUT / "training" / f"midgame-value-{seed}.csv",
        [
            dict(
                seed=seed,
                games=560,
                positions=len(g),
                cv_brier=np.average((pred - y[test]) ** 2, weights=w[test]),
                constant_brier=np.average((prior - y[test]) ** 2, weights=w[test]),
            )
        ],
    )
    np.save(folder / "value.npy", regress(x, y, w))
    print("fit midgame", seed, len(x), flush=True)


@functools.lru_cache(maxsize=4)
def value(seed):
    path = RUN / "midgame" / f"seed-{seed}" / "value.npy"
    if not path.exists():
        path = OUT / "weights" / f"midgame-value-seed-{seed}.npy"
    return np.load(path)


def backed_up_values(worlds, beta, depth):
    def pwin(leaf):
        if isinstance(leaf, (int, float)):
            return leaf
        return 1 / (1 + np.exp(-np.clip(features(leaf) @ beta, -25, 25)))

    return np.mean(
        [
            [[pwin(leaf) for leaf in (trace if depth > 1 else [trace])] for trace in world]
            for world in worlds
        ],
        axis=0,
    )


def estimates(model, obs, view, seed, check, depth, *, wide=False):
    pairs, logp = distribution(model, obs)
    phi = mechanics(view, pairs)
    prior = logp.numpy() + phi @ fitted(seed)
    ranks = prior.argsort()[::-1].tolist()
    base = pairs[ranks[0]].tolist()
    fallback = ([base], np.zeros((1, depth)), np.zeros(1))
    if len(pairs) < 2 or any(view.force_switch):
        return *fallback, 0
    pos = projection(view)
    if not valid(pos):
        return *fallback, 0
    ids = list(dict.fromkeys([ranks[0], int(logp.argmax()), int(phi[:, 0].argmax()), *ranks]))[:4]
    if wide:
        # Expand the current shortlist with high-prior commands that cover different
        # active Pokémon actions. The comparison uses the exact same four initial worlds.
        candidates = pairs.numpy()
        while len(ids) < min(8, len(ranks)):
            distances = (candidates[:, None, :] != candidates[ids][None, :, :]).sum(2).min(1)
            diversified = prior + 0.8 * distances
            diversified[ids] = -np.inf
            ids.append(int(diversified.argmax()))
    orders = [text_choice(view, pairs[i].tolist()).removeprefix("/choose ") for i in ids]
    worlds = forecaster()(pos, orders, depth=depth)
    if len(worlds) < 2:
        return *fallback, len(worlds)
    beta = np.zeros(23) if check else value(seed)
    if check:
        beta[1:3] = 1
        beta[16:18] = 1

    expected = backed_up_values(worlds, beta, depth)
    return pairs[ids].tolist(), expected, 0.025 * (prior[ids] - prior[ids[0]]), len(worlds)


def choose(model, obs, view, seed, check, *, weights=(1.0,)):
    actions, expected, prior, n = estimates(model, obs, view, seed, check, len(weights))
    idx = int(np.argmax(expected @ weights + prior))
    return actions[idx], n, bool(idx)


def successive_halving(model, obs, view, seed, *, audit=False):
    actions, expected, prior, n = estimates(model, obs, view, seed, False, 1, wide=True)
    if n < 2:
        return choose(model, obs, view, seed, False)[0], 0
    scores = expected[:, 0] + prior
    # Retain the winner of the original four-command search as a control and
    # evaluate three challengers in independent hypothetical hidden worlds.
    anchor = int(scores[:4].argmax())
    finalists = list(dict.fromkeys([anchor, *scores.argsort()[::-1].tolist()]))[:4]
    orders = [text_choice(view, actions[i]).removeprefix("/choose ") for i in finalists]
    forecasts = forecaster()(projection(view), orders, samples=8, salt="halving-final-")
    if len(forecasts) < 2:
        return actions[anchor], n * len(actions)
    final = backed_up_values(forecasts, value(seed), 1)[:, 0]
    means = (n * expected[finalists, 0] + len(forecasts) * final) / (n + len(forecasts))
    idx = finalists[int((means + prior[finalists]).argmax())]
    count = n * len(actions) + len(forecasts) * len(finalists)
    if audit:
        assert 0 < count <= 64 and len(set(orders)) == len(orders)
    return actions[idx], count


@torch.no_grad()
def score(task):
    seed, cell, check = task
    i, j, d, own, foe, start = cell
    model, match = pilot(seed), Match()
    rows = []
    for k in range(GAMES if not check else 2):
        sim, side = start + k, ("p1", "p2")[k % 2]
        match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done = False
        changes = worlds = 0
        while not done:
            a, n, changed = choose(model, batch_obs(match.obs()), match.view, seed, check)
            done, r = match.step(a)
            changes += changed
            worlds += n
        rows.append(
            dict(
                method="belief-es",
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
                matched_worlds=worlds,
                changed_commands=changes,
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "train", "validation", "custom-dev"))
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        rows = score((2, (0, 0, 0, SPEC["own_train"][0], SPEC["foe_train"][0], 666000000), True))
        assert sum(r["matched_worlds"] for r in rows) > 8
        print(
            "public midgame search",
            [(r["matched_worlds"], r["changed_commands"]) for r in rows],
            flush=True,
        )
        collect((2, 0))
    elif args.stage == "train":
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, _ in enumerate(pool.map(collect, [(s, b) for b in range(40) for s in SEEDS])):
                if (i + 1) % 40 == 0:
                    print("midgame blocks", i + 1, flush=True)
        for s in SEEDS:
            fit(s)
    else:
        cells = list(schedule(args.stage))
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for s in SEEDS:
                dest = OUT / "scores" / f"{args.stage}-belief-es-{s}.csv"
                if dest.exists():
                    continue
                rows = [
                    r for group in pool.map(score, [(s, c, False) for c in cells]) for r in group
                ]
                atomic_rows(dest, rows)
                print(args.stage, s, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
