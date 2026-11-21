"""Monte Carlo state-value learning for public positions where neither side has reserves."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_mechanics_es import action, fitted
from mb_policy_iteration_eval import OUT, RUN, SEEDS, verify
from mb_residual_preview import pilot
from mb_rollout_branches import pairing
from vgc_posttrain import Match, atomic_rows, batch_obs


def no_reserves(view):
    if any(view.force_switch):
        return False
    for team, active in (
        (view.team, view.active_pokemon),
        (view.opponent_team, view.opponent_active_pokemon),
    ):
        living = sum(m is not None and not m.fainted for m in active)
        if living == 0 or sum(m.fainted for m in team.values()) + living != 4:
            return False
    return True


def visible_mon(m, own):
    if m is None or m.fainted:
        return None
    item = m.item
    result = dict(
        species=m.species,
        base=m.base_species,
        ability=m.ability or "",
        item="" if item in (None, "unknown_item") else item,
        moves=list(m.moves),
        pp=[mv.current_pp for mv in m.moves.values()],
        hp=m.current_hp_fraction,
        status=m.status.name.lower() if m.status else "",
        boosts=dict(m.boosts),
        baseSpeed=m.base_stats["spe"],
        first=m.first_turn,
        protect=m.protect_counter,
        last=m.last_move.id if m.last_move else "",
    )
    if own:
        result["stats"] = {k: v for k, v in m.stats.items() if v is not None}
    return result


def public_position(view):
    return dict(
        own=[visible_mon(m, True) for m in view.active_pokemon],
        foe=[visible_mon(m, False) for m in view.opponent_active_pokemon],
        turn=view.turn,
        weather={w.name.lower(): int(s) for w, s in view.weather.items()},
        field={w.name.lower(): int(s) for w, s in view.fields.items()},
        sides=[
            {w.name.lower(): int(s) for w, s in cond.items()}
            for cond in (view.side_conditions, view.opponent_side_conditions)
        ],
    )


def value_features(position):
    boards = []
    speed = []
    for side, mons in enumerate((position["own"], position["foe"])):
        live = [m for m in mons if m and m["hp"] > 0]
        hp = [m["hp"] for m in live]
        boosts = [
            sum(m["boosts"].get(k, 0) for k in ("atk", "spa", "def", "spd", "spe")) / 6
            for m in live
        ]
        incapacitated = sum(m["status"] in ("slp", "frz") for m in live)
        chip = sum(m["status"] in ("brn", "psn", "tox") for m in live)
        conditions = position["sides"][side]
        boards.append(
            np.array(
                [
                    len(live),
                    sum(hp),
                    max(hp, default=0.0),
                    min(hp, default=0.0),
                    sum(boosts),
                    incapacitated,
                    chip,
                    sum(m["status"] == "par" for m in live),
                    sum(m["boosts"].get("evasion", 0) for m in live) / 6,
                    int("reflect" in conditions) + int("lightscreen" in conditions),
                ]
            )
        )
        speeds = []
        for m in live:
            boost = m["boosts"].get("spe", 0)
            v = m["baseSpeed"] * (2 + max(0, boost)) / (2 + max(0, -boost))
            v *= 2 if "tailwind" in conditions else 1
            v *= 0.5 if m["status"] == "par" else 1
            speeds.append(v)
        speed.append(speeds)
    a, b = boards
    first = (
        np.mean([float(x > y) - float(x < y) for x in speed[0] for y in speed[1]])
        if all(speed)
        else 0.0
    )
    if "trick_room" in position["field"] or "trickroom" in position["field"]:
        first = -first
    return np.array([1.0, *(a - b), first, (a[0] - b[0]) * (a[1] + b[1]), first * (a[1] + b[1])])


@torch.no_grad()
def collect(task):
    seed, block = task
    dest = RUN / "endgame" / f"seed-{seed}" / f"games-{block:02}.npz"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    model = pilot(seed)
    w = fitted(seed)
    match = Match()
    features = []
    outcomes = []
    games = []
    steps = []
    for g in range(block * 14, block * 14 + 14):
        own, foe, side = pairing(seed, g)
        if g % 2:
            own, foe = foe, own
        sim = 891_000_000 + seed * 1_000_000 + g
        match.reset(own, foe, sim, side, preview="native")
        done = False
        frames = []
        length = 0
        while not done:
            if no_reserves(match.view):
                frames.append(value_features(public_position(match.view)))
            a = action(model, batch_obs(match.obs()), match.view, w)
            done, r = match.step(a)
            length += 1
        features.extend(frames)
        outcomes.extend([r] * len(frames))
        games.extend([g] * len(frames))
        steps.append(length)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, x=np.array(features).reshape(-1, 14), y=outcomes, groups=games, steps=steps
    )
    tmp.replace(dest)


def regress(x, y, w):
    beta = np.zeros(x.shape[1])
    reg = np.eye(len(beta)) * 4
    reg[0, 0] = 0.01
    for _ in range(15):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -25, 25)))
        beta -= np.linalg.solve(
            x.T @ ((w * p * (1 - p))[:, None] * x) + reg, x.T @ (w * (p - y)) + reg @ beta
        )
    return beta


def fit(seed):
    folder = RUN / "endgame" / f"seed-{seed}"
    data = {k: [] for k in ("x", "y", "groups", "steps")}
    for b in range(40):
        with np.load(folder / f"games-{b:02}.npz") as d:
            for k in data:
                data[k].append(d[k])
    x, r, g, total = (np.concatenate(parts) for parts in data.values())
    y = (r + 1) / 2
    g = g.astype(int)
    w = 1 / np.bincount(g)[g]
    val = g % 5 == 0
    beta = regress(x[~val], y[~val], w[~val])
    p = 1 / (1 + np.exp(-np.clip(x[val] @ beta, -25, 25)))
    prior = np.average(y[~val], weights=w[~val])
    atomic_rows(
        OUT / "training" / f"endgame-{seed}.csv",
        [
            dict(
                seed=seed,
                games=len(total),
                eligible_games=len(set(g)),
                decisions=total.sum(),
                eligible_decisions=len(x),
                cv_brier=np.average((p - y[val]) ** 2, weights=w[val]),
                constant_brier=np.average((prior - y[val]) ** 2, weights=w[val]),
            )
        ],
    )
    np.save(folder / "value.npy", regress(x, y, w))
    print("fit endgame value", seed, len(x), len(set(g)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "train"))
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    verify()
    if args.stage == "check":
        collect((2, 0))
    else:
        tasks = [(s, b) for b in range(40) for s in SEEDS]
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, _ in enumerate(pool.map(collect, tasks)):
                if (i + 1) % 40 == 0:
                    print("endgame blocks", i + 1, flush=True)
        for s in SEEDS:
            fit(s)
