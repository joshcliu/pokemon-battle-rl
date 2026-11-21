"""Per-encounter policy search using open sheets, training-only spread priors and independent model games."""

import argparse
import functools
import hashlib
import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_mechanics_es import FEATURES, action, fitted
from mb_offline_lineup_value import LINES, matchup
from mb_offline_lineup_value import weights as lineup_weights
from mb_policy_iteration_eval import GAMES, OUT, RUN, SEEDS, SPEC, schedule, verify
from mb_public_preview_search import inferred_sheet, key_for, public_fixture, public_sets, read_plan
from mb_residual_preview import pilot
from poke_env.battle import Move, Pokemon
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text

STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
AXES = np.array(
    [
        [1.5, 1.0, 0.8, -0.2, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1.0, -0.4, 1.0, 0.8, 0, 0, 0, 0.6, 0, 0],
        [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1.0, 1.2, 0.8, 0, 0, 0],
        [0, 0.7, 0.6, -1.3, -0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, -0.4],
    ]
)
assert AXES.shape == (4, len(FEATURES))


@functools.lru_cache(maxsize=1024)
def role(species, moves):
    mon = Pokemon(gen=9, species=species)
    attacks = [Move(m, gen=9) for m in moves]
    p = sum(m.base_power for m in attacks if m.category.name == "PHYSICAL")
    s = sum(m.base_power for m in attacks if m.category.name == "SPECIAL")
    return np.array(
        [
            mon.base_stats["spe"] / 100,
            p / (1 + p + s),
            s / (1 + p + s),
            sum(m.base_power == 0 for m in attacks) / 4,
            mon.base_stats["atk"] / 150,
            mon.base_stats["spa"] / 150,
        ]
    )


@functools.lru_cache(maxsize=1)
def donors():
    bank = []
    for path in dict.fromkeys(SPEC["own_train"] + SPEC["foe_train"]):
        for m in Teambuilder.parse_showdown_team(team_text(path).strip()):
            species = to_id_str(m.species or m.nickname)
            moves = tuple(sorted(to_id_str(x) for x in m.moves))
            evs = tuple(m.evs or [0] * 6)
            assert max(evs) <= 32 and sum(evs) <= 66 and m.nature
            bank.append((species, moves, role(species, moves), evs, m.nature))
    return bank


def estimate(public, rng):
    if rng.random() < 0.2:
        return inferred_sheet(public, int(rng.integers(2**31)))
    bank = donors()
    lines = []
    for species, ability, item, moves in public:
        f = role(species, tuple(moves))
        scores = np.array(
            [
                3 * (species == s)
                + 0.7 * len(set(moves) & set(ms))
                - 1.4 * np.linalg.norm(f - feat)
                for s, ms, feat, _, _ in bank
            ]
        )
        ids = scores.argsort()[-12:]
        p = np.exp(scores[ids] - scores[ids].max())
        _, _, _, evs, nature = bank[int(rng.choice(ids, p=p / p.sum()))]
        text = " / ".join(f"{v} {key}" for key, v in zip(STATS, evs, strict=True) if v)
        lines.append(
            "\n".join(
                [
                    species + (" @ " + item if item != "unknown_item" else ""),
                    "Ability: " + ability,
                    "Level: 50",
                    *(["EVs: " + text] if text else []),
                    nature + " Nature",
                    *["- " + m for m in moves],
                ]
            )
        )
    return "\n\n".join(lines) + "\n"


@torch.no_grad()
def optimize(own, public, seed):
    old = read_plan(own, public, seed)
    key = hashlib.sha256((key_for(own, public, seed) + "-specialize-v1").encode()).hexdigest()
    dest = RUN / "public-adaptation" / f"{key}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    rng = np.random.default_rng(int(key[:16], 16))
    base = fitted(seed)
    model = pilot(seed)
    match = Match()
    n = 0
    worlds = 0
    with tempfile.TemporaryDirectory(dir=RUN) as directory:

        def world():
            nonlocal worlds
            path = Path(directory) / f"world-{worlds}.txt"
            path.write_text(estimate(public, rng))
            worlds += 1
            return str(path), int(rng.integers(1_000_000, 180_000_000)), ("p1", "p2")[worlds % 2]

        def score(line, coeff, scenario):
            nonlocal n
            path, sim, side = scenario
            match.reset(own, path, sim, side, lineup=line)
            torch.manual_seed(sim + 700_000_000)
            done = False
            while not done:
                a = action(model, batch_obs(match.obs()), match.view, coeff)
                done, r = match.step(a)
            n += 1
            return (1 + r) / 2

        # Six preview candidates. Preserve the old 64-rollout policy as an anchor.
        candidates = [old["selected"]]
        scenario = world()

        def offline(view):
            idx = int((matchup(view) @ lineup_weights(seed)).argmax())
            return "".join(str(i + 1) for i in LINES[idx])

        match.reset(own, scenario[0], scenario[1], scenario[2], preview_picker=offline)
        candidates.append(match.lineup)
        for _ in range(16):
            path, sim, side = world()
            match.reset(own, path, sim, side, preview="native")
            candidates.append(match.lineup)
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) >= 5:
                break
        while len(candidates) < 6:
            candidates = list(
                dict.fromkeys(
                    [*candidates, "".join(map(str, rng.choice(np.arange(1, 7), 4, replace=False)))]
                )
            )
        active = candidates[:6]
        totals = {x: 0 for x in active}
        counts = {x: 0 for x in active}
        for repeats, keep in ((4, 3), (4, 2), (6, 1)):
            for _ in range(repeats):
                scenario = world()
                for line in active:
                    totals[line] += score(line, base, scenario)
                    counts[line] += 1
            active = sorted(active, key=lambda x: totals[x] / counts[x], reverse=True)[:keep]
        line = active[0]
        # Two mirrored policy-gradient estimates use the same model worlds for each sign.
        coord = np.zeros(4)
        for _ in range(2):
            noise = rng.normal(size=(4, 4))
            diff = np.zeros(4)
            for j, d in enumerate(noise):
                for _ in range(3):
                    scenario = world()
                    diff[j] += score(line, base + (coord + 0.85 * d) @ AXES, scenario) - score(
                        line, base + (coord - 0.85 * d) @ AXES, scenario
                    )
            grad = (noise * (diff / 3)[:, None]).mean(0)
            coord = np.clip(coord + 0.8 * grad, -1.5, 1.5)
        adapted = base + coord @ AXES
        finalists = [(old["selected"], base), (line, base), (line, adapted)]
        validation = np.zeros(3)
        # New hidden profiles and random seeds judge the three candidates. None of
        # these internal battles shares a seed or opponent selection with a scored game.
        for _ in range(12):
            scenario = world()
            for j, (candidate, w) in enumerate(finalists):
                validation[j] += score(candidate, w, scenario)
        idx = int(validation.argmax())
        if validation[idx] < validation[0] + 1.5:
            idx = 0
        selected, weights = finalists[idx]
        assert n == 132
    result = dict(
        key=key,
        seed=seed,
        selected=selected,
        weights=weights.tolist(),
        finalist=idx,
        modeled_games=n,
        baseline_planning_games=64,
        validation=validation.tolist(),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, suffix=".tmp", delete=False) as tmp:
        tmp.write(json.dumps(result) + "\n")
    Path(tmp.name).replace(dest)
    return result


@torch.no_grad()
def play(task):
    seed, cell = task
    i, j, d, own, foe, start = cell
    public = public_fixture(own, foe)
    plan = optimize(own, public, seed)
    match, model = Match(), pilot(seed)
    weights = np.array(plan["weights"])
    rows = []
    for k in range(GAMES):
        sim, side = start + k, ("p1", "p2")[k % 2]

        def pick(view):
            assert public_sets(view) == public
            return plan["selected"]

        match.reset(own, foe, sim, side, preview_picker=pick)
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            a = action(model, batch_obs(match.obs()), match.view, weights)
            done, r = match.step(a)
        rows.append(
            dict(
                method="adaptive-es",
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
                finalist=plan["finalist"],
                public_key=plan["key"],
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
        own, foe = SPEC["own_train"][0], SPEC["foe_train"][0]
        public = public_fixture(own, foe)
        print("training-spread donors", len(donors()), flush=True)
        result = optimize(own, public, 2)
        assert result == optimize(own, public, 2)
        print(
            "public encounter planning verified",
            result["modeled_games"],
            result["validation"],
            flush=True,
        )
    else:
        cells = list(schedule(args.stage))
        with ProcessPoolExecutor(
            args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for s in SEEDS:
                file = OUT / "scores" / f"{args.stage}-adaptive-es-{s}.csv"
                if file.exists():
                    continue
                rows = [r for batch in pool.map(play, [(s, c) for c in cells]) for r in batch]
                atomic_rows(file, rows)
                print(args.stage, s, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
