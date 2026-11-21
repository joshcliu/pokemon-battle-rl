"""Terminal-return evolution strategies for a mechanics-aware residual on frozen PPO."""

import argparse
import functools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, RUN, SEEDS, distribution, verify
from mb_residual_preview import pilot
from poke_env.battle import Field, PokemonType, SideCondition, Weather
from vgc_posttrain import Match, atomic_rows, batch_obs

from pokemon_rl.league import stat

FEATURES = [
    "hp_removed",
    "knockouts",
    "first_strike_damage",
    "overkill",
    "ally_damage",
    "fake_out",
    "protect_threat",
    "repeat_protect",
    "switch_advantage",
    "voluntary_switch",
    "speed_control",
    "setup",
    "status",
    "recovery",
    "mega",
    "friendly_target",
]
START = np.array(
    [1.5, 1.2, 0.7, -0.4, -1.2, 0.8, 0.25, -0.6, 0.6, -0.25, 0.5, 0.3, 0.3, 0.3, 0.15, -1.3]
)
UPDATES = 24
DIRECTIONS = 8
GAMES = 4
SIGMA = 0.65


def mechanics(view, actions):
    own, foe = view.active_pokemon, view.opponent_active_pokemon
    rain, sun = Weather.RAINDANCE in view.weather, Weather.SUNNYDAY in view.weather
    psychic = Field.PSYCHIC_TERRAIN in view.fields
    grassy = Field.GRASSY_TERRAIN in view.fields
    trick = Field.TRICK_ROOM in view.fields
    tail = [
        SideCondition.TAILWIND in x for x in (view.side_conditions, view.opponent_side_conditions)
    ]
    hp = np.array([m.current_hp_fraction if m else 0.0 for m in foe])

    @functools.lru_cache(None)
    def speed(m, side):
        v = stat(m, "spe") * (2 if tail[side] else 1)
        if (sun and m.ability == "chlorophyll") or (rain and m.ability == "swiftswim"):
            v *= 2
        if m.status and m.status.name == "PAR":
            v /= 2
        return v

    @functools.lru_cache(None)
    def power(m, move, target):
        kind, bp = move.type, move.base_power
        if move.id == "weatherball" and (sun or rain):
            kind, bp = (PokemonType.FIRE if sun else PokemonType.WATER), 100
        if move.id in ("eruption", "waterspout"):
            bp = 150 * m.current_hp_fraction
        if move.id == "lastrespects":
            bp = 50  # unknown faint count is left to the PPO representation
        if not bp or not kind:
            return 0.0
        if move.id == "fakeout" and not m.first_turn:
            return 0.0
        if (kind == PokemonType.GROUND and target.ability == "levitate") or (
            move.priority > 0
            and psychic
            and PokemonType.FLYING not in target.types
            and target.ability != "levitate"
        ):
            return 0.0
        if target.ability in {"flashfire", "wellbakedbody"} and kind == PokemonType.FIRE:
            return 0.0
        if target.ability in {"waterabsorb", "stormdrain", "dryskin"} and kind == PokemonType.WATER:
            return 0.0
        if (
            target.ability in {"lightningrod", "motordrive", "voltabsorb"}
            and kind == PokemonType.ELECTRIC
        ):
            return 0.0
        physical = move.category.name == "PHYSICAL"
        atk = stat(m, "atk" if physical else "spa")
        defense = stat(target, "def" if physical or move.id == "psyshock" else "spd")
        mult = target.damage_multiplier(kind) * (1.5 if kind in m.types else 1.0)
        if kind in (PokemonType.FIRE, PokemonType.WATER) and (rain or sun):
            mult *= 1.5 if (kind == PokemonType.FIRE) == sun else 0.5
        if (grassy and kind == PokemonType.GRASS) or (psychic and kind == PokemonType.PSYCHIC):
            mult *= 1.3
        if m.status and m.status.name == "BRN" and physical and m.ability != "guts":
            mult *= 0.5
        return min(
            2.0,
            (0.44 * bp * atk / max(1.0, defense) + 2)
            * 0.925
            * mult
            * float(move.accuracy)
            * move.expected_hits
            / max(1.0, stat(target, "hp")),
        )

    @functools.lru_cache(None)
    def threat(mon):
        return sum(
            max((power(f, m, mon) for m in f.moves.values()), default=0.0)
            for f in foe
            if f and not f.fainted
        )

    indices = actions.numpy()
    pieces = []
    for slot in range(2):
        by_id = np.zeros((107, len(FEATURES) + 2))
        m = own[slot]
        for order in np.unique(indices[:, slot]):
            r = by_id[order]
            if order == 0:
                continue
            if order <= 6:
                bench = list(view.team.values())[order - 1]
                offensive = max(
                    (
                        power(bench, mv, f)
                        for mv in bench.moves.values()
                        for f in foe
                        if f and not f.fainted
                    ),
                    default=0.0,
                )
                r[8] = np.clip((threat(m) if m else 0.0) - threat(bench) + 0.4 * offensive, -2, 2)
                r[9] = float(not view.force_switch[slot])
                continue
            from poke_env.environment import DoublesEnv

            move = DoublesEnv._action_to_order_individual(order, view, True, slot).order
            if move.id == "recharge":
                continue
            move_target = getattr(move.target, "name", "")
            target = (order - 7) % 5 - 2
            r[14] = float(27 <= order <= 46)
            spread = move_target in ("ALL_ADJACENT", "ALL_ADJACENT_FOES") or (
                psychic and move.id == "expandingforce"
            )
            if target < 0 and move.base_power:
                ally = own[-target - 1]
                r[15] = float(ally is not None)
                if ally:
                    r[4] = power(m, move, ally)
            if move_target == "ALL_ADJACENT" and own[1 - slot]:
                r[4] = 0.75 * power(m, move, own[1 - slot])
            for j, f in enumerate(foe):
                if not f or f.fainted or target < 0 or (not spread and target not in (0, j + 1)):
                    continue
                hit = power(m, move, f) * (0.75 if spread and all(foe) else 1.0)
                r[len(FEATURES) + j] = hit
                faster = move.priority > 0 or (
                    (speed(m, 0) > speed(f, 1)) != trick and move.priority == 0
                )
                r[2] += min(hp[j], hit) * faster
                if move.id == "fakeout" and hit and f.ability not in ("innerfocus", "shielddust"):
                    r[5] = 1.0
            if move.id in (
                "protect",
                "detect",
                "spikyshield",
                "kingsshield",
                "banefulbunker",
                "silktrap",
            ):
                r[6] = min(1.5, threat(m))
                r[7] = min(1.0, m.protect_counter)
            if move.id in ("tailwind", "trickroom", "icywind", "electroweb"):
                slow = np.mean([speed(m, 0) < speed(f, 1) for f in foe if f] or [0.0])
                if move.id == "trickroom":
                    r[10] = (2 * slow - 1) * (-1 if trick else 1)
                elif move.id == "tailwind":
                    r[10] = -1 if tail[0] else (0.3 + slow) * (-1 if trick else 1)
                else:
                    r[10] = 0.5 * slow * (-1 if trick else 1)
            boosts = move.boosts or {}
            if move_target == "SELF" and any(v > 0 for v in boosts.values()):
                r[11] = m.current_hp_fraction / (1 + max(0, *m.boosts.values()))
            if (
                move.id in ("spore", "sleeppowder", "willowisp", "thunderwave", "yawn")
                and target > 0
            ):
                f = foe[target - 1]
                if f and not f.status:
                    r[12] = float(move.accuracy)
            r[13] = move.heal * (1 - m.current_hp_fraction)
        pieces.append(by_id[indices[:, slot]])
    combined = pieces[0] + pieces[1]
    hit = combined[:, len(FEATURES) :]
    combined[:, 0] = np.minimum(hp, hit).sum(1)
    combined[:, 1] = ((hp > 0) & (hit >= hp)).sum(1)
    combined[:, 3] = np.maximum(0, hit - hp).sum(1)
    return combined[:, : len(FEATURES)]


def folder(s):
    return RUN / "es" / f"seed-{s}"


@functools.lru_cache(maxsize=4)
def fitted(seed):
    ck = json.loads((folder(seed) / "weights.json").read_text())
    assert ck["update"] == UPDATES and ck["features"] == FEATURES
    return np.array(ck["weights"])


def action(model, obs, view, weights):
    pairs, logp = distribution(model, obs)
    if len(pairs) == 1:
        return pairs[0].tolist()
    feature = mechanics(view, pairs)
    assert np.isfinite(feature).all()
    idx = (logp.numpy() + feature @ weights).argmax()
    return pairs[idx].tolist()


@torch.no_grad()
def sample(task):
    seed, u, d, weights, noise = task
    model, match = pilot(seed), Match()
    rewards = [[], []]
    for k in range(GAMES):
        idx = (u * DIRECTIONS + d) * GAMES + k
        from mb_rollout_branches import pairing

        own, foe, side = pairing(seed, idx)
        sim = 760_000_000 + seed * 1_000_000 + idx
        for sign, n in ((1, 0), (-1, 1)):
            match.reset(own, foe, sim, side, preview="native")
            torch.manual_seed(sim + 300_000_000)
            done = False
            while not done:
                choices = action(
                    model, batch_obs(match.obs()), match.view, weights + sign * SIGMA * noise
                )
                done, r = match.step(choices)
            rewards[n].append(r)
    return np.array(rewards)


def train(jobs):
    state = {}
    for seed in SEEDS:
        p = folder(seed) / "weights.json"
        if p.exists():
            state[seed] = json.loads(p.read_text())
        else:
            state[seed] = dict(
                update=0, seed=seed, weights=START.tolist(), features=FEATURES, logs=[]
            )
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for u in range(UPDATES):
            tasks = []
            for s in SEEDS:
                if state[s]["update"] > u:
                    continue
                assert state[s]["update"] == u
                rng = np.random.default_rng([220675, s, u])
                for d in range(DIRECTIONS):
                    tasks.append(
                        (s, u, d, np.array(state[s]["weights"]), rng.normal(size=len(START)))
                    )
            returns = list(pool.map(sample, tasks))
            for s in SEEDS:
                ranked = [
                    (t[-1], r.mean(axis=1))
                    for t, r in zip(tasks, returns, strict=True)
                    if t[0] == s
                ]
                if not ranked:
                    continue
                noise, scores = map(np.array, zip(*ranked, strict=True))
                elite = scores.max(1).argsort()[-4:]
                norm = max(0.35, scores[elite].std())
                grad = ((scores[elite, 0] - scores[elite, 1])[:, None] * noise[elite]).mean(
                    0
                ) / norm
                theta = np.array(state[s]["weights"]) + 0.18 * grad / (
                    max(1.0, np.linalg.norm(grad) / 3)
                )
                state[s]["weights"] = theta.clip(-5.0, 5.0).tolist()
                state[s]["update"] = u + 1
                state[s]["logs"].append(
                    dict(
                        update=u + 1,
                        games=(u + 1) * DIRECTIONS * GAMES * 2,
                        paired_return_sd=float(np.std(scores[:, 0] - scores[:, 1])),
                        mean_return=float(scores.mean()),
                    )
                )
                dest = folder(s)
                dest.mkdir(parents=True, exist_ok=True)
                tmp = dest / "weights.tmp"
                tmp.write_text(json.dumps(state[s]) + "\n")
                tmp.replace(dest / "weights.json")
                atomic_rows(OUT / "training" / f"es-{s}.csv", state[s]["logs"])
            if tasks:
                print("ES update", u + 1, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "check"))
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    verify()
    if args.stage == "train":
        train(args.jobs)
    else:
        print(sample((2, 0, 0, START, np.ones_like(START))), flush=True)
