"""Closed-sheet MC461 cross-play against pinned VGC-Bench opponents in local Showdown."""

import argparse
import atexit
import csv
import io
import json
import multiprocessing
import os
import random
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from fit_expanded_preview import POLICIES, pick
from gymnasium import spaces
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer, baselines

from pokemon_rl.battle import DEFAULT_FORMAT, ROOT, LocalBattle
from pokemon_rl.league import FrozenActor

CACHE = ROOT / ".cache/vgcbench-external"
ZIP = CACHE / "vgcbench-bc100-mc461-20261118.zip"
EXTERNAL = ROOT / "teams/vgcbench-mb-transfer"
HEAD = ROOT / "results/sketch5-preview-expanded"
RUN = ROOT / "runs/sketch5-vgcbench-v1"
BC_HASH = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"
BASELINE_HASH = "8e5812fa9bbb879aaeec32c706f3eee9bd8bf7de0ae102eff0be3d1c0f6a8e85"
UPSTREAM_SHA256 = {
    "vgc_bench/src/policy.py": "ab8969ebe18f88be955b19d1b9f77157a65ee23f38f9e8d1d5d601ae12f822a4",
    "vgc_bench/src/policy_player.py": "3c8cefbdea8d68a6504d9dd5fbf97a2d689b80cecb3e7522d67138b8b129d86f",
    "vgc_bench/src/utils.py": "534b93ea18806b84a52f21998b8485b56915794751ab77497e882734caaf0313",
    "data/abilities.json": "c6110d7963365d44d5741b204df8a92aa543db1d2a20a6e96233815c96d09d68",
    "data/items.json": "0bdc97131a61608de77cc1de8e4142d39561895d9cfc796d9f6a1fb576624c9b",
    "data/moves.json": "f0cb56065ec93cdf028f4ca187df0b9943a25a75812cd8a89d2c90e2ce24373e",
}
BOTS = ("random", "max-base-power", "simple-heuristics", "bc-learned", "bc-first-four")
CLASSES = dict(
    zip(BOTS[:3], (RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer), strict=True)
)


def specification(count):
    prior = read(HEAD / "manifest.json")
    source = read(EXTERNAL / "source.json")
    selections = read(HEAD / "selection-lock.json")
    fresh = [str((EXTERNAL / r["team"]).relative_to(ROOT)) for r in source["teams"]]
    assert all(digest(p) == r["sha256"] for p, r in zip(fresh, source["teams"], strict=True))
    teams = prior["teams"]["test"] + fresh
    assets = [
        *teams,
        prior["parent"],
        prior["team"],
        HEAD / "preview-head.pt",
        HEAD / "selection-lock.json",
        ROOT / "results/sketch5-preview/preview-head.pt",
        EXTERNAL / "source.json",
    ]
    assert digest(ZIP) == BC_HASH and digest(Path(baselines.__file__)) == BASELINE_HASH
    assert all(digest(CACHE / "source" / p) == h for p, h in UPSTREAM_SHA256.items())
    return dict(
        format=DEFAULT_FORMAT,
        open_team_sheets=False,
        learner=prior["parent"],
        learner_team=prior["team"],
        selectors=list(POLICIES),
        fallback_margin=selections["fallback_margin"],
        opponents=list(BOTS),
        teams=teams,
        external=fresh,
        battles_per_cell=count,
        seed_base=941000000,
        primary="bc-learned on the 12 external teams",
        simulator=read(ROOT / "simulator.lock.json"),
        source_revision=source["revision"],
        model_revision="204c76741829ca0681629e41382043c385850d5c",
        model_path="results/saves_bc/seed1/100.zip",
        model_sha256=BC_HASH,
        poke_env_baselines_sha256=BASELINE_HASH,
        versions={
            k: version(k) for k in ("poke-env", "stable-baselines3", "gymnasium", "torch", "numpy")
        },
        sha256={str((ROOT / p).relative_to(ROOT)): digest(p) for p in assets},
    )


@lru_cache(maxsize=2)
def simulator(fmt, sheets, directory):
    torch.set_num_threads(1)
    battle = LocalBattle(fmt, timeout=60, open_team_sheets=sheets, simulator=directory)
    atexit.register(battle.close)
    return battle


@lru_cache(maxsize=6)
def frozen_actor(learner):
    return FrozenActor(ROOT / learner)


@lru_cache(maxsize=5)
def rival(name, fmt):
    kwargs = dict(battle_format=fmt, start_listening=False)
    if name in CLASSES:
        return CLASSES[name](**kwargs)
    with chdir(CACHE / "source"):
        from vgc_bench.src.policy_player import MaskedActorCriticPolicy, PolicyPlayer
        from vgc_bench.src.utils import act_len, chunk_obs_len
    obs = spaces.Dict(
        {
            "observation": spaces.Box(-1, np.inf, shape=(12 * chunk_obs_len,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(2 * act_len),
        }
    )
    model = MaskedActorCriticPolicy(
        obs,
        spaces.MultiDiscrete([act_len] * 2),
        lambda _: 0.0,
        d_model=256,
        choose_on_teampreview=True,
    )
    with zipfile.ZipFile(ZIP) as z:
        weights = torch.load(
            io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True
        )
    model.load_state_dict(weights, strict=True)
    model.eval()
    return PolicyPlayer(policy=model, deterministic=False, **kwargs)


def seed_actions(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def game(spec, team, family, label, seed, side):
    battle = simulator(spec["format"], spec["open_team_sheets"], spec.get("simulator_path"))
    actor = frozen_actor(spec["learner"])
    actor.reset()
    other = rival(family, spec["format"])
    foe = "p2" if side == "p1" else "p1"
    picks = {}

    def learner_preview(public):
        picks["public"] = public
        picks["lineup"] = pick(HEAD, label, spec["fallback_margin"], public)
        return tuple(map(int, picks["lineup"]))

    def opponent_preview(_):
        view = battle.players[foe]
        assert view.teampreview and len(view.team) == len(view.opponent_team) == 6
        assert view.format == spec["format"]
        if spec["open_team_sheets"]:
            assert all(len(mon.moves) == 4 for mon in view.opponent_team.values())
            assert all(len(mon.moves) == 4 for mon in battle.players[side].opponent_team.values())
        # On this Showdown revision the preview request marks the first two slots
        # active. All six are legal draft choices; the upstream converter also
        # checks available_switches when decoding its two masked preview actions.
        if family == "bc-learned":
            for switches in view.available_switches:
                switches[:] = view.team.values()
        if family == "bc-first-four":
            for mon in list(view.team.values())[:4]:
                mon._selected_in_teampreview = True
            lineup = "1234"
        else:
            lineup = other.teampreview(view).removeprefix("/team ")
        picks["opponent_lineup"] = lineup
        return tuple(map(int, lineup))

    seed_actions(seed)
    texts = [(ROOT / t).read_text() for t in (spec["learner_team"], team)]
    battle.reset(
        seed,
        *(texts if side == "p1" else reversed(texts)),
        preview_pickers={side: learner_preview, foe: opponent_preview},
    )
    # Matching the battle-action stream across preview interventions avoids RNG drift.
    seed_actions(seed + 1000000000)
    requests = defaults = 0
    while not battle.ended and battle.turn < 200 and requests < 600:
        choices = {}
        for seat in ("p1", "p2"):
            if not battle.needs_action(seat):
                continue
            view = battle.players[seat]
            if seat == side:
                legal = battle.legal_actions(seat)
                order = legal[actor.choose(view, legal)]
            else:
                order = other.choose_move(view)
                defaults += int(order.message == "/choose default")
            choices[seat] = order.message.removeprefix("/choose ")
        battle.step(choices)
        requests += 1
    winner = "Learner" if side == "p1" else "Opponent"
    outcome = (
        "cutoff"
        if not battle.ended
        else "tie"
        if not battle.winner
        else "win"
        if battle.winner == winner
        else "loss"
    )
    return dict(
        roster=Path(team).stem,
        opponent_team=team,
        family=family,
        seed=seed,
        side=side,
        lineup=picks["lineup"],
        opponent_lineup=picks["opponent_lineup"],
        outcome=outcome,
        turn=battle.turn,
        opponent_defaults=defaults,
    ), picks["public"]


def cell(spec, root, i, family, start, count):
    team = spec["teams"][i]
    dest = root / "cells" / Path(team).stem / f"{family}-{start}.csv"
    if dest.is_file():
        return
    stream = min(BOTS.index(family), 3)  # Share BC seeds for the preview control.
    base = spec["seed_base"] + (i * 4 + stream) * 1000
    rows = []
    for k in range(start, start + count):
        cache = {}
        public = None
        for label in POLICIES:
            lineup = pick(HEAD, label, spec["fallback_margin"], public) if public else None
            if lineup not in cache:
                row, public = game(
                    spec, team, family, label, base + k, "p1" if k % 2 == 0 else "p2"
                )
                cache[row["lineup"]] = row
            else:
                row = cache[lineup]
            rows.append(dict(candidate=label, **row))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    write_csv(temp, rows)
    temp.replace(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--battles", type=int, default=64)
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--verify", action="store_true", help="Replay ten scored games in a fresh process"
    )
    args = p.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        p.error("Set PYTHONHASHSEED=0 for reproducible external policies")
    if args.battles < 4 or args.battles > 1000 or args.battles % 2:
        p.error("Use an even number of battles from 4 to 1000")
    spec = specification(args.battles)
    if args.verify:
        assert read(args.run / "manifest.json") == spec
        verified = []
        for j, family in enumerate(BOTS):
            for i, team in enumerate((spec["teams"][j], spec["external"][j])):
                with (args.run / "cells" / Path(team).stem / f"{family}-0.csv").open() as stream:
                    recorded = list(csv.DictReader(stream))[4 * i + (i + j) % len(POLICIES)]
                actual, _ = game(
                    spec,
                    team,
                    family,
                    recorded["candidate"],
                    int(recorded["seed"]),
                    recorded["side"],
                )
                assert all(recorded[k] == str(v) for k, v in actual.items()), (recorded, actual)
                verified.append(recorded)
        write_csv(args.run / "reproduced-games.csv", verified)
        print(f"Reproduced {len(verified)} scored games")
        return
    if args.smoke:
        for f in BOTS:
            for i, t in enumerate((spec["teams"][0], spec["external"][0])):
                print(
                    game(spec, t, f, POLICIES[i], 101208 + i, "p1" if i == 0 else "p2")[0],
                    flush=True,
                )
        return
    args.run.mkdir(parents=True, exist_ok=True)
    lock = args.run / "manifest.json"
    if lock.exists():
        assert spec == read(lock)
    else:
        lock.write_text(json.dumps(spec, indent=2) + "\n")
    errors = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        tasks = {
            pool.submit(cell, spec, args.run, i, f, k, min(8, args.battles - k)): (t, f, k)
            for i, t in enumerate(spec["teams"])
            for f in BOTS
            for k in range(0, args.battles, 8)
        }
        for n, future in enumerate(as_completed(tasks), 1):
            try:
                future.result()
            except Exception as exc:
                team, family, offset = tasks[future]
                path = args.run / f"failure-{Path(team).stem}-{family}-{offset}.txt"
                path.write_text(traceback.format_exc())
                errors.append(path)
                print(f"Failed {tasks[future]}: {exc}", flush=True)
                continue
            if n % 40 == 0 or n == len(tasks):
                print(f"Completed {n}/{len(tasks)} batches", flush=True)

    if errors:
        raise RuntimeError(f"{len(errors)} failed batches; see {errors[0]}")


if __name__ == "__main__":
    main()
