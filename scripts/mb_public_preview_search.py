"""Plan once from a legal open team sheet using independent simulated battles."""

import argparse
import hashlib
import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, RUN, SEEDS, SPEC, decide, schedule, verify
from mb_residual_preview import pilot
from vgc_posttrain import Match, atomic_rows, batch_obs, team_text


def public_sets(view):
    # Whitelist exactly what the format reveals. Sorting removes opponent slot order.
    mons = [
        (m.species, m.ability, m.item, tuple(sorted(m.moves))) for m in view.opponent_team.values()
    ]
    assert len(mons) == 6 and all(a and item and len(m) == 4 for _, a, item, m in mons)
    return sorted(mons)


def inferred_sheet(public, sample):
    # Infer spreads from public species and move categories. Never load a tournament
    # opponent's private spread or actual chosen four into the planning simulator.
    from poke_env.battle import Move, MoveCategory, Pokemon

    rng = np.random.default_rng(sample)
    entries = []
    for species, ability, item, moves in public:
        mon = Pokemon(gen=9, species=species)
        attacks = [Move(m, gen=9) for m in moves]
        phys = sum(m.base_power for m in attacks if m.category == MoveCategory.PHYSICAL)
        spec = sum(m.base_power for m in attacks if m.category == MoveCategory.SPECIAL)
        physical = phys * mon.base_stats["atk"] > spec * mon.base_stats["spa"]
        stat = "Atk" if physical else "SpA"
        speed = mon.base_stats["spe"] >= 80 and rng.random() < 0.8
        if phys == spec == 0:
            stats = "32 HP / 18 Def / 16 SpD"
            nature = "Calm"
        elif speed:
            stats = f"2 HP / 32 {stat} / 32 Spe"
            nature = (
                ("Jolly" if physical else "Timid")
                if rng.random() < 0.65
                else ("Adamant" if physical else "Modest")
            )
        else:
            stats = f"32 HP / 32 {stat} / 2 SpD"
            nature = "Adamant" if physical else "Modest"
        item_text = f" @ {item}" if item != "unknown_item" else ""
        entries.append(
            "\n".join(
                [
                    species + item_text,
                    f"Ability: {ability}",
                    "Level: 50",
                    f"EVs: {stats}",
                    f"{nature} Nature",
                    *["- " + m for m in moves],
                ]
            )
        )
    return "\n\n".join(entries) + "\n"


def public_fixture(own, foe):
    # This adapter extracts the observation at preview and throws away the match.
    # The planner itself receives only the player's own roster and this public tuple.
    match = Match()
    seen = []

    def snapshot(view):
        seen.extend(public_sets(view))
        return "1234"

    match.reset(own, foe, 570033815, "p1", preview_picker=snapshot)
    return seen


def key_for(own, public, seed):
    data = json.dumps([team_text(own), public, seed, "public-preview-v1"], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


@torch.no_grad()
def plan(own, public, seed):
    model = pilot(seed)
    key = key_for(own, public, seed)
    gen = np.random.default_rng(int(key[:16], 16))
    match = Match()
    candidates = {}

    def add(line):
        # Preserve native slot order for the first representative of each (lead pair, back pair).
        identity = "".join(sorted(line[:2])) + "." + "".join(sorted(line[2:]))
        candidates.setdefault(identity, line)

    with tempfile.TemporaryDirectory(dir=RUN) as work:
        guesses = []
        for which in range(4):
            path = Path(work) / f"belief-{which}.txt"
            path.write_text(inferred_sheet(public, int(gen.integers(2**31))))
            guesses.append(str(path))
        for n in range(24):
            match.reset(
                own,
                guesses[n % 4],
                int(gen.integers(1_000_000, 180_000_000)),
                "p1",
                preview="native",
            )
            add(match.lineup)
            if len(candidates) == 6:
                break
        while len(candidates) < 8:
            add("".join(str(x) for x in gen.choice(np.arange(1, 7), 4, replace=False)))
        active = list(candidates.values())
        totals = {c: 0 for c in active}
        counts = {c: 0 for c in active}
        for stage, repeats in enumerate((4, 4, 8)):
            for rep in range(repeats):
                game_seed = int(gen.integers(1_000_000, 180_000_000))
                for line in active:
                    match.reset(
                        own, guesses[rep % 4], game_seed, ("p1", "p2")[rep % 2], lineup=line
                    )
                    torch.manual_seed(game_seed + 700_000_000)
                    done = False
                    while not done:
                        act = decide(model, batch_obs(match.obs()), "map", seed)
                        done, reward = match.step(act)
                    totals[line] += (reward + 1) / 2
                    counts[line] += 1
            active.sort(key=lambda c: totals[c] / counts[c], reverse=True)
            active = active[: len(active) // 2]
        assert len(active) == 1 and sum(counts.values()) == 64
        return dict(key=key, seed=seed, selected=active[0], counts=counts, totals=totals)


def read_plan(own, public, seed):
    key = key_for(own, public, seed)
    dest = RUN / "public-preview" / (key + ".json")
    if dest.exists():
        result = json.loads(dest.read_text())
        assert result["key"] == key and result["seed"] == seed
        return result
    result = plan(own, public, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, suffix=".tmp", delete=False) as tmp:
        tmp.write(json.dumps(result) + "\n")
    Path(tmp.name).replace(dest)
    return result


@torch.no_grad()
def score(task):
    seed, cell, battle_mode = task
    i, j, direction, own, foe, start = cell
    public = public_fixture(own, foe)
    picked = read_plan(own, public, seed)
    model, match = pilot(seed), Match()
    rows = []
    for k in range(8):
        sim, side = start + k, ("p1", "p2")[k % 2]

        def picker(view):
            assert public_sets(view) == public
            return picked["selected"]

        match.reset(own, foe, sim, side, preview_picker=picker)
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            choice = decide(model, batch_obs(match.obs()), battle_mode, seed, match.view)
            done, reward = match.step(choice)
        rows.append(
            dict(
                method=f"search-{battle_mode}",
                training_seed=seed,
                i=i,
                j=j,
                direction=direction,
                k=k,
                own=Path(own).stem,
                foe=Path(foe).stem,
                sim=sim,
                side=side,
                lineup=match.lineup,
                opp_lineup=match.opponent.lineup,
                reward=reward,
                turns=match.battle.turn,
                public_key=picked["key"],
            )
        )
    return rows


def evaluate(split, jobs, battle_mode):
    cells = list(schedule(split))
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for seed in SEEDS:
            dest = OUT / "scores" / f"{split}-search-{battle_mode}-{seed}.csv"
            if dest.exists():
                continue
            rows = [
                r
                for group in pool.map(score, [(seed, c, battle_mode) for c in cells])
                for r in group
            ]
            assert len(rows) == 8 * len(cells)
            atomic_rows(dest, rows)
            print(split, seed, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "validation", "custom-dev"))
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument(
        "--battle",
        choices=(
            "map",
            "es-initial",
            "es-trained",
            "mcq-40-8",
            "context-ppo",
            "endgame-es",
            "belief-es",
            "wide-belief-es",
            "multistep-es",
            "joint-ppo-16384",
            "joint-ppo-65536",
        ),
        default="map",
    )
    args = p.parse_args()
    verify()
    if args.stage == "check":
        own, foe = SPEC["own_train"][3], SPEC["foe_train"][0]
        public = public_fixture(own, foe)
        # Change private stats and reorder the opponent's six. The search key and
        # generated model of that opponent must remain identical.
        with tempfile.TemporaryDirectory(dir=RUN) as directory:
            p = Path(directory) / "different-private.txt"
            p.write_text("\n\n".join(reversed(inferred_sheet(public, 0).strip().split("\n\n"))))
            assert public_fixture(own, str(p)) == public
        result = plan(own, public, 2)
        print(
            "hidden-spread invariance and 64 independent rollout games verified", result, flush=True
        )
    else:
        evaluate(args.stage, args.jobs, args.battle)
