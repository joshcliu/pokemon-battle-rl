"""Pair final setup games, extract opening and turn-two events, and export replay evidence."""

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from analyze_long_rewards import load, write_csv
from analyze_rewards import records
from opponent_diversity import BATTLES, EVAL_SEED, SEEDS
from reconstruct_setup_games import ORIGINAL

POLICIES = ("damage", "mixture")
CASES = {(2, 500015), (3, 500000), (4, 500012), (5, 500030)}
BOOST_MOVES = {"Dragon Dance", "Calm Mind", "Swords Dance"}


def replay_events(path):
    turn = 0
    knocked = {"p1": [], "p2": []}
    boost_uses = 0
    non_setup_opener = False
    protect_block = False
    pending = None
    winner = None
    for line in path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        kind = parts[1]
        if kind == "turn":
            turn = int(parts[2])
            pending = None
        elif kind == "win":
            winner = parts[2]
        elif 1 <= turn <= 2:
            if (
                turn == 1
                and kind == "move"
                and parts[2].startswith("p2")
                and parts[3] not in BOOST_MOVES
            ):
                non_setup_opener = True
            if turn == 1 and parts[1:4] == ["-activate", "p1b: Garchomp", "move: Protect"]:
                protect_block = True
            if kind == "faint":
                knocked[parts[2][:2]].append(parts[2].split(": ", 1)[1])
            elif kind == "move":
                pending = (
                    parts[2] if parts[2].startswith("p2") and parts[3] in BOOST_MOVES else None
                )
            elif kind == "-boost" and pending == parts[2] and int(parts[4]) > 0:
                boost_uses += 1
                pending = None
    return winner, {
        "foe_used_non_setup_move_t1": int(non_setup_opener),
        "garchomp_protect_block_t1": int(protect_block),
        "enemy_kos_by_t2": len(knocked["p2"]),
        "own_kos_by_t2": len(knocked["p1"]),
        "foes_fainted_by_t2": "|".join(knocked["p2"]),
        "own_fainted_by_t2": "|".join(knocked["p1"]),
        "successful_enemy_setup_moves_by_t2": boost_uses,
    }


def compare(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    with (raw / "opening-swaps.csv").open() as f:
        swaps = {
            (int(r["training_seed"]), int(r["eval_seed"]), r["recipient"]): r
            for r in csv.DictReader(f)
        }
    assert len(swaps) == 2 * len(SEEDS) * BATTLES
    groups = Counter()
    metrics = defaultdict(Counter)
    swaps_by_group = defaultdict(Counter)
    features = []
    with ZipFile(out / "damage-win-mix-loss-replays.zip", "w", compression=ZIP_DEFLATED) as archive:
        for seed in SEEDS:
            bases = {c: raw / f"seed-{seed}" / c / "final" for c in POLICIES}
            games = {c: records(bases[c] / "setup.jsonl") for c in POLICIES}
            for c in POLICIES:
                assert games[c] == records(ORIGINAL / f"seed-{seed}" / c / "eval/setup.jsonl")
            for i in range(BATTLES):
                d, m = [games[c][i] for c in POLICIES]
                assert all(
                    d[k] == m[k]
                    for k in ("episode", "seed", "simulator_seed", "opponent_team_index")
                )
                assert (d["episode"], d["seed"]) == (i, EVAL_SEED + i)
                team = ("balance", "sun")[d["opponent_team_index"]]
                group = f"{d['outcome']}/{m['outcome']}"
                groups[team, group] += 1
                for c, g in zip(POLICIES, (d, m), strict=True):
                    base = bases[c] / "replays" / f"setup-{i:04}"
                    log = base.with_suffix(".log")
                    winner, metrics_row = replay_events(log)
                    assert winner == ("Learner" if g["outcome"] == "win" else "Opponent")
                    choices = load(base.with_suffix(".input.json"))
                    opening = choices[2]["choices"]["p1"]
                    swap = swaps[seed, g["seed"], c]
                    assert (
                        swap["original_opening"] == opening
                        and swap["original_outcome"] == g["outcome"]
                    )
                    assert swap["swapped_outcome"] in {"win", "loss"}
                    obs = {
                        "foe_used_non_setup_move_t1": metrics_row["foe_used_non_setup_move_t1"],
                        "garchomp_protect_block_t1": metrics_row["garchomp_protect_block_t1"],
                        "knocked_out_enemy_by_t2": int(metrics_row["enemy_kos_by_t2"] > 0),
                        "lost_own_by_t2": int(metrics_row["own_kos_by_t2"] > 0),
                        "both_enemy_leads_survived_t2": int(metrics_row["enemy_kos_by_t2"] == 0),
                        "foe_setup_moves_by_t2": metrics_row["successful_enemy_setup_moves_by_t2"],
                        "opening_garchomp_protect": int(
                            opening.split(", ")[1].startswith("move protect")
                        ),
                        "opening_garchomp_to_scizor": int(
                            opening.split(", ")[1] == "switch Scizor"
                        ),
                        "opening_garchomp_rockslide": int(
                            opening.split(", ")[1].startswith("move rockslide")
                        ),
                    }
                    for roster in (team, "all"):
                        for scope in (group, "all"):
                            metrics[roster, scope, c].update(obs | {"games": 1})
                            swaps_by_group[roster, scope, c].update(
                                {
                                    "games": 1,
                                    "native_wins": int(g["outcome"] == "win"),
                                    "swapped_wins": int(swap["swapped_outcome"] == "win"),
                                    "same_opening": int(opening == swap["substituted_opening"]),
                                }
                            )
                    features.append(
                        {
                            "training_seed": seed,
                            "eval_seed": g["seed"],
                            "simulator_seed": g["simulator_seed"],
                            "roster": team,
                            "damage_mix_outcomes": group,
                            "policy": c,
                            "turns": g["turn"],
                            "opening": opening,
                            "swapped_outcome": swap["swapped_outcome"],
                            **metrics_row,
                            **obs,
                        }
                    )
                    stem = f"seed-{seed}-battle-{g['seed']}-{team}/{c}"
                    if group == "win/loss":
                        for suffix in (".log", ".input.json"):
                            archive.write(base.with_suffix(suffix), stem + suffix)
                    if (seed, g["seed"]) in CASES:
                        assert group == "win/loss"
                        for suffix in (".log", ".html"):
                            dest = out / "examples" / (stem + suffix)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(base.with_suffix(suffix), dest)
    write_csv(out / "replay-events.csv.gz", features)
    for filename, table in [
        ("early-game-counts.csv", metrics),
        ("opening-swap-counts.csv", swaps_by_group),
    ]:
        write_csv(
            out / filename,
            [
                {"roster": r, "damage_mix_outcomes": g, "policy": p, **counts}
                for (r, g, p), counts in sorted(table.items())
            ],
        )
    write_csv(
        out / "paired-outcomes.csv",
        [
            {"roster": r, "damage_mix_outcomes": g, "battles": v}
            for (r, g), v in sorted(groups.items())
        ],
    )
    print(json.dumps({str(k): v for k, v in sorted(groups.items())}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    compare(args.runs, args.output)
