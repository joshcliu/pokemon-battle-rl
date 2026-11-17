"""Replay paired frozen-policy battles, recording commands and two evaluation interventions."""

import argparse
import csv
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from analyze_policy_size import locations
from analyze_rewards import records
from poke_env.battle import Move, Pokemon
from poke_env.data import GenData
from recover_action_checkpoints import BUDGETS, FULL, ORIGINAL
from roster_transfer import EVAL_SEED, SEEDS, STYLES, split

from pokemon_rl.battle import ACTION_SIZE, ROOT
from pokemon_rl.env import BattleEnv
from pokemon_rl.history import DecisionWindow, FullBattleHistory, make_policy

N = 25
MODES = ("16k", "32k", "native", "rotate", "retarget", "small")
HISTORY_LIMITS = {"last1": 1, "last8": 8}
CHART = GenData.from_gen(9).type_chart


def rotate(team):
    """Cyclically relabel all four move slots, retaining every move on its original Pokémon."""
    lines = team.splitlines()
    slots = [i for i, line in enumerate(lines) if line.startswith("- ")]
    assert len(slots) == 24
    for offset in range(0, len(slots), 4):
        block = slots[offset : offset + 4]
        assert block[-1] - block[0] == 3
        moves = [lines[i] for i in block]
        for i, move in zip(block, moves[1:] + moves[:1], strict=True):
            lines[i] = move
    return "\n".join(lines) + "\n"


def matchup(move, foe):
    return move.type.damage_multiplier(*foe.types, type_chart=CHART)


def ordinary(move):
    return (
        isinstance(move, Move)
        and move.base_power > 0
        and move.id not in {"freezedry", "flyingpress", "weatherball", "terablast"}
    )


def improve_target(pair, legal, player):
    ids = list(pair)
    edits = 0
    for pos in (0, 1):
        joint = legal[tuple(ids)]
        order = (joint.first_order, joint.second_order)[pos]
        move, current = order.order, order.move_target
        if not ordinary(move) or current == 0:
            continue
        choices = []
        for candidate, combo in legal.items():
            if candidate[1 - pos] != ids[1 - pos]:
                continue
            alt = (combo.first_order, combo.second_order)[pos]
            if not (
                isinstance(alt.order, Move)
                and alt.order.id == move.id
                and alt.mega == order.mega
                and alt.move_target > 0
            ):
                continue
            foe = player.opponent_active_pokemon[alt.move_target - 1]
            if foe is not None:
                choices.append((matchup(move, foe), -alt.move_target, candidate[pos]))
        if not choices:
            continue
        best, _, action_id = max(choices)
        target = player.opponent_active_pokemon[current - 1] if current > 0 else None
        if current < 0 or (target is not None and best > matchup(move, target)):
            ids[pos] = action_id
            edits += 1
    return tuple(ids), edits


def orders(pair, legal, player, info):
    chosen = legal[pair]
    result = []
    for pos, order in enumerate((chosen.first_order, chosen.second_order)):
        actor = player.active_pokemon[pos]
        move = order.order
        is_move = isinstance(move, Move)
        target = order.move_target if is_move else 0
        mon = (
            player.opponent_active_pokemon[target - 1]
            if target > 0
            else player.active_pokemon[-target - 1]
            if target < 0
            else None
        )
        mult, alternative = None, None
        if ordinary(move) and target > 0 and mon is not None:
            mult = matchup(move, mon)
            other = player.opponent_active_pokemon[2 - target]
            # Keep the partner's choice, move, and gimmick fixed.
            other_id = pair[pos] + (3 - 2 * target)
            alt_pair = (other_id, pair[1]) if pos == 0 else (pair[0], other_id)
            if other is not None and alt_pair in legal:
                alt = (legal[alt_pair].first_order, legal[alt_pair].second_order)[pos]
                assert (
                    alt.order.id == move.id
                    and alt.move_target == 3 - target
                    and alt.mega == order.mega
                )
                alternative = matchup(move, other)
        if is_move:
            kind = "status" if move.category.name == "STATUS" else "attack"
        elif isinstance(move, Pokemon):
            kind = "forced_switch" if player.force_switch[pos] else "switch"
        else:
            kind = "pass"
        result.append(
            info
            | {
                "pos": pos,
                "actor": actor.species if actor else "",
                "kind": kind,
                "choice": move.id
                if is_move
                else move.species
                if isinstance(move, Pokemon)
                else str(move),
                "move_slot": ((pair[pos] - 7) % 20) // 5 + 1 if is_move else "",
                "target": target,
                "target_species": mon.species if mon else "",
                "ally_attack": int(kind == "attack" and target < 0 and -target - 1 != pos),
                "mega": int(is_move and order.mega),
                "multiplier": mult,
                "other_multiplier": alternative,
            }
        )
    return result


def event_counts(lines):
    counts = Counter()
    pending = None
    for line in lines:
        p = line.split("|")
        if len(p) < 2:
            continue
        if p[1] in {"move", "cant", "turn", "upkeep", "switch", "drag", ""}:
            pending = None
        if p[1] == "move" and p[2].startswith("p1"):
            pending = p[3]
            counts["executed_moves"] += 1
            if pending == "Fake Out":
                counts["fake_out_uses"] += 1
            if pending == "Protect":
                counts["protect_uses"] += 1
        elif pending and p[1] == "-fail":
            counts["move_fail_events"] += 1
            counts["fake_out_fails"] += pending == "Fake Out"
            counts["protect_fails"] += pending == "Protect"
            pending = None
    return {
        k: counts[k]
        for k in (
            "executed_moves",
            "move_fail_events",
            "fake_out_uses",
            "fake_out_fails",
            "protect_uses",
            "protect_fails",
        )
    }


def load_policy(root, seed, mode):
    original, ev = locations(ORIGINAL, seed, "chunks-1" if mode == "small" else FULL)
    if mode in ("16k", "32k"):
        budget = BUDGETS[0 if mode == "16k" else 1]
        p = root / f"early/seed-{seed}/train/checkpoint-{budget}.pt"
    else:
        p = original
        with (ROOT / "results/sketch5-policy-size/checkpoints.csv").open() as f:
            expected = next(
                r["sha256"]
                for r in csv.DictReader(f)
                if int(r["seed"]) == seed
                and r["model"] == ("chunks-1" if mode == "small" else FULL)
            )
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
    saved = torch.load(p, weights_only=True, map_location="cpu")
    model = make_policy(saved["obs_size"], saved["hidden_size"], saved["transformer_chunks"])
    model.load_state_dict(saved["policy"])
    return model.eval(), ev


@torch.no_grad()
def play(root, mode, seed, roster, group, n):
    torch.set_num_threads(1)
    model, archive = load_policy(root, seed, mode)
    short = Path(roster).stem
    out = root / "trace" / mode / f"seed-{seed}" / short
    out.mkdir(parents=True, exist_ok=False)
    games, actions = [], []
    for style in STYLES:
        base = BattleEnv(opponent=style, team=roster)
        if mode == "rotate":
            base.learner_teams = [rotate(t) for t in base.learner_teams]
        env = DecisionWindow(base, 1) if mode == "small" else FullBattleHistory(base)
        archived = (
            records(archive / short / "games" / f"{style}.jsonl") if group == "held_out" else []
        )
        try:
            for i in range(n):
                es = EVAL_SEED + i
                key = {
                    "mode": mode,
                    "seed": seed,
                    "split": group,
                    "roster": short,
                    "style": style,
                    "eval_seed": es,
                }
                torch.manual_seed(es)
                obs, start = env.reset(seed=es)
                if archived:
                    assert (es, start["simulator_seed"], start["opponent_team_index"]) == tuple(
                        archived[i][k] for k in ("seed", "simulator_seed", "opponent_team_index")
                    )
                edits = 0
                decisions = 0
                while True:
                    legal = base.legal()
                    mask = np.zeros((ACTION_SIZE, ACTION_SIZE), bool)
                    for a, b in legal:
                        mask[a, b] = True
                    prefix = obs[-HISTORY_LIMITS[mode] :] if mode in HISTORY_LIMITS else obs
                    selected, logp, _, _ = model.sample(
                        torch.from_numpy(prefix)[None], torch.from_numpy(mask)[None]
                    )
                    pair = tuple(selected[0].tolist())
                    player = base.battle.players["p1"]
                    if mode == "retarget":
                        pair, changes = improve_target(pair, legal, player)
                        edits += changes
                    decisions += 1
                    if mode in ("native", "small") and group == "held_out":
                        actions.extend(
                            orders(
                                pair,
                                legal,
                                player,
                                key
                                | {
                                    "decision": decisions,
                                    "turn": base.battle.turn,
                                    "joint_prob": float(logp.exp().item()),
                                },
                            )
                        )
                    obs, _, done, cutoff, info = env.step(np.asarray(pair))
                    if done or cutoff:
                        break
                if archived and mode in ("native", "small"):
                    assert (info["turn"], info["outcome"]) == (
                        archived[i]["turn"],
                        archived[i]["outcome"],
                    )
                games.append(
                    key
                    | {
                        "opponent_team_index": start["opponent_team_index"],
                        "simulator_seed": start["simulator_seed"],
                        "outcome": info["outcome"],
                        "turns": info["turn"],
                        "decisions": decisions,
                        "edits": edits,
                        **event_counts(base.battle.log),
                    }
                )
                if i < 6 and group == "held_out" and mode in ("native", "small", "retarget"):
                    (out / f"{style}-{es}.log").write_text("\n".join(base.battle.log) + "\n")
        finally:
            env.close()
    write_csv(out / "games.csv", games)
    if actions:
        write_csv(out / "actions.csv.gz", actions)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--modes", nargs="+", choices=(*MODES, *HISTORY_LIMITS), default=MODES)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--held-out-only", action="store_true")
    p.add_argument("--battles", type=int, default=N, choices=range(1, 101), metavar="1..100")
    args = p.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [
            pool.submit(play, args.runs.resolve(), mode, seed, roster, group, args.battles)
            for mode in args.modes
            for seed in SEEDS
            for label, group in (("test", "held_out"), ("train", "practice"))
            if label == "test" or (not args.held_out_only and mode not in ("small", "retarget"))
            for roster in split()[label]
        ]
        for task in tasks:
            task.result()
