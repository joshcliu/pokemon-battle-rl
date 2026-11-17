"""Measure whether target preference follows an isolated swap of opponent type features."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from analyze_roster_transfer import OPPONENT_TEAMS
from poke_env.battle import Move
from poke_env.data import GenData
from roster_transfer import EVAL_SEED, split

from pokemon_rl.battle import ACTION_SIZE
from pokemon_rl.env import BattleEnv
from pokemon_rl.features import MON_FEATURES, TYPES
from pokemon_rl.history import make_policy


def probe_cases():
    cases = []
    chart = GenData.from_gen(9).type_chart
    for roster in split()["test"]:
        for opponent in OPPONENT_TEAMS:
            env = BattleEnv(team=roster, opponent_teams=[opponent])
            try:
                board, _ = env.reset(seed=EVAL_SEED)
                player = env.battle.players["p1"]
                foes = player.opponent_active_pokemon
                # The last six mon records encode opponents in player-visible order.
                mon_rows = board[-6 * MON_FEATURES :].reshape(6, MON_FEATURES)
                rows = [np.flatnonzero(mon_rows[:, 4 + i]).item() for i in (0, 1)]
                typed = slice(6, 6 + len(TYPES))
                for i, foe in zip(rows, foes, strict=True):
                    np.testing.assert_array_equal(
                        mon_rows[i, typed], [t in foe.types for t in TYPES]
                    )
                swapped = board.copy()
                swapped[-6 * MON_FEATURES :].reshape(6, MON_FEATURES)[rows, typed] = mon_rows[
                    rows[::-1], typed
                ]

                moves = defaultdict(dict)
                for (a1, _), pair in env.legal().items():
                    order = pair.first_order
                    move = order.order
                    if (
                        isinstance(move, Move)
                        and move.base_power > 0
                        and not order.mega
                        and order.move_target in (1, 2)
                    ):
                        moves[move.id][order.move_target] = (a1, move)
                for name, choices in sorted(moves.items()):
                    if set(choices) != {1, 2} or name in {"flyingpress", "freezedry"}:
                        continue
                    move = choices[1][1]
                    multipliers = [
                        move.type.damage_multiplier(*foe.types, type_chart=chart) for foe in foes
                    ]
                    if multipliers[0] == multipliers[1]:
                        continue
                    cases.append(
                        (
                            {
                                "roster": Path(roster).stem,
                                "opponent_team": Path(opponent).stem,
                                "move": name,
                                "move_type": move.type.name,
                                "left_type_multiplier": multipliers[0],
                                "right_type_multiplier": multipliers[1],
                            },
                            np.stack([board, swapped]),
                            [choices[t][0] for t in (1, 2)],
                            0 if multipliers[0] > multipliers[1] else 1,
                        )
                    )
            finally:
                env.close()
    if not cases:
        raise ValueError("No eligible target pairs")
    return cases


@torch.inference_mode()
def score_targets(checkpoint, cases, condition):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    chunks = saved.get("transformer_chunks")
    policy = make_policy(saved["obs_size"], saved["hidden_size"], chunks)
    policy.load_state_dict(saved["policy"])
    policy.eval()
    boards = torch.from_numpy(np.concatenate([pair for _, pair, _, _ in cases]))
    if chunks:
        # Every probe is the first decision. Match DecisionWindow.reset exactly.
        start = torch.full((len(boards), 2), float(ACTION_SIZE))
        token = torch.cat((boards, start), dim=1)
        length = 1 if chunks == "full" else chunks
        boards = token.new_zeros((len(token), length, token.shape[1]))
        boards[:, :, -2:] = -1
        boards[:, 0] = token
    logits = policy.first(policy.encoder(boards)).reshape(len(cases), 2, ACTION_SIZE)
    rows = []
    for (key, _, ids, better), views in zip(cases, logits, strict=True):
        # Conditional probability among the two legal target choices for this move.
        probs = views[:, ids].softmax(dim=1)
        original, swapped = probs[0, better].item(), probs[1, 1 - better].item()
        rows.append(
            condition
            | key
            | {
                "original_preferred_pct": 100 * original,
                "swapped_preferred_pct": 100 * swapped,
                "type_probe_pct": 50 * (original + swapped),
            }
        )
    return rows
