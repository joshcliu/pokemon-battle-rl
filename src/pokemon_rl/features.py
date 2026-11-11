"""Compact, explicit features from one player's observed battle history only."""

import numpy as np
from poke_env.battle import Field, PokemonType, SideCondition, Status, Weather

TYPES = list(PokemonType)
STATUSES = list(Status)
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
MON_FEATURES = 6 + len(TYPES) + len(STATUSES) + len(STATS) + len(BOOSTS) + 4 * (8 + len(TYPES))
OBSERVATION_SIZE = 5 + len(Weather) + len(Field) + 2 * len(SideCondition) + 12 * MON_FEATURES


def move_features(move):
    if move is None:
        return [0.0] * (8 + len(TYPES))
    return [
        1.0,
        min(move.base_power, 300) / 300,
        float(move.accuracy),
        float(move.category.name == "PHYSICAL"),
        float(move.category.name == "SPECIAL"),
        float(move.category.name == "STATUS"),
        move.priority / 7,
        move.current_pp / max(move.max_pp, 1),
        *[float(move.type == t) for t in TYPES],
    ]


def mon_features(mon, active, selected=True):
    if mon is None:
        return [0.0] * MON_FEATURES
    moves = list(mon.moves.values())[:4]
    return [
        1.0,
        mon.current_hp_fraction,
        float(mon.fainted),
        float(selected),
        float(mon is active[0]),
        float(mon is active[1]),
        *[float(t in mon.types) for t in TYPES],
        *[float(mon.status == status) for status in STATUSES],
        *[mon.base_stats.get(stat, 0) / 255 for stat in STATS],
        *[mon.boosts.get(stat, 0) / 6 for stat in BOOSTS],
        *[value for move in moves + [None] * (4 - len(moves)) for value in move_features(move)],
    ]


def observation(player):
    """No simulator state, opponent requests, opponent identity, or team-file lookup."""
    values = [
        min(player.turn, 200) / 200,
        *map(float, player.force_switch),
        *map(float, player.can_mega_evolve),
        *[float(w in player.weather) for w in Weather],
        *[float(f in player.fields) for f in Field],
        *[float(c in player.side_conditions) for c in SideCondition],
        *[float(c in player.opponent_side_conditions) for c in SideCondition],
    ]
    selected = {p["ident"] for p in player.last_request.get("side", {}).get("pokemon", [])}
    own = list(player.team.items())[:6]
    for ident, mon in own:
        values.extend(mon_features(mon, player.active_pokemon, ident in selected))
    for _ in range(6 - len(own)):
        values.extend(mon_features(None, [None, None]))
    # Only encountered opponents appear here. Team-preview species are not inferred
    # into unrevealed moves, abilities, items, stats, or health.
    opponents = list(player.opponent_team.values())[:6]
    for mon in opponents + [None] * (6 - len(opponents)):
        values.extend(mon_features(mon, player.opponent_active_pokemon))
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Unexpected observation shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite battle observation")
    return array
