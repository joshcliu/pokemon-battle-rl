"""Small, disclosed baselines. The damage score is a proxy, not a damage calculator."""

import numpy as np
from poke_env.battle import Move, Pokemon
from poke_env.battle.target import Target

STYLES = ("random", "damage", "switch", "setup", "mixture")


def damage_score(order, player, pos):
    mon = player.active_pokemon[pos]
    move = order.order
    if not isinstance(move, Move) or mon is None:
        return 0.0
    if move.base_power == 0:
        return 0.0
    if order.move_target > 0:
        targets = [player.opponent_active_pokemon[order.move_target - 1]]
    elif order.move_target < 0:
        return -float(move.base_power)  # Deliberate friendly fire is not a baseline goal.
    else:
        targets = [p for p in player.opponent_active_pokemon if p is not None]
    effectiveness = sum(p.damage_multiplier(move) for p in targets if p is not None)
    stab = 1.5 if move.type in mon.types else 1.0
    score = move.base_power * float(move.accuracy) * stab * effectiveness
    if move.target == Target.ALL_ADJACENT:
        ally = player.active_pokemon[1 - pos]
        if ally is not None:
            score -= move.base_power * ally.damage_multiplier(move)
    return score + 0.01 * float(order.mega)


class Heuristic:
    def __init__(self, style: str, rng: np.random.Generator):
        if style not in STYLES:
            raise ValueError(f"Unknown opponent style: {style}")
        self.style = str(rng.choice(["damage", "switch"])) if style == "mixture" else style
        self.rng = rng

    def choose(self, player, legal):
        pairs = list(legal)
        if self.style == "random":
            return pairs[int(self.rng.integers(len(pairs)))]
        if self.style == "switch" and self.rng.random() < 0.35:
            switches = [
                pair
                for pair in pairs
                if any(
                    isinstance(o.order, Pokemon)
                    for o in (legal[pair].first_order, legal[pair].second_order)
                )
            ]
            if switches:
                return switches[int(self.rng.integers(len(switches)))]
        cache = {}

        def score(order, pos):
            key = (pos, order.message)
            if key not in cache:
                value = damage_score(order, player, pos)
                mon, move = player.active_pokemon[pos], order.order
                if self.style == "setup" and isinstance(move, Move) and mon is not None:
                    boosts = move.boosts or {}
                    if move.target == Target.SELF and any(
                        change > 0 and mon.boosts.get(stat, 0) < 2
                        for stat, change in boosts.items()
                    ):
                        value += 1000
                cache[key] = value
            return cache[key]

        scores = np.array(
            [score(legal[p].first_order, 0) + score(legal[p].second_order, 1) for p in pairs]
        )
        best = np.flatnonzero(scores == scores.max())
        return pairs[int(self.rng.choice(best))]
