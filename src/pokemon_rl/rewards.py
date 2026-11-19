"""Battle rewards from visible HP changes and episode outcomes."""

from .env import BattleEnv


def fraction(text):
    hp = text.split()[0]
    if hp == "0":
        return 0.0
    left, right = hp.split("/")
    return int(left) / int(right.rstrip("gyr"))


class PublicHP:
    """The two starter rosters have unique nicknames on each side."""

    def __init__(self):
        self.hp = {}
        self.move_side = None

    def update(self, lines):
        direct = 0.0
        for line in lines:
            parts = line.split("|")
            kind = parts[1] if len(parts) > 1 else ""
            if kind == "move":
                self.move_side = parts[2][:2]
            elif kind in {"turn", "upkeep", ""}:
                self.move_side = None
            elif kind in {"switch", "drag", "-damage", "-heal"}:
                side, name = parts[2][:2], parts[2].split(": ", 1)[1]
                key = (side, name)
                new = fraction(parts[4] if kind in {"switch", "drag"} else parts[3])
                if kind == "-damage" and self.move_side == "p1" and side == "p2":
                    if not any(field.startswith("[from]") for field in parts[4:]):
                        direct += max(self.hp[key] - new, 0.0)
                self.hp[key] = new
        return direct

    def potential(self):
        # Four selected per side. Unseen Pokemon count as full health.
        return sum((hp - 1) * (1 if side == "p1" else -1) for (side, _), hp in self.hp.items()) / 4


class RewardEnv(BattleEnv):
    def __init__(self, *, mode, gamma, **kwargs):
        super().__init__(**kwargs)
        self.mode, self.gamma = mode, gamma

    def reset(self, **kwargs):
        initial = super().reset(**kwargs)
        self.tracker = PublicHP()
        self.tracker.update(self.battle.log)
        self.total_reward = 0.0
        return initial

    def step(self, action):
        n = len(self.battle.log)
        before = self.tracker.potential()
        obs, outcome_reward, terminated, truncated, info = super().step(action)
        damage = self.tracker.update(self.battle.log[n:])
        after = 0.0 if terminated else self.tracker.potential()
        reward = (
            damage if self.mode == "damage-only" else (outcome_reward + self.gamma * after - before)
        )
        self.total_reward += reward
        if terminated or truncated:
            info["episode_training_return_undiscounted"] = self.total_reward
        return obs, reward, terminated, truncated, info


def health_deficit(mon):
    if mon.fainted:
        return 1.0
    # Team sheets can include a Pokémon that has never entered play.
    return 1.0 - mon.current_hp_fraction if mon.max_hp else 0.0


def public_board(view):
    """Net HP advantage and fraction of opposing Pokémon fainted, from the acting side."""
    own = view.team.values()
    foes = view.opponent_team.values()
    own_damage = sum(map(health_deficit, own))
    foe_damage = sum(map(health_deficit, foes))
    knocked_out = sum(m.fainted for m in foes)
    assert 0 <= own_damage <= 4.00001 and 0 <= foe_damage <= 4.00001 and 0 <= knocked_out <= 4
    return (foe_damage - own_damage) / 4, knocked_out / 4


def shaped_outcome(outcome, before, after, *, terminal, gamma, hp_weight, ko_weight):
    """Redistribute discounted outcome credit with two zero-at-terminal potentials."""
    next_hp, next_ko = (0.0, 0.0) if terminal else after
    hp = hp_weight * (gamma * next_hp - before[0])
    ko = ko_weight * (gamma * next_ko - before[1])
    return outcome + hp + ko, hp, ko
