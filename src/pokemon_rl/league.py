"""Frozen opponent populations. Every actor receives only its own player view."""

import json
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import Field, Move, Pokemon, PokemonType, SideCondition, Weather

from .battle import ACTION_SIZE, ROOT
from .features import observation
from .history import make_policy
from .opponents import Heuristic


def stat(mon, key):
    value = mon.stats.get(key) or (mon.base_stats[key] + (77 if key == "hp" else 23))
    stage = mon.boosts.get(key, 0)
    return value * (2 + max(0, stage)) / (2 + max(0, -stage))


class Tactical:
    """Estimate damage per foe, reward knockouts, and coordinate the two commands."""

    style = "tactical"

    def __init__(self, rng):
        self.rng = rng

    def choose(self, player, legal):
        foes = player.opponent_active_pokemon
        hp = [p.current_hp_fraction if p else 0 for p in foes]
        own = player.active_pokemon
        sun = Weather.SUNNYDAY in player.weather
        rain = Weather.RAINDANCE in player.weather
        psychic = Field.PSYCHIC_TERRAIN in player.fields
        grassy = Field.GRASSY_TERRAIN in player.fields

        def command(order, slot):
            mon, move = own[slot], order.order
            if isinstance(move, Pokemon):
                score = sum(
                    max(foe.damage_multiplier(t) for t in move.types)
                    - max(move.damage_multiplier(t) for t in foe.types)
                    for foe in foes
                    if foe and not foe.fainted
                )
                return (
                    0.0,
                    0.0,
                    0.025 * score - (0.75 if mon and not player.force_switch[slot] else 0),
                )
            if not mon or not isinstance(move, Move):
                return (0.0, 0.0, 0.0)
            if move.id == "tailwind":
                return (
                    0.0,
                    0.0,
                    0.78 if SideCondition.TAILWIND not in player.side_conditions else -1.0,
                )
            if move.base_power == 0:
                setup = move.id in {"dragondance", "swordsdance", "calmmind", "clangoroussoul"}
                value = (
                    0.65
                    if setup and mon.current_hp_fraction >= 0.8 and max(mon.boosts.values()) < 1
                    else -0.5
                )
                return (0.0, 0.0, value + 0.015 * order.mega)
            if order.move_target < 0:
                return (0.0, 0.0, -4.0)
            kind, power = move.type, move.base_power
            if move.id == "weatherball" and (sun or rain):
                kind = PokemonType.FIRE if sun else PokemonType.WATER
                power = 100
            physical = move.category.name == "PHYSICAL"
            power *= stat(mon, "atk" if physical else "spa") * move.expected_hits
            power *= 1.5 if kind in mon.types else 1.0
            if (kind == PokemonType.FIRE and sun) or (kind == PokemonType.WATER and rain):
                power *= 1.5
            if (kind == PokemonType.FIRE and rain) or (kind == PokemonType.WATER and sun):
                power *= 0.5
            if psychic and kind == PokemonType.PSYCHIC:
                power *= 1.3 * (1.5 if move.id == "expandingforce" else 1.0)
            if grassy and kind == PokemonType.GRASS:
                power *= 1.3
            spread = move.target.name in ("ALL_ADJACENT", "ALL_ADJACENT_FOES") or (
                psychic and move.id == "expandingforce"
            )
            priority = move.priority > 0 or (grassy and move.id == "grassyglide")
            dmg = [0.0, 0.0]
            for i, foe in enumerate(foes):
                if not foe or foe.fainted or (not spread and order.move_target not in (0, i + 1)):
                    continue
                immune = (kind == PokemonType.GROUND and foe.ability == "levitate") or (
                    priority
                    and psychic
                    and PokemonType.FLYING not in foe.types
                    and foe.ability != "levitate"
                )
                if immune or (move.id == "fakeout" and not mon.first_turn):
                    continue
                mult = foe.damage_multiplier(kind)
                amount = (
                    (0.44 * power / stat(foe, "def" if physical else "spd") + 2)
                    * mult
                    / stat(foe, "hp")
                )
                dmg[i] = amount * float(move.accuracy) * (0.75 if spread and all(hp) else 1.0)
            bias = 0.04 * priority + 0.055 * order.mega
            if move.id == "fakeout" and max(dmg) > 0:
                bias += 0.48
            if move.id == "solarbeam" and not sun:
                dmg = [x * 0.46 for x in dmg]
            if move.target.name == "ALL_ADJACENT" and own[1 - slot]:
                bias -= 0.28 * own[1 - slot].damage_multiplier(kind)
            return (*dmg, bias)

        cache = {}
        scores = []
        for pair, joint in legal.items():
            for slot, order in enumerate((joint.first_order, joint.second_order)):
                key = slot, pair[slot]
                if key not in cache:
                    cache[key] = command(order, slot)
            a, b = cache[0, pair[0]], cache[1, pair[1]]
            value = a[2] + b[2]
            for i in (0, 1):
                hit = a[i] + b[i]
                value += (
                    min(hp[i], hit) + 0.12 * min(hit, 1.5) + 0.45 * (hp[i] > 0 and hit >= hp[i])
                )
            scores.append(value)
        best = np.flatnonzero(np.isclose(scores, max(scores), atol=1e-8, rtol=0))
        return list(legal)[int(self.rng.choice(best))]


class FrozenActor:
    def __init__(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.style = str(path)
        self.full = saved.get("transformer_chunks")
        if self.full not in (None, "full"):
            raise ValueError("League supports feedforward or full-history policies")
        with torch.random.fork_rng(devices=[]):
            self.model = make_policy(saved["obs_size"], saved["hidden_size"], self.full)
        self.model.load_state_dict(saved["policy"])
        self.model.eval()
        self.reset()

    def reset(self):
        self.prefix = []
        self.previous = (ACTION_SIZE, ACTION_SIZE)

    @torch.inference_mode()
    def choose(self, player, legal):
        obs = observation(player)
        if self.full:
            self.prefix.append(np.concatenate((obs, np.asarray(self.previous, dtype=np.float32))))
            obs = np.stack(self.prefix)
        mask = torch.zeros((ACTION_SIZE, ACTION_SIZE), dtype=torch.bool)
        pairs = np.asarray(list(legal))
        mask[pairs[:, 0], pairs[:, 1]] = True
        action = self.model.sample(torch.from_numpy(obs)[None], mask[None], deterministic=True)[0]
        self.previous = tuple(action[0].tolist())
        return self.previous


class OpponentPool:
    def __init__(self, path, teams):
        self.spec = json.loads(Path(path).read_text())
        if set(teams) != set(self.spec) or any(not options for options in self.spec.values()):
            raise ValueError("Pool must specify at least one actor for every opponent roster")
        self.models = {}

    def draw(self, team, rng):
        choices = self.spec[team]
        key = choices[int(rng.integers(len(choices)))]
        if key == "tactical":
            return Tactical(rng)
        if key.endswith(".pt"):
            if key not in self.models:
                self.models[key] = FrozenActor(ROOT / key)
            actor = self.models[key]
            actor.reset()
            return actor
        return Heuristic(key, rng)
