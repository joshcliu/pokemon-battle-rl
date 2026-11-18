import json
from types import SimpleNamespace

import numpy as np
import torch
from poke_env.battle import Move

from pokemon_rl.history import make_policy
from pokemon_rl.league import FrozenActor, LeadRouter
from pokemon_rl.opponents import Heuristic


def test_setup_bot_accepts_forced_recharge_with_no_target():
    recharge = Move("recharge", gen=9)
    assert recharge.target is None
    order = SimpleNamespace(order=recharge, message="move recharge")
    player = SimpleNamespace(active_pokemon=[object(), object()])
    legal = {(7, 7): SimpleNamespace(first_order=order, second_order=order)}
    assert Heuristic("setup", np.random.default_rng(1)).choose(player, legal) == (7, 7)


def test_loading_frozen_opponent_preserves_training_rng(tmp_path):
    model = make_policy(7, 8)
    cp = tmp_path / "opponent.pt"
    torch.save({"obs_size": 7, "hidden_size": 8, "policy": model.state_dict()}, cp)
    state = torch.get_rng_state()
    FrozenActor(cp)
    assert torch.equal(state, torch.get_rng_state())


def test_router_uses_only_visible_opening_and_resets_between_battles(tmp_path, monkeypatch):
    class Expert:
        def __init__(self, path):
            self.label = path.name

        def reset(self):
            self.decisions = 0

        def choose(self, player, legal):
            self.decisions += 1
            return self.label, self.decisions

    monkeypatch.setattr("pokemon_rl.league.FrozenActor", Expert)
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps({"routes": {"gardevoir+gyarados": "specialist.pt"}, "default": "fallback.pt"})
    )
    router = LeadRouter(path)
    player = SimpleNamespace(
        opponent_active_pokemon=[SimpleNamespace(species=s) for s in ("gyarados", "gardevoir")]
    )
    assert router.choose(player, {}) == ("specialist.pt", 1)
    player.opponent_active_pokemon = [SimpleNamespace(species="charizard"), None]
    assert router.choose(player, {}) == ("specialist.pt", 2)
    router.reset()
    assert router.choose(player, {}) == ("fallback.pt", 1)
