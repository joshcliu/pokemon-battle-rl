import numpy as np
import pytest

from pokemon_rl.battle import ROOT, LocalBattle
from pokemon_rl.env import BattleEnv
from pokemon_rl.features import observation
from pokemon_rl.opponents import Heuristic

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ROOT / ".cache/pokemon-showdown/dist/sim/index.js").exists(),
        reason="Run scripts/setup_simulator.py first",
    ),
]


def play(seed):
    env = BattleEnv(opponent="mixture")
    try:
        assert env.observation_space is not None
        obs, info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        agent = Heuristic("random", np.random.default_rng(seed))
        states = [obs]
        for _ in range(600):
            mask = env.joint_mask()
            for a1, a2 in env.legal():
                assert mask[a1, a2]
                assert not (1 <= a1 <= 6 and a1 == a2)  # Cannot share a switch slot.
                assert not (27 <= a1 <= 46 and 27 <= a2 <= 46)  # One Mega per team.
            action = agent.choose(env.battle.players["p1"], env.legal())
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            states.append(obs)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert reward in (-1, 0, 1)
        log = [line for line in env.battle.log if not line.startswith("|t:|")]
        return info, states, log
    finally:
        env.close()


def test_seed_reproduces_battle_and_actions():
    first, second = play(31), play(31)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_unrevealed_opponent_moves_do_not_change_learner_observation():
    own = (ROOT / "teams/sun.txt").read_text()
    other = (ROOT / "teams/balance.txt").read_text()
    with LocalBattle() as battle:
        battle.reset(7, own, other)
        obs = observation(battle.players["p1"])
        legal = set(battle.legal_actions("p1"))
        battle.reset(7, own, other.replace("- Waterfall", "- Aqua Tail"))
        np.testing.assert_array_equal(obs, observation(battle.players["p1"]))
        assert legal == set(battle.legal_actions("p1"))
        assert "aquatail" in battle.players["p2"].active_pokemon[0].moves
        assert "aquatail" not in battle.players["p1"].opponent_active_pokemon[0].moves


def test_invalid_action_does_not_advance_simulator():
    env = BattleEnv()
    try:
        env.reset(seed=0)
        before = list(env.battle.log)
        for pair in [(106, 106), (1.5, 2), (-2, -2), (1,)]:
            with pytest.raises(ValueError):
                env.step(pair)
        assert env.battle.log == before
    finally:
        env.close()


def test_time_limit_is_truncation_without_declared_result():
    env = BattleEnv(max_turns=1)
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(next(iter(env.legal())))
        assert truncated and not terminated and reward == 0
        assert info["outcome"] == "truncated"
        assert not env.battle.ended
    finally:
        env.close()


def test_opponent_styles_keep_matched_team_and_seed_schedules():
    schedules = []
    for style in ("damage", "switch", "mixture"):
        env = BattleEnv(opponent=style)
        try:
            schedule = []
            for i in range(4):
                _, info = env.reset(seed=12 if i == 0 else None)
                schedule.append((info["simulator_seed"], info["opponent_team_index"]))
                env.step(next(iter(env.legal())))
            schedules.append(schedule)
        finally:
            env.close()
    assert schedules[0] == schedules[1] == schedules[2]
