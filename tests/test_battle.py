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


def test_preview_picks_both_sides_before_leads_are_revealed():
    team = (ROOT / "teams/reg-mc-regionals/MC461.txt").read_text()
    views = []

    def pick(visible):
        views.append(visible)
        return (5, 6, 1, 3)

    with LocalBattle() as battle:
        battle.reset(77, team, team, preview_pickers={"p1": pick, "p2": pick})
        assert len(views) == 2 and views[0] == views[1] == tuple(sorted(views[0]))
        assert len(views[0]) == 6 and "Kingambit" in views[0]
        assert battle.input_log[1]["choices"] == {"p1": "team 5613", "p2": "team 5613"}
        for side in ("p1", "p2"):
            leads = [m.species for m in battle.players[side].active_pokemon]
            assert leads[0] == "kingambit" and leads[1].startswith("charizard")
            chosen = [m["ident"].split(": ")[1] for m in battle.requests[side]["side"]["pokemon"]]
            assert len(chosen) == 4 and chosen[-2:] == ["Meowstic-F", "Indeedee"]
        with pytest.raises(ValueError, match="Invalid team preview order"):
            battle.reset(77, team, team, preview_pickers={"p1": lambda _: (1, 1, 2, 3)})


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


def test_changing_learner_pool_preserves_opponent_schedule_and_uses_selected_team():
    pool = ["teams/sun.txt", "teams/balance.txt"]
    leads = {"teams/sun.txt": "charizard", "teams/balance.txt": "gyarados"}
    fixed = BattleEnv(opponent="mixture")
    varied = BattleEnv(opponent="mixture", learner_teams=pool)
    selected = set()
    try:
        for seed in range(12):
            _, old = fixed.reset(seed=seed)
            _, new = varied.reset(seed=seed)
            for field in ("simulator_seed", "opponent", "opponent_team_index"):
                assert old[field] == new[field]
            selected.add(new["learner_team"])
            assert (
                varied.battle.players["p1"].active_pokemon[0].species == leads[new["learner_team"]]
            )
        assert selected == set(pool)
    finally:
        fixed.close()
        varied.close()


def test_mirror_env_assigns_the_same_sampled_team_to_each_player():
    rosters = ["teams/balance.txt", "teams/sun.txt"]
    env = BattleEnv(learner_teams=rosters, opponent_teams=rosters, mirror_teams=True)
    try:
        seen = set()
        for seed in range(8):
            _, info = env.reset(seed=seed)
            assert info["learner_team"] == rosters[info["opponent_team_index"]]
            p1, p2 = (env.battle.players[s] for s in ("p1", "p2"))
            assert [p.species for p in p1.team.values()] == [p.species for p in p2.team.values()]
            seen.add(info["learner_team"])
        assert seen == set(rosters)
    finally:
        env.close()


def test_shuffle_only_changes_learner_move_positions_and_preserves_episode_schedule():
    pool = ["teams/balance.txt", "teams/sun.txt"]
    fixed = BattleEnv(learner_teams=pool, opponent="mixture")
    shuffled = BattleEnv(learner_teams=pool, opponent="mixture", shuffle_move_slots=True)
    try:
        layouts = []
        for seed in (12, None, None, 12):
            _, a = fixed.reset(seed=seed)
            _, b = shuffled.reset(seed=seed)
            assert a == b
            assert {o.message for o in fixed.legal().values()} == {
                o.message for o in shuffled.legal().values()
            }
            for side in ("p1", "p2"):
                before = [
                    (m.species, list(m.moves)) for m in fixed.battle.players[side].team.values()
                ]
                after = [
                    (m.species, list(m.moves)) for m in shuffled.battle.players[side].team.values()
                ]
                if side == "p2":
                    assert before == after
                else:
                    assert before != after
                    assert [(s, sorted(ms)) for s, ms in before] == [
                        (s, sorted(ms)) for s, ms in after
                    ]
                    layouts.append(after)
        assert layouts[0] == layouts[-1]
    finally:
        fixed.close()
        shuffled.close()


def test_hidden_shadow_tag_reissues_only_trapped_players_choice():
    env = BattleEnv(team="teams/reg-mc-regionals/MC407.txt")
    try:
        env.reset(seed=175004)
        battle = env.battle
        battle.step(
            {
                "p1": "move sludgebomb mega 2, move uturn 1",
                "p2": "switch Tyranitar, move psychic -1",
            }
        )
        battle.step({"p1": "pass, switch MOCHI"})
        battle.step({"p2": "pass, switch Arcanine"})
        assert battle.requests["p2"]["active"][1]["maybeTrapped"]
        turn = battle.turn
        battle.step(
            {"p1": "switch FLOW, move throatchop 2", "p2": "move crunch mega -2, switch Gyarados"}
        )
        assert battle.turn == turn and not battle.needs_action("p1")
        assert battle.requests["p2"]["active"][1]["trapped"]
        battle.step({"p2": "move rockslide mega, move protect"})
        assert battle.ended or battle.needs_action("p1") or battle.needs_action("p2")
        assert any(line.startswith("|switch|p1a: FLOW|") for line in battle.log)
    finally:
        env.close()
