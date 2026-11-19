import pytest

from pokemon_rl.rewards import PublicHP


def test_damage_and_hp_advantage_are_distinct():
    tracker = PublicHP()
    tracker.update(
        ["|switch|p1a: A|A|100/100", "|switch|p2a: B|B|100/100", "|switch|p2b: C|C|100/100"]
    )
    damage = tracker.update(
        [
            "|move|p1a: A|Rock Slide|p2a: B",
            "|-damage|p2a: B|50/100g",
            "|-damage|p2b: C|75/100",
            "|-damage|p1a: A|60/100|[from] ability: Rough Skin",
            "|-damage|p2a: B|40/100|[from] brn",
            "|move|p2b: C|Earthquake|p1a: A",
            "|-damage|p2a: B|35/100",
            "|switch|p2a: D|D|100/100",
            "|switch|p2b: B|B|35/100",
        ]
    )
    assert damage == pytest.approx(0.75)
    assert tracker.potential() == pytest.approx((0.65 + 0.25 - 0.4) / 4)


def test_preview_slots_and_own_faints_are_not_opposing_knockouts():
    from types import SimpleNamespace

    from pokemon_rl.rewards import public_board

    def mon(hp, *, fainted=False, seen=True):
        return SimpleNamespace(current_hp_fraction=hp, max_hp=100 if seen else 0, fainted=fainted)

    # Slot 5 was on the open sheet but has never appeared. The foe took chip damage.
    team = {"a": mon(0, fainted=True), "e": mon(0, seen=False)}
    foe = {"a": mon(0.5), "b": mon(0, fainted=True), "e": mon(0, seen=False)}
    assert public_board(SimpleNamespace(team=team, opponent_team=foe)) == (0.125, 0.25)
    foe["a"] = mon(0, fainted=True)
    assert public_board(SimpleNamespace(team=team, opponent_team=foe)) == (0.25, 0.5)


@pytest.mark.parametrize("hp_weight,ko_weight", [(0, 0), (0.25, 0), (0, 0.25), (0.25, 0.25)])
def test_factorial_shaping_preserves_discounted_terminal_outcome(hp_weight, ko_weight):
    from pokemon_rl.rewards import shaped_outcome

    trajectory = [(0, 0), (0.125, 0), (0.25, 0.25), (-0.25, 0.75)]
    total = 0
    for i, (before, after) in enumerate(zip(trajectory, trajectory[1:])):
        terminal = i == 2
        r, hp, ko = shaped_outcome(
            -1 if terminal else 0,
            before,
            after,
            terminal=terminal,
            gamma=0.99,
            hp_weight=hp_weight,
            ko_weight=ko_weight,
        )
        if i == 0:
            assert hp == pytest.approx(hp_weight * 0.99 * 0.125) and ko == 0
        if i == 1:
            assert ko == pytest.approx(ko_weight * 0.99 * 0.25)
        total += 0.99**i * r
    assert total == pytest.approx(-(0.99**2))
