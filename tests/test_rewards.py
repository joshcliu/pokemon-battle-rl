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
