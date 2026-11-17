"""Play one local Showdown battle against the trained Transformer and save its replay."""

import argparse
from pathlib import Path

from evaluate_counterleague import PUBLIC, read
from fixed_league import matches
from train_counterleague import TEAM


def replay(public, matchup, team, seed, side, out):
    lock = read(public / "locked.json")
    model = read(public / "candidates.json")[lock[matchup]]
    pool = read(public / "opponents.json")
    opponent_team = next(t for t in pool if Path(t).stem == team)
    own_team = TEAM if matchup == "fixed_team" else opponent_team
    offset = int(side == "p2")
    (row,) = matches(
        {},
        model,
        own_team,
        pool[opponent_team][0],
        opponent_team,
        seed - offset,
        offset,
        1,
        replay_dir=out,
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, default=PUBLIC)
    p.add_argument(
        "--matchup",
        choices=["primary", "single", "fixed_team"],
        default="fixed_team",
        help="primary and single play mirrors; fixed_team always pilots MC461",
    )
    p.add_argument("--team", choices=["MC463", "MC646", "MC751", "MC781", "MC461"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--side", choices=["p1", "p2"], default="p1")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    row = replay(a.public, a.matchup, a.team, a.seed, a.side, a.out)
    print(
        f"Challenger {row['outcome']} on turn {row['turn']}. Replay: {a.out}/{a.seed}-{a.side}.html"
    )
