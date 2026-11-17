"""Compare exact opening action distributions after permuting the same four moves."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from analyze_roster_transfer import OPPONENT_TEAMS
from roster_transfer import EVAL_SEED, SEEDS, split
from trace_battle_actions import load_policy, rotate

from pokemon_rl.battle import ACTION_SIZE
from pokemon_rl.env import BattleEnv
from pokemon_rl.features import MON_FEATURES, TYPES, move_features
from pokemon_rl.history import FullBattleHistory


def opening(roster, opponent, rotated):
    base = BattleEnv(team=roster, opponent_teams=[opponent])
    if rotated:
        base.learner_teams = [rotate(t) for t in base.learner_teams]
    env = FullBattleHistory(base)
    try:
        obs, _ = env.reset(seed=EVAL_SEED)
        moves = {
            m.id: tuple(move_features(m))
            for mon in base.battle.players["p1"].team.values()
            for m in mon.moves.values()
        }
        return obs, {k: v.message for k, v in base.legal().items()}, moves
    finally:
        env.close()


@torch.no_grad()
def distribution(model, obs, labels):
    feat = model.encoder(torch.from_numpy(obs)[None])
    mask = torch.zeros((ACTION_SIZE, ACTION_SIZE), dtype=torch.bool)
    for i, j in labels:
        mask[i, j] = True
    first_ids = mask.any(1).nonzero().flatten()
    first = model.distribution(model.first(feat), mask.any(1)[None]).probs[0]
    encoded = torch.cat((feat.expand(len(first_ids), -1), model.first_embedding(first_ids)), dim=1)
    second = model.distribution(model.second(encoded), mask[first_ids]).probs
    joint = torch.zeros_like(mask, dtype=feat.dtype)
    joint[first_ids] = first[first_ids, None] * second
    probs = {name: joint[k].item() for k, name in labels.items()}
    assert abs(sum(probs.values()) - 1) < 2e-6
    return probs


def analyze(root, out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    models = {seed: load_policy(root, seed, "native")[0] for seed in SEEDS}
    rows, descriptors = [], defaultdict(set)
    for label, rosters in split().items():
        for roster in rosters:
            for opponent in OPPONENT_TEAMS:
                obs, legal, moves = opening(roster, opponent, False)
                changed, new_legal, _ = opening(roster, opponent, True)
                assert set(legal.values()) == set(new_legal.values())
                # Verify that only the four own move descriptors in each record rotated.
                expected = obs.copy()
                own = expected[0, :-2][-12 * MON_FEATURES :].reshape(12, MON_FEATURES)[:6]
                move_width = 8 + len(TYPES)
                for mon in own:
                    slots = mon[-4 * move_width :].reshape(4, move_width)
                    slots[:] = np.roll(slots, -1, axis=0)
                np.testing.assert_equal(expected, changed)
                for name, vector in moves.items():
                    descriptors[vector].add(name)
                for seed, model in models.items():
                    a, b = distribution(model, obs, legal), distribution(model, changed, new_legal)
                    rows.append(
                        {
                            "seed": seed,
                            "split": "practice" if label == "train" else "held_out",
                            "roster": Path(roster).stem,
                            "opponent": Path(opponent).stem,
                            "semantic_tv": sum(abs(p - b[k]) for k, p in a.items()) / 2,
                            "original_top": max(a, key=a.get),
                            "rotated_top": max(b, key=b.get),
                        }
                    )
    write_csv(out / "opening-move-order.csv", rows)
    write_csv(
        out / "identical-move-descriptors.csv",
        [{"moves": " | ".join(sorted(names))} for names in descriptors.values() if len(names) > 1],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    analyze(args.runs.resolve(), args.output.resolve())
