"""Compare the frozen policy interventions with all archived held-out games."""

import argparse
from pathlib import Path

from analyze_battle_actions import read
from analyze_long_rewards import write_csv
from analyze_policy_size import locations
from analyze_rewards import records
from analyze_roster_finetuning import interval
from recover_action_checkpoints import FULL, ORIGINAL
from roster_transfer import BATTLES, EVAL_SEED, SEEDS, STYLES, split
from trace_battle_actions import HISTORY_LIMITS, N

INTERVENTIONS = ("retarget", "rotate", *HISTORY_LIMITS)


def trace_rows(root, mode, seed, team, size):
    rows = read(root / "trace" / mode / f"seed-{seed}" / team / "games.csv")
    indexed = {(r["style"], int(r["eval_seed"])): r for r in rows}
    assert len(rows) == len(indexed) == len(STYLES) * size
    return indexed


def compare(root, out):
    games = []
    for seed in SEEDS:
        _, archive = locations(ORIGINAL, seed, FULL)
        for roster in split()["test"]:
            team = Path(roster).stem
            baseline = trace_rows(root, "native", seed, team, N)
            for mode in INTERVENTIONS:
                first = trace_rows(root, mode, seed, team, N) if mode not in HISTORY_LIMITS else {}
                full = trace_rows(root / "confirm", mode, seed, team, BATTLES)
                for style in STYLES:
                    original = records(archive / team / "games" / f"{style}.jsonl")
                    assert len(original) == BATTLES
                    for i, old in enumerate(original):
                        es = EVAL_SEED + i
                        key = (style, es)
                        r = full[key]
                        assert (old["seed"], old["learner_team"], old["opponent"]) == (
                            es,
                            roster,
                            style,
                        )
                        assert int(r["simulator_seed"]) == old["simulator_seed"]
                        assert int(r["opponent_team_index"]) == old["opponent_team_index"]
                        original_result = (old["outcome"], old["turn"], old["learner_decisions"])
                        result = (r["outcome"], int(r["turns"]), int(r["decisions"]))
                        if i < N:
                            if first:
                                assert first[key] == r
                            n = baseline[key]
                            assert original_result == (
                                n["outcome"],
                                int(n["turns"]),
                                int(n["decisions"]),
                            )
                        if mode == "retarget" and int(r["edits"]) == 0:
                            assert result == original_result
                        if old["learner_decisions"] <= HISTORY_LIMITS.get(mode, 0):
                            assert result == original_result
                        games.append(
                            {
                                "seed": seed,
                                "roster": team,
                                "style": style,
                                "eval_seed": es,
                                "simulator_seed": old["simulator_seed"],
                                "opponent_team_index": old["opponent_team_index"],
                                "mode": mode,
                                "native_outcome": old["outcome"],
                                "outcome": r["outcome"],
                                "native_turns": old["turn"],
                                "native_decisions": old["learner_decisions"],
                                "turns": int(r["turns"]),
                                "target_edits": int(r["edits"]),
                            }
                        )
    summary = []
    for mode in INTERVENTIONS:
        for window, start, end in (
            ("first25", 0, N),
            ("remaining75", N, BATTLES),
            ("all100", 0, BATTLES),
        ):
            rows = [
                r for r in games if r["mode"] == mode and start <= r["eval_seed"] - EVAL_SEED < end
            ]
            assert len(rows) == len(SEEDS) * len(split()["test"]) * len(STYLES) * (end - start)
            assert {r[k] for r in rows for k in ("outcome", "native_outcome")} <= {"win", "loss"}
            original = sum(r["native_outcome"] == "win" for r in rows)
            wins = sum(r["outcome"] == "win" for r in rows)
            diffs = {}
            for s in SEEDS:
                group = [r for r in rows if r["seed"] == s]
                d = sum((r["outcome"] == "win") - (r["native_outcome"] == "win") for r in group)
                diffs[f"seed_{s}_pp"] = 100 * d / len(group)
            summary.append(
                {
                    "mode": mode,
                    "window": window,
                    "battles": len(rows),
                    "native_wins": original,
                    "wins": wins,
                    "native_win_pct": 100 * original / len(rows),
                    "win_pct": 100 * wins / len(rows),
                    "loss_to_win": sum(
                        r["native_outcome"] == "loss" and r["outcome"] == "win" for r in rows
                    ),
                    "win_to_loss": sum(
                        r["native_outcome"] == "win" and r["outcome"] == "loss" for r in rows
                    ),
                    "games_retargeted": sum(r["target_edits"] > 0 for r in rows),
                    **diffs,
                    **interval(list(diffs.values())),
                }
            )
    write_csv(out / "confirmation.csv", summary)
    write_csv(out / "confirmation-games.csv.gz", games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.runs.resolve(), args.output.resolve())
