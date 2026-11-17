"""Audit the slot-training experiment and report paired effects on familiar and held-out teams."""

import argparse
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import torch
from analyze_battle_actions import read
from analyze_long_rewards import load, settings, write_csv
from analyze_policy_size import locations
from analyze_rewards import records
from analyze_roster_finetuning import identical_weights, interval
from analyze_roster_transfer import OPPONENT_TEAMS, same_runtime
from evaluate_move_slots import TESTS, checkpoint, eval_path
from probe_move_order import distribution, opening
from probe_target_typing import probe_cases, score_targets
from recover_action_checkpoints import FULL, ORIGINAL
from roster_transfer import BATTLES, EVAL_SEED, ROOT, SEEDS, STYLES, split
from train_move_slots import BUDGETS, MODES, train_dir

from pokemon_rl.history import make_policy

NUMBERS = (
    "steps",
    "seed",
    "eval_seed",
    "simulator_seed",
    "opponent_team_index",
    "turns",
    "decisions",
    "better",
    "eligible",
    "ally_attacks",
    "attacks",
    "edits",
)


def audit_training(root, budgets):
    provenance, exposure = [], []
    for seed in SEEDS:
        old, _ = locations(ORIGINAL, seed, FULL)
        original_manifest = load(old.parent / "manifest.json")
        archived = records(old.parent / "episodes.jsonl")
        schedules = {}
        for mode in MODES:
            directory = train_dir(root, seed, mode)
            manifest = load(directory / "manifest.json")
            same_runtime(manifest, original_manifest)
            args = manifest["args"]
            ignore = {"run_dir", "save_at", "steps", "shuffle_move_slots"}
            assert settings(args, ignore) == settings(original_manifest["args"], ignore)
            assert args.get("shuffle_move_slots", False) == (mode == "shuffled")
            assert args["save_at"] == list(BUDGETS) and args["steps"] == BUDGETS[-1]
            assert (
                args["learner_teams"] == split()["train"]
                and args["opponent_teams"] == OPPONENT_TEAMS
            )
            assert identical_weights(directory / "checkpoint-0.pt", old.parent / "checkpoint-0.pt")
            if mode == "fixed":
                assert identical_weights(directory / f"checkpoint-{BUDGETS[0]}.pt", old)
            # At the interim checkpoint the training logs may still be growing. Consume complete rows.
            episodes = records_through(directory / "episodes.jsonl", "step", max(budgets))
            if max(budgets) == BUDGETS[-1]:
                final = load(directory / "summary.json")
                assert (final["steps"], final["episodes"]) == (BUDGETS[-1], len(episodes))
            if mode == "fixed":
                assert [r for r in episodes if r["step"] <= BUDGETS[0]] == archived
                prefix = records_through(directory / "metrics.jsonl", "steps", BUDGETS[0])
                prior = records(old.parent / "metrics.jsonl")
                assert all(
                    settings(a, {"steps_per_second"}) == settings(b, {"steps_per_second"})
                    for a, b in zip(prefix, prior, strict=True)
                )
            schedules[mode] = [
                tuple(
                    r[k]
                    for k in (
                        "episode",
                        "simulator_seed",
                        "learner_team",
                        "opponent",
                        "opponent_team_index",
                    )
                )
                for r in episodes
            ]
            for steps in budgets:
                path = checkpoint(root, seed, mode, steps)
                saved = torch.load(path, weights_only=True, map_location="cpu")
                assert (
                    saved["seed"],
                    saved["steps"],
                    saved["transformer_chunks"],
                    saved["hidden_size"],
                ) == (seed, steps, "full", 256)
                counts = Counter(r["learner_team"] for r in episodes if r["step"] <= steps)
                assert set(counts) == set(split()["train"])
                exposure.extend(
                    {
                        "seed": seed,
                        "training": mode,
                        "steps": steps,
                        "roster": Path(t).stem,
                        "training_battles": n,
                    }
                    for t, n in sorted(counts.items())
                )
                provenance.append(
                    {
                        "seed": seed,
                        "training": mode,
                        "steps": steps,
                        "training_battles": sum(counts.values()),
                        "training_commit": manifest["git_commit"],
                        "evaluated_checkpoint_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                )
        assert all(a == b for a, b in zip(schedules["fixed"], schedules["shuffled"]))
    return provenance, exposure


def records_through(path, field, limit):
    import json

    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row[field] > limit:
                break
            rows.append(row)
    return rows


def score(rows):
    wins = sum(r["outcome"] == "win" for r in rows)
    return {"battles": len(rows), "wins": wins, "win_pct": 100 * wins / len(rows)}


def get_board_probes(root, budgets):
    cases = probe_cases()
    type_rows, order_rows = [], []
    policies = {}
    for steps in budgets:
        for mode in MODES:
            for seed in SEEDS:
                cp = checkpoint(root, seed, mode, steps)
                cond = {"steps": steps, "training": mode, "seed": seed}
                type_rows.extend(score_targets(cp, cases, cond))
                saved = torch.load(cp, weights_only=True, map_location="cpu")
                model = make_policy(saved["obs_size"], saved["hidden_size"], "full")
                model.load_state_dict(saved["policy"])
                policies[steps, mode, seed] = model.eval()
    for scope, rosters in split().items():
        for roster in rosters:
            for opponent in OPPONENT_TEAMS:
                orig, legal, _ = opening(roster, opponent, False)
                permuted, new_legal, _ = opening(roster, opponent, True)
                assert set(legal.values()) == set(new_legal.values())
                for (steps, mode, seed), model in policies.items():
                    a = distribution(model, orig, legal)
                    b = distribution(model, permuted, new_legal)
                    order_rows.append(
                        {
                            "steps": steps,
                            "training": mode,
                            "seed": seed,
                            "split": scope,
                            "roster": Path(roster).stem,
                            "opponent": Path(opponent).stem,
                            "semantic_tv": sum(abs(p - b[k]) for k, p in a.items()) / 2,
                            "same_top_order": int(max(a, key=a.get) == max(b, key=b.get)),
                        }
                    )
    return type_rows, order_rows


def analyze(root, out, budgets):
    torch.set_num_threads(1)
    provenance, exposure = audit_training(root, budgets)
    probes, opening_rows = get_board_probes(root, budgets)
    data, seeded, teams, schedule = [], [], [], {}
    # Match every evaluation replicate, including its simulator seed and opponent roster.
    archive = ROOT / "results/sketch5-action-diagnosis"
    prior = {
        (int(r["seed"]), r["roster"], r["style"], int(r["eval_seed"]), r["mode"]): r
        for r in read(archive / "confirmation-games.csv.gz")
        if r["mode"] in ("rotate", "retarget")
    }
    for steps in budgets:
        for mode in MODES:
            for scope, rosters in split().items():
                for seed in SEEDS:
                    games = defaultdict(list)
                    for roster in rosters:
                        matched = {}
                        for test in TESTS:
                            rows = read(eval_path(root, seed, mode, steps, roster, test))
                            for r in rows:
                                r.update({k: int(r[k]) for k in NUMBERS})
                                assert (
                                    r["steps"],
                                    r["training"],
                                    r["split"],
                                    r["seed"],
                                    r["roster"],
                                    r["test"],
                                ) == (steps, mode, scope, seed, Path(roster).stem, test)
                                assert r["outcome"] in ("win", "loss", "tie", "truncated")
                                plan = r["simulator_seed"], r["opponent_team_index"]
                                assert schedule.setdefault(r["eval_seed"], plan) == plan
                                key = r["style"], r["eval_seed"]
                                result = r["outcome"], r["turns"], r["decisions"]
                                if test == "native":
                                    assert key not in matched
                                    matched[key] = result
                                if test == "retarget" and r["edits"] == 0:
                                    assert matched[key] == result
                                if (
                                    steps == BUDGETS[0]
                                    and mode == "fixed"
                                    and scope == "test"
                                    and test in ("rotate", "retarget")
                                ):
                                    old = prior[seed, Path(roster).stem, *key, test]
                                    assert (r["outcome"], r["turns"]) == (
                                        old["outcome"],
                                        int(old["turns"]),
                                    )
                            expected = {
                                (st, EVAL_SEED + i) for st in STYLES for i in range(BATTLES)
                            }
                            assert (
                                len(rows) == len(expected)
                                and {(r["style"], r["eval_seed"]) for r in rows} == expected
                            )
                            games[test].extend(rows)
                            teams.append(
                                {
                                    "steps": steps,
                                    "training": mode,
                                    "split": scope,
                                    "seed": seed,
                                    "roster": Path(roster).stem,
                                    "test": test,
                                    **score(rows),
                                }
                            )
                    for rows in games.values():
                        data.extend(rows)
                    native = games["native"]
                    base = score(native)
                    rotated, retarget = [score(games[t])["win_pct"] for t in ("rotate", "retarget")]
                    probed = [
                        r["type_probe_pct"]
                        for r in probes
                        if (r["steps"], r["training"], r["seed"]) == (steps, mode, seed)
                    ]
                    tvs = [
                        r["semantic_tv"]
                        for r in opening_rows
                        if (r["steps"], r["training"], r["seed"], r["split"])
                        == (steps, mode, seed, scope)
                    ]
                    seeded.append(
                        {
                            "steps": steps,
                            "training": mode,
                            "split": scope,
                            "seed": seed,
                            **base,
                            "rotated_win_pct": rotated,
                            "random_order_win_pct": score(games["random"])["win_pct"],
                            "rotation_change_pp": rotated - base["win_pct"],
                            "retarget_win_pct": retarget,
                            "retarget_gain_pp": retarget - base["win_pct"],
                            "higher_target_pct": 100
                            * sum(r["better"] for r in native)
                            / sum(r["eligible"] for r in native),
                            "ally_attack_pct": 100
                            * sum(r["ally_attacks"] for r in native)
                            / sum(r["attacks"] for r in native),
                            "opening_tv_pct": 100 * mean(tvs),
                            "type_probe_pct": mean(probed) if scope == "test" else None,
                        }
                    )
    metrics = (
        "win_pct",
        "rotated_win_pct",
        "random_order_win_pct",
        "rotation_change_pp",
        "retarget_win_pct",
        "retarget_gain_pp",
        "higher_target_pct",
        "ally_attack_pct",
        "opening_tv_pct",
        "type_probe_pct",
    )
    by_seed = {(r["steps"], r["training"], r["split"], r["seed"]): r for r in seeded}
    summary, paired = [], []
    for scope in split():
        for metric in metrics:
            if scope == "train" and metric == "type_probe_pct":
                continue
            for steps in budgets:
                for mode in MODES:
                    vals = [by_seed[steps, mode, scope, s][metric] for s in SEEDS]
                    summary.append(
                        {
                            "split": scope,
                            "steps": steps,
                            "training": mode,
                            "metric": metric,
                            **interval(vals),
                        }
                    )
            comparisons = [((st, "shuffled"), (st, "fixed")) for st in budgets]
            if len(budgets) == 2:
                comparisons += [((budgets[1], m), (budgets[0], m)) for m in MODES]
            for left, right in comparisons:
                vals = [
                    by_seed[*left, scope, s][metric] - by_seed[*right, scope, s][metric]
                    for s in SEEDS
                ]
                paired.append(
                    {
                        "split": scope,
                        "metric": metric,
                        "left_steps": left[0],
                        "left": left[1],
                        "right_steps": right[0],
                        "right": right[1],
                        **{f"seed_{s}": v for s, v in zip(SEEDS, vals, strict=True)},
                        **interval(vals),
                    }
                )
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("summary.csv", summary),
        ("paired.csv", paired),
        ("per-seed.csv", seeded),
        ("per-team-seed.csv", teams),
        ("training-exposure.csv", exposure),
        ("checkpoints.csv", provenance),
        ("type-probe.csv", probes),
        ("opening-order.csv", opening_rows),
        ("games.csv.gz", data),
    ]:
        write_csv(out / name, rows)
    by_team = defaultdict(list)
    for r in teams:
        if r["test"] == "native":
            by_team[r["split"], r["roster"], r["steps"], r["training"]].append(r)
    write_csv(
        out / "per-team.csv",
        [
            {
                "split": key[0],
                "roster": key[1],
                "steps": key[2],
                "training": key[3],
                "wins": sum(r["wins"] for r in rows),
                "games": sum(r["battles"] for r in rows),
                "win_pct": mean(r["win_pct"] for r in rows),
            }
            for key, rows in sorted(by_team.items())
        ],
    )
    if len(budgets) == 2:
        plot(out, seeded)


def plot(out, seeded):
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    panels = (
        ("train", "win_pct", "15 familiar teams", "Unaided win rate (%)"),
        ("test", "win_pct", "7 held-out teams", "Unaided win rate (%)"),
        ("test", "retarget_gain_pp", "7 held-out teams", "Gain from target rule (points)"),
    )
    for ax, (scope, metric, title, ylabel) in zip(axes, panels, strict=True):
        for mode, label, color in (
            ("fixed", "Fixed", "#17657a"),
            ("shuffled", "Shuffled", "#ad542c"),
        ):
            lines = []
            for s in SEEDS:
                rows = sorted(
                    (
                        r
                        for r in seeded
                        if (r["split"], r["seed"], r["training"]) == (scope, s, mode)
                    ),
                    key=lambda r: r["steps"],
                )
                y = [r[metric] for r in rows]
                lines.append(y)
                ax.plot([0, 1], y, color=color, alpha=0.22, lw=1)
            ax.plot(
                [0, 1],
                [mean(ys) for ys in zip(*lines, strict=True)],
                color=color,
                marker="o",
                lw=2.5,
                label=label,
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["65,536", "262,144"],
            xlabel="PPO decisions",
            ylabel=ylabel,
            title=title,
            xlim=(-0.12, 1.12),
        )
        ax.spines[["top", "right"]].set_visible(False)
    low = min(ax.get_ylim()[0] for ax in axes[:2])
    high = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(max(0, low - 1), min(100, high + 1))
    axes[0].legend(frameon=False)
    axes[2].axhline(0, color="#888", ls="--", lw=1)
    fig.savefig(out / "move-slots.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interim", action="store_true")
    a = p.parse_args()
    analyze(a.runs.resolve(), a.output.resolve(), BUDGETS[:1] if a.interim else BUDGETS)
