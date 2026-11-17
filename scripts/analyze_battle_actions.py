"""Aggregate matched replay diagnostics and retain the examples discussed in the report."""

import argparse
import csv
import gzip
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean

from analyze_long_rewards import write_csv
from analyze_roster_finetuning import interval
from roster_transfer import EVAL_SEED, SEEDS, STYLES, split
from trace_battle_actions import MODES, N


def read(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as f:
        return list(csv.DictReader(f))


def paired(left, right):
    values = [a - b for a, b in zip(left, right, strict=True)]
    return {**{f"seed_{s}": v for s, v in zip(SEEDS, values, strict=True)}, **interval(values)}


def analyze(root, out):
    trace = root / "trace"
    games = [r for p in sorted(trace.glob("*/*/*/games.csv")) for r in read(p)]
    actions = [r for p in sorted(trace.glob("*/*/*/actions.csv.gz")) for r in read(p)]
    scopes = {"held_out": split()["test"], "practice": split()["train"]}
    rates, summaries, per_seed, deltas = {}, [], [], []
    schedule = {}
    for mode in MODES:
        for scope, teams in scopes.items():
            if scope == "practice" and mode in ("small", "retarget"):
                continue
            subset = [r for r in games if r["mode"] == mode and r["split"] == scope]
            expected = {
                (str(s), Path(t).stem, opp, str(es))
                for s in SEEDS
                for t in teams
                for opp in STYLES
                for es in range(EVAL_SEED, EVAL_SEED + N)
            }
            batch = {}
            for r in subset:
                key = tuple(r[k] for k in ("seed", "roster", "style", "eval_seed"))
                assert key not in batch
                batch[key] = r
                plan = (r["simulator_seed"], r["opponent_team_index"])
                assert schedule.setdefault(key, plan) == plan
            assert set(batch) == expected
            vals = []
            for s in SEEDS:
                seeded = [r for r in subset if int(r["seed"]) == s]
                wins = sum(r["outcome"] == "win" for r in seeded)
                pct = 100 * wins / len(seeded)
                vals.append(pct)
                per_seed.append(
                    {
                        "mode": mode,
                        "split": scope,
                        "seed": s,
                        "wins": wins,
                        "battles": len(seeded),
                        "win_pct": pct,
                    }
                )
            rates[mode, scope] = vals
            summaries.append(
                {
                    "mode": mode,
                    "split": scope,
                    "wins": sum(r["outcome"] == "win" for r in subset),
                    "battles": len(subset),
                    "win_pct": mean(vals),
                }
            )
    for scope in scopes:
        pairs = [("32k", "16k"), ("native", "32k"), ("native", "16k"), ("rotate", "native")]
        if scope == "held_out":
            pairs += [("retarget", "native"), ("small", "native")]
        for left, right in pairs:
            deltas.append(
                {
                    "split": scope,
                    "left": left,
                    "right": right,
                    **paired(rates[left, scope], rates[right, scope]),
                }
            )

    def game_key(r):
        return tuple(r[k] for k in ("seed", "roster", "style", "eval_seed"))

    native = {game_key(r): r for r in games if r["mode"] == "native" and r["split"] == "held_out"}
    transitions = Counter()
    edited = Counter()
    for r in games:
        if r["mode"] == "retarget":
            orig = native[game_key(r)]
            transitions[orig["outcome"], r["outcome"]] += 1
            edited["games_with_edits"] += int(r["edits"]) > 0
            edited["slot_edits"] += int(r["edits"])
            if int(r["edits"]) == 0:
                assert (orig["outcome"], orig["turns"], orig["decisions"]) == (
                    r["outcome"],
                    r["turns"],
                    r["decisions"],
                )
    counts = []
    for mode in ("native", "small"):
        subset = [r for r in actions if r["mode"] == mode]
        g = [r for r in games if r["mode"] == mode and r["split"] == "held_out"]
        kinds = Counter(r["kind"] for r in subset)
        assert sum(kinds.values()) == 2 * sum(int(r["decisions"]) for r in g)
        eligible = [
            r
            for r in subset
            if r["other_multiplier"] and float(r["other_multiplier"]) != float(r["multiplier"])
        ]
        counts.append(
            {
                "mode": mode,
                **{k: kinds[k] for k in ("attack", "status", "switch", "forced_switch", "pass")},
                "explicit_ally_attacks": sum(int(r["ally_attack"]) for r in subset),
                "target_comparisons": len(eligible),
                "lower_multiplier": sum(
                    float(r["multiplier"]) < float(r["other_multiplier"]) for r in eligible
                ),
                **{
                    k: sum(int(r[k]) for r in g)
                    for k in (
                        "executed_moves",
                        "protect_uses",
                        "protect_fails",
                        "fake_out_uses",
                        "fake_out_fails",
                    )
                },
            }
        )
    probe = read(root / "order-probe/opening-move-order.csv")
    effect = []
    for scope, teams in scopes.items():
        rows = [r for r in probe if r["split"] == scope]
        assert len(rows) == len(teams) * 2 * len(SEEDS)
        effect.append(
            {
                "split": scope,
                "openings": len(rows),
                "changed_top_joint_order": sum(r["original_top"] != r["rotated_top"] for r in rows),
                "mean_semantic_tv": mean(float(r["semantic_tv"]) for r in rows),
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("win-rates.csv", summaries),
        ("per-seed.csv", per_seed),
        ("paired-differences.csv", deltas),
        ("action-counts.csv", counts),
        ("games.csv.gz", games),
        ("actions.csv.gz", actions),
        ("opening-sensitivity.csv", effect),
        (
            "retarget-outcomes.csv",
            [
                {"native": k[0], "retarget": k[1], "games": v}
                for k, v in sorted(transitions.items())
            ],
        ),
        ("target-edits.csv", [dict(edited)]),
        (
            "recovered-checkpoints.csv",
            [
                {
                    "seed": s,
                    "steps": step,
                    "sha256": hashlib.sha256(
                        (root / f"early/seed-{s}/train/checkpoint-{step}.pt").read_bytes()
                    ).hexdigest(),
                }
                for s in SEEDS
                for step in (16384, 32768)
            ],
        ),
    ]:
        write_csv(out / name, rows)
    for name in ("opening-move-order.csv", "identical-move-descriptors.csv"):
        shutil.copyfile(root / "order-probe" / name, out / name)
    replays = out / "replays"
    replays.mkdir()
    for team, style, es, models in [
        ("MC646", "switch", 610004, ("native", "small")),
        ("MC751", "switch", 610004, ("native", "small")),
        ("MC403", "damage", 610004, ("native", "retarget")),
        ("MC462", "switch", 610005, ("native", "retarget")),
    ]:
        for m in models:
            stem = f"{style}-{es}"
            shutil.copyfile(
                trace / m / "seed-2" / team / f"{stem}.log", replays / f"{team}-s2-{stem}-{m}.log"
            )
    plot(out, rates)


def plot(out, rates):
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
    for scope, label, color in (
        ("practice", "15 practice teams", "#19738a"),
        ("held_out", "7 held-out teams", "#aa552d"),
    ):
        cells = [rates[m, scope] for m in ("16k", "32k", "native")]
        for y in zip(*cells, strict=True):
            a.plot([16, 32, 64], y, color=color, alpha=0.19, lw=1)
        a.plot([16, 32, 64], [mean(x) for x in cells], color=color, marker="o", label=label)
    a.set(
        xticks=[16, 32, 64],
        xticklabels=["16,384", "32,768", "65,536"],
        xlabel="PPO decisions",
        ylabel="Win rate (%)",
        title="Frozen policies on matched battles",
    )
    a.legend(frameon=False)
    for condition, scope, label, color in [
        ("rotate", "practice", "Move rotation · practice", "#19738a"),
        ("rotate", "held_out", "Move rotation · held-out", "#aa552d"),
        ("retarget", "held_out", "Target rule · held-out", "#aa552d"),
    ]:
        ds = [x - y for x, y in zip(rates[condition, scope], rates["native", scope], strict=True)]
        b.scatter(ds, [label] * 4, color=color, alpha=0.4, s=23)
        b.scatter([mean(ds)], [label], color=color, marker="D", s=45)
    b.axvline(0, color="#666", ls="--", lw=1)
    b.set(xlabel="Win-rate change (percentage points)", title="Interventions at 65,536 decisions")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "diagnosis.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    analyze(a.runs.resolve(), a.output.resolve())
