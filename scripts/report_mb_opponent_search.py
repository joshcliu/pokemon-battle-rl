"""Report paired development gains and independent confirmation for the public opponent models."""

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from confirm_mb_opponent_search import FIRST, LOCK, OLD, OUT, SEEDS, SPEC, dev, read
from report_mb_policy_iteration import interval
from vgc_posttrain import atomic_rows


def pct(x):
    return f"{x:.2f}%"


def names(files):
    return ", ".join(Path(p).stem for p in files)


def main():
    incumbent = {}
    summary = []
    files = sorted((OUT / "scores").glob("validation-*.csv"))
    modes = sorted({p.stem[len("validation-") :].rsplit("-", 1)[0] for p in files})
    for mode in [OLD, *modes]:
        folder = FIRST if mode == OLD else OUT
        if any(not (folder / "scores" / f"validation-{mode}-{s}.csv").exists() for s in SEEDS):
            continue
        if mode != OLD:
            dev(mode)
        x = np.full((4, 4, 8, 2, 8), np.nan)
        for si, s in enumerate(SEEDS):
            for r in read(folder / "scores" / f"validation-{mode}-{s}.csv"):
                idx = (si, *(int(r[k]) for k in ("i", "j", "direction", "k")))
                assert np.isnan(x[idx])
                x[idx] = int(r["reward"] == "1")
        assert np.isfinite(x).all()
        avg = x.mean(3)
        if mode == OLD:
            incumbent = avg
        delta = avg - incumbent
        lo, hi = interval(avg)
        dl, dh = interval(delta)
        summary.append(
            dict(
                method=mode,
                wins=int(x.sum()),
                games=x.size,
                win_pct=100 * x.mean(),
                lo95=lo,
                hi95=hi,
                paired_gain=100 * delta.mean(),
                gain_lo95=dl,
                gain_hi95=dh,
                seeds=" / ".join(f"{100 * z.mean():.2f}" for z in x),
                directions=" / ".join(f"{100 * x[:, :, :, d].mean():.2f}" for d in (0, 1)),
            )
        )
    if summary:
        atomic_rows(OUT / "development.csv", summary)
    lines = [
        "# Learning to predict opponent replies for model-based RL",
        "",
        "Our first policy won 80.71% on development and 77.17% on its independent 16-roster test. We retired that test, kept the original development matrix, and reserved another 16 teams before scoring any new method. Each reserved roster shares at most three species with any roster previously used in the project. [First confirmation](../sketch5-mb-policy-iteration/confirmation/README.md) · [Second-round protocol](protocol.json)",
        "",
    ]
    fresh = LOCK.parent / "results.csv"
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        winner = next(r for r in summary if r["method"] == lock["method"])
        lines += [
            f"We froze the three-step reply planner after it won {winner['wins']:,} of 2,048 development games ({winner['win_pct']:.2f}%, 95% interval {winner['lo95']:.2f}%–{winner['hi95']:.2f}%). Across the four pilot seeds it scored {winner['seeds']}%. All {len(lock['frozen_assets'])} deployment and evaluation assets were fingerprinted before the reserved battles. [Frozen configuration and six-way selection](confirmation/candidate.json)",
            "",
        ]
    if fresh.exists():
        r = read(fresh)
        method = json.loads(LOCK.read_text())["method"]
        mainrow = next(x for x in r if x["method"] == method and x["split"] == "primary")
        lines.extend(
            [
                f"The frozen second-round policy won **{int(mainrow['wins']):,} of {int(mainrow['games']):,} ({float(mainrow['win_pct']):.2f}%, 95% interval {float(mainrow['lo95']):.2f}%–{float(mainrow['hi95']):.2f}%)** on its new reciprocal test. The same-schedule gain over the first-round policy was {float(mainrow['incumbent_gain']):+.2f} points ({float(mainrow['incumbent_lo95']):+.2f} to {float(mainrow['incumbent_hi95']):+.2f}).",
                "",
                "| Frozen evaluation | Original sampled PPO | First-round planner | Second-round planner (95% interval) | Gain vs. original PPO (95% interval) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for split, label in (
            ("primary", "Reciprocal vs. BC"),
            ("custom", "Original Sun Room vs. BC"),
            ("transfer", "Reciprocal vs. independent PPO"),
        ):
            t = {x["method"]: x for x in r if x["split"] == split}
            v = t[method]
            lines.append(
                f"| {label} | {float(t['native']['win_pct']):.2f}% | {float(t[OLD]['win_pct']):.2f}% | {float(v['win_pct']):.2f}% ({float(v['lo95']):.2f}–{float(v['hi95']):.2f}) | {float(v['baseline_gain']):+.2f} ({float(v['baseline_lo95']):+.2f} to {float(v['baseline_hi95']):+.2f}) |"
            )
        archive = LOCK.parent / "games.csv.gz"
        with gzip.open(archive, "rt") as f:
            assigned = defaultdict(list)
            n = 0
            for row in csv.DictReader(f):
                n += 1
                if row["split"] == "primary":
                    assigned[row["method"], row["own"]].append(row["reward"] == "1")
        assert n == 52224 and all(len(z) == 512 for z in assigned.values())
        teams = [Path(p).stem for p in SPEC["confirmation_a"] + SPEC["confirmation_b"]]
        at80 = sum(np.mean(assigned[method, t]) >= 0.8 for t in teams)
        gained = sum(sum(assigned[method, t]) > sum(assigned[OLD, t]) for t in teams)
        lines += [
            "",
            f"The second policy reached 80% with {at80} of 16 assigned rosters. It improved on the first-round planner with {gained} of 16. The primary rates for pilot seeds 2–5 were {mainrow['seeds']}%. Group A as learner and Group B as learner scored {mainrow['directions']}%.",
            "",
            f"We checked all {n:,} game records, including starting fours, both seats, terminal outcomes and simulator seeds. The SHA-256 of the game archive is `{hashlib.sha256(archive.read_bytes()).hexdigest()}`. [Frozen deployment assets](confirmation/candidate.json) · [Raw games](confirmation/games.csv.gz) · [Every matchup](confirmation/matchups.csv) · [Paired effects and intervals](confirmation/results.csv)",
            "",
        ]
    lines += [
        "## Development",
        "",
        "| Method | Wins out of 2,048 | Win rate (95% interval) | Difference vs. first-round planner (points) |",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['method']} | {r['wins']:,} | {r['win_pct']:.2f}% ({r['lo95']:.2f}–{r['hi95']:.2f}) | {r['paired_gain']:+.2f} ({r['gain_lo95']:+.2f} to {r['gain_hi95']:+.2f}) |"
        )
    lines += [
        "",
        "We fixed six candidate configurations before the first complete four-seed result. The highest development win count qualifies at 1,680 of 2,048 wins (82%). A fixed order resolves ties. Every development comparison uses the incumbent’s assigned rosters, simulator seats, seeds and both submitted lineups. Intervals resample training seeds, both team axes and game replicates within seat. [Selection rule](selection.json)",
        "",
        "## What the policy learns and simulates",
        "",
        "All arms use the same four PPO battle pilots, trained mechanics preferences, terminal win/loss/draw reward and 196-game public lineup planner as the first candidate. We changed how the bot evaluates the moves it could play next.",
        "",
        "- **Neural opponent responses.** For each public matchup, simulate 48 BC team previews across four hypothetical private stat spreads. Condition possible reserves on the Pokémon already revealed. Construct an opponent-perspective embedding for each hypothetical world using guessed stats and selected reserves. Three of each four worlds sample the released actor's predicted moves. The fourth samples the earlier mechanics-based reply model. Each possible move is judged by the learned value after Showdown resolves the joint turn.",
        "- **Search width and depth.** Compare four actions under four worlds, eight actions under eight worlds, and three-step lookahead. The first step uses the modeled BC response. Further steps use the mechanics-based continuation policy.",
        "- **Factored joint-action search.** Hold one partner's move fixed while testing up to nine commands for the other. Combine the most promising commands for both partners, then compare up to eight joint finalists on eight fresh modeled worlds.",
        "- **World value learning.** Fit a separate residual state-value network from final returns in 1,120 training battles per seed. The collector sees the fully specified training world, which matches the information available inside a hypothetical simulation. At deployment its input is the sampled world. The contest opponent's actual private state is inaccessible to this model.",
        "",
    ]
    if (OUT / "public-input-audit.csv").exists():
        rows = read(OUT / "public-input-audit.csv")
        assert len(rows) == 16 and all(
            r["identical_hypotheses_logits_and_action"] == "True" for r in rows
        )
        lines += [
            "Across 16 training fixtures, changing both the true hidden reserves and private stats with the public board fixed produced identical hypothesized reserves, response logits and selected commands in both seats. [Intervention records](public-input-audit.csv)",
            "",
        ]
    if (OUT / "move-latency.csv").exists():
        timings = read(OUT / "move-latency.csv")
        comparisons = {r["mode"]: r for r in timings}
        timing_modes = (
            ("incumbent", json.loads(LOCK.read_text())["method"])
            if LOCK.exists()
            else ("incumbent", "reply-wide", "coordinate")
        )
        detail = ", ".join(f"{m} {float(comparisons[m]['median_ms']):.0f} ms" for m in timing_modes)
        lines += [
            f"Median move-selection times on the same {timings[0]['matched_decisions']} decisions from eight training-roster games were {detail}. These timings used the shared Sketch5 host. [Median and 95th percentile by method](move-latency.csv) · [Cached and uncached preview timing](preview-latency.csv)",
            "",
        ]
    if (OUT / "reply-predictions.csv").exists():
        rows = read(OUT / "reply-predictions.csv")
        by = {m: [r for r in rows if r["model"] == m] for m in ("neural", "heuristic")}

        def fmt(m, c):
            return f"{100 * np.mean([float(r[c]) for r in by[m]]):.2f}%"

        lines += [
            f"On {len(by['neural'])} training-roster decisions, a sampled neural response matched the actual BC command, including its target and Mega choice, on {fmt('neural', 'slot_match')} of Pokémon slots. The old response model matched {fmt('heuristic', 'slot_match')}. Exact joint-action agreement was {fmt('neural', 'joint_match')} and {fmt('heuristic', 'joint_match')}, respectively. [Per-decision predictions](reply-predictions.csv)",
            "",
        ]
    if (OUT / "reserve-predictions.csv").exists():
        reserve = read(OUT / "reserve-predictions.csv")
        grouped = defaultdict(list)
        for row in reserve:
            grouped[row["seed"], row["game"]].append(row)
        if all(len(z) == 4 and sum(float(r["target"]) for r in z) == 2 for z in grouped.values()):
            error = np.mean([(float(r["posterior"]) - float(r["target"])) ** 2 for r in reserve])
            lines += [
                f"On {len(grouped)} opening positions, the modeled reserve probabilities had a Brier score of {error:.3f}. Assigning 50% to each of the four unseen Pokémon scored 0.250. Lower is better. [Reserve forecasts and checked outcomes](reserve-predictions.csv)",
                "",
            ]
    lines += [
        "## Exact team matrix",
        "",
        f"- Training learner pool (28) — {names(SPEC['own_train'])}.",
        f"- Training opponent pool (20) — {names(SPEC['foe_train'])}.",
        f"- Development A — {names(SPEC['development_a'])}.",
        f"- Development B — {names(SPEC['development_b'])}.",
        f"- Second confirmation A — {names(SPEC['confirmation_a'])}.",
        f"- Second confirmation B — {names(SPEC['confirmation_b'])}.",
        "- The learner plays every A team against every B team, then the two sides exchange rosters. The new reciprocal test plays 16 games per direction per seed, alternating seats. The original Sun Room team separately plays all 16 tournament sheets. The independent opponent is shared PPO seed 2 at 262,144 steps with native BC preview. [Sun Room import](../../teams/reg-mb-original/sun-room.txt)",
        "",
        "The roster exclusions cover our own training and model selection. The released behavior-cloning checkpoint provides the common starting point for the pilots and uses its upstream replay data. These results measure performance in the pinned Showdown simulator.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    for r in summary:
        print(r["method"], r["wins"], pct(r["win_pct"]), flush=True)


if __name__ == "__main__":
    main()
