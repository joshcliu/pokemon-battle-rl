"""Calibrate a fixed opponent population and evaluate frozen policies on reserved seeds."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from train_fixed_league import ARCHIVE, ROOT, ROSTERS, TEAM

from pokemon_rl.battle import LocalBattle
from pokemon_rl.league import FrozenActor, LeadRouter, Tactical, visible_leads
from pokemon_rl.opponents import Heuristic

FAMILIES = ("damage", "tactical", "specialist")


def name(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def actor(key, rng):
    if key.endswith(".json"):
        return LeadRouter(ROOT / key)
    if key.endswith(".pt"):
        return FrozenActor(ROOT / key)
    if key == "tactical":
        return Tactical(rng)
    return Heuristic(key, rng)


def matches(condition, learner, team, opponent, other_team, base, start, count, replay_dir=None):
    torch.set_num_threads(1)
    texts = [(ROOT / t).read_text() for t in (team, other_team)]
    actors = [
        actor(k, np.random.default_rng(0)) if k.endswith((".pt", ".json")) else None
        for k in (learner, opponent)
    ]
    rows = []
    with LocalBattle() as battle:
        for i in range(start, start + count):
            es = base + i
            sides = ("p1", "p2") if i % 2 == 0 else ("p2", "p1")
            players = {}
            for pos, (side, key) in enumerate(zip(sides, (learner, opponent), strict=True)):
                if actors[pos]:
                    actors[pos].reset()
                    players[side] = actors[pos]
                else:
                    players[side] = actor(
                        key, np.random.default_rng(np.random.SeedSequence([es, 51 + pos]))
                    )
            battle.reset(es, *(texts if i % 2 == 0 else reversed(texts)))
            leads = visible_leads(battle.players[sides[0]])
            requests = 0
            while not battle.ended and battle.turn < 200 and requests < 600:
                choices = {}
                for side in ("p1", "p2"):
                    if battle.needs_action(side):
                        legal = battle.legal_actions(side)
                        pair = players[side].choose(battle.players[side], legal)
                        choices[side] = legal[pair].message.removeprefix("/choose ")
                battle.step(choices)
                requests += 1
            own = "Learner" if sides[0] == "p1" else "Opponent"
            if replay_dir is not None:
                battle.save(replay_dir, f"{es}-{sides[0]}")
            outcome = (
                "cutoff"
                if not battle.ended
                else "tie"
                if not battle.winner
                else "win"
                if battle.winner == own
                else "loss"
            )
            rows.append(
                {
                    **condition,
                    "learner": learner,
                    "team": team,
                    "opponent": opponent,
                    "opponent_team": other_team,
                    "visible_leads": leads,
                    "seed": es,
                    "side": sides[0],
                    "outcome": outcome,
                    "turn": battle.turn,
                }
            )
    return rows


def run(out, jobs, tasks):
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=jobs) as workers:
        futures = [workers.submit(matches, *args) for args in tasks]
        rows = [r for future in futures for r in future.result()]
    write_csv(out / "games.csv.gz", rows)
    scores = {}
    for r in rows:
        key = r["candidate"], r["roster"], r["family"]
        scores.setdefault(key, Counter())[r["outcome"]] += 1
    summary = []
    for (candidate, roster, family), counts in sorted(scores.items()):
        n = counts.total()
        summary.append(
            dict(
                candidate=candidate,
                roster=roster,
                family=family,
                battles=n,
                wins=counts["win"],
                losses=counts["loss"],
                ties=counts["tie"],
                cutoffs=counts["cutoff"],
                win_pct=100 * counts["win"] / n,
            )
        )
    write_csv(out / "scores.csv", summary)
    return summary


def specialists(root, jobs):
    tasks = []
    ref = name(ARCHIVE / "seed-2/fixed/train/checkpoint-262144.pt")
    for ri, t in enumerate(ROSTERS):
        for s in (71, 73):
            for step in (32768, 65536):
                cp = name(root / f"rivals/{Path(t).stem}-{s}/train/checkpoint-{step}.pt")
                for j, (opp, other) in enumerate(
                    [
                        (ref, TEAM),
                        ("tactical", TEAM),
                        ("tactical", "teams/reg-mc-regionals/MC646.txt"),
                        ("damage", t),
                    ]
                ):
                    cond = {"candidate": cp, "roster": Path(t).stem, "family": str(j)}
                    tasks.append((cond, cp, t, opp, other, 8100000 + ri * 10000 + j * 1000, 0, 64))
    scores = run(root / "selection", jobs, tasks)
    totals = Counter()
    for row in scores:
        totals[row["candidate"]] += row["wins"]
    pool = {}
    for t in ROSTERS:
        best = max(sorted(k for k in totals if f"/{Path(t).stem}-" in k), key=totals.get)
        pool[t] = ["damage", "tactical", best]
    (root / "pool.json").write_text(json.dumps(pool, indent=2) + "\n")


def evaluate(out, pool, roster, candidates, base, n, jobs, calibrate=False):
    tasks = []
    for label, learner in candidates.items():
        for ri, t in enumerate(ROSTERS):
            for fi, (family, opp) in enumerate(zip(FAMILIES, pool[t], strict=True)):
                cond = {"candidate": label, "roster": Path(t).stem, "family": family}
                cell = base + (ri * len(FAMILIES) + fi) * 10000
                for start in range(0, n, 50):
                    tasks.append(
                        (
                            cond,
                            learner,
                            t if calibrate else roster,
                            opp,
                            t,
                            cell,
                            start,
                            min(50, n - start),
                        )
                    )
    return run(out, jobs, tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "calibrate", "eval"])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--team", default=TEAM)
    p.add_argument(
        "--candidates", type=Path, help="JSON mapping of labels to checkpoints or heuristics"
    )
    p.add_argument("--base", type=int, default=10000000)
    p.add_argument("--battles", type=int, default=100)
    p.add_argument("--jobs", type=int, default=24)
    args = p.parse_args()
    root = args.runs.resolve()
    if args.stage == "select":
        specialists(root, args.jobs)
    else:
        pool = json.loads((root / "pool.json").read_text())
        candidates = (
            {"random": "random", "damage": "damage", "tactical": "tactical"}
            if args.stage == "calibrate"
            else json.loads(args.candidates.read_text())
        )
        evaluate(
            args.out,
            pool,
            args.team,
            candidates,
            args.base,
            args.battles,
            args.jobs,
            args.stage == "calibrate",
        )
