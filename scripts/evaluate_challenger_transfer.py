"""Evaluate frozen challengers and their ancestors on the original seven test rosters."""

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from analyze_long_rewards import write_csv
from fixed_league import matches
from roster_transfer import ROOT, split

MODELS = {
    "tournament-mlp": "runs/sketch5-finetune-v1/seed-2/scratch/train/checkpoint.pt",
    "tournament-full65": "runs/sketch5-policy-size-v1/seed-2/wide-transformer-full/train/checkpoint.pt",
    "tournament-full262": "runs/sketch5-move-slots-v1/seed-2/fixed/train/checkpoint-262144.pt",
    "parent": "results/sketch5-fixed-league/weights/expert-4.pt",
    "adapted": "results/sketch5-counterleague/weights/defender.pt",
    "challenger-mirror": "results/sketch5-counterleague/weights/counter-12.pt",
    "challenger-fixed": "results/sketch5-counterleague/weights/counter-8.pt",
}
SOURCE_RUNS = {
    **{k: str(Path(v).parent) for k, v in list(MODELS.items())[:3]},
    "parent": "runs/sketch5-fixed-league-v1/league/warm-2/train",
    "adapted": "runs/sketch5-counterleague-v1/adapt/297/train",
    "challenger-mirror": "runs/sketch5-counterleague-v1/shared/367/train",
    "challenger-fixed": "runs/sketch5-counterleague-v1/fixed-counter/379/train",
}
FAMILIES = ("damage", "switch", "setup", "random", "tactical")
FRESH = {"MC407", "MC778", "MC462", "MC403", "MC405"}


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def opponents():
    result = [
        (style, team, style)
        for style in FAMILIES
        for team in ("teams/balance.txt", "teams/sun.txt")
    ]
    pool = read(ROOT / "results/sketch5-fixed-league/pool.json")
    result.extend(("ppo", team, actors[2]) for team, actors in pool.items())
    return result


def make_manifest():
    torch.set_num_threads(1)
    provenance = {}
    for label, checkpoint in MODELS.items():
        saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=True)
        source = read(ROOT / SOURCE_RUNS[label] / "manifest.json")
        args = source["args"]
        assert saved["seed"] == args["seed"]
        assert saved["steps"] == args["steps"]
        assert args["reward_mode"] == "win-loss"
        assert all(digest(t) == d["sha256"] for t, d in source["teams"].items())
        provenance[label] = {
            "path": checkpoint,
            "sha256": digest(checkpoint),
            "steps_in_this_stage": saved["steps"],
            "seed": saved["seed"],
            "history": saved["transformer_chunks"],
            "width": saved["hidden_size"],
            "training_manifest": SOURCE_RUNS[label] + "/manifest.json",
            "learner_teams": args.get("learner_teams") or [args["team"]],
            "opponent_teams": args["opponent_teams"],
            "init_checkpoint": args.get("init_checkpoint"),
        }
    teams = split()["test"]
    assert len(teams) == 7
    assert FRESH == {Path(t).stem for t in teams} - {"MC646", "MC751"}
    # Confirm the five exact rosters were never selected on either side in this ancestry.
    for r in provenance.values():
        assert not FRESH.intersection(
            Path(t).stem for t in r["learner_teams"] + r["opponent_teams"]
        )
    specs = opponents()
    files = {t for _, t, _ in specs} | set(teams)
    files.update(k for _, _, k in specs if k.endswith(".pt"))
    # Reserve physical simulator seeds and exclude ancestral training and prior evaluations.
    used = set()
    for directory in (
        "sketch5-move-slots-v1",
        "sketch5-policy-size-v1",
        "sketch5-finetune-v1",
        "sketch5-fixed-league-v1",
        "sketch5-counterleague-v1",
    ):
        for log in (ROOT / "runs" / directory).rglob("episodes.jsonl"):
            with log.open() as f:
                used.update(json.loads(line)["simulator_seed"] for line in f if line.endswith("\n"))
    base = 88000000
    while any(
        base + mode * 2000000 + (i * len(specs) + j) * 1000 + k in used
        for mode in (0, 1)
        for i in range(7)
        for j in range(len(specs))
        for k in range(200)
    ):
        base += 5000000
    return {
        "models": provenance,
        "teams": teams,
        "opponents": specs,
        "sha256": {f: digest(f) for f in sorted(files)},
        "fresh_five": sorted(FRESH),
        "base": base,
        "greedy_games_per_cell": 200,
        "sampled_games_per_cell": 100,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "simulator": read(ROOT / "simulator.lock.json"),
    }


def evaluate_cell(root, spec, label, mode, team_index, opponent_index):
    family, other_team, other = spec["opponents"][opponent_index]
    team = spec["teams"][team_index]
    dest = root / mode / label / Path(team).stem / f"{family}-{Path(other_team).stem}.csv"
    sampled = mode == "sampled"
    count = spec[f"{mode}_games_per_cell"]
    base = (
        spec["base"]
        + 2000000 * sampled
        + (team_index * len(spec["opponents"]) + opponent_index) * 1000
    )
    if dest.exists():
        with dest.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == count and {int(r["seed"]) for r in rows} == set(
            range(base, base + count)
        )
        return
    condition = {"candidate": label, "roster": Path(team).stem, "family": family, "mode": mode}
    rows = matches(
        condition,
        spec["models"][label]["path"],
        team,
        other,
        other_team,
        base,
        0,
        count,
        sample_learner=sampled,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".tmp")
    write_csv(temporary, rows)
    temporary.replace(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--mode", choices=["greedy", "sampled"], required=True)
    a = p.parse_args()
    root = a.runs.resolve()
    manifest = root / "manifest.json"
    if not manifest.exists():
        spec = make_manifest()
        root.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(spec, indent=2) + "\n")
    spec = read(manifest)
    assert all(digest(r["path"]) == r["sha256"] for r in spec["models"].values())
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    assert spec["simulator"] == read(ROOT / "simulator.lock.json")
    cells = [
        (i, j)
        for i in range(len(spec["teams"]))
        for j, (family, _, _) in enumerate(spec["opponents"])
        if a.mode == "greedy" or family in ("damage", "switch")
    ]
    with ProcessPoolExecutor(max_workers=a.jobs) as workers:
        futures = [
            workers.submit(evaluate_cell, root, spec, label, a.mode, i, j)
            for label in (
                "challenger-mirror",
                "parent",
                "tournament-full262",
                "challenger-fixed",
                "tournament-mlp",
                "tournament-full65",
                "adapted",
            )
            for i, j in cells
        ]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 50 == 0 or n == len(futures):
                print(f"{a.mode}: {n}/{len(futures)} matchup cells complete", flush=True)
