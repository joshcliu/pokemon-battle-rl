"""Train roster specialists and fixed-team PPO candidates for a frozen doubles league."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import ROOT, launch

TEAM = "teams/reg-mc-regionals/MC461.txt"
ROSTERS = (
    "teams/balance.txt",
    "teams/sun.txt",
    "teams/reg-mc-regionals/MC463.txt",
    "teams/reg-mc-regionals/MC751.txt",
)
ARCHIVE = ROOT / "runs/sketch5-move-slots-v1"


def train(root, label, team, opposition, seed, steps, warm=None, pool=None, rate=None):
    job = root / label
    job.mkdir(parents=True)
    args = [
        "--team",
        team,
        "--opponent-teams",
        *opposition,
        "--opponent",
        "mixture",
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--reward-mode",
        "win-loss",
        "--save-at",
        *[str(n) for n in (16384, 32768, 65536, 131072, 262144) if n <= steps],
        "--run-dir",
        str(job / "train"),
    ]
    if warm:
        args.extend(
            ["--hidden-size", "256", "--transformer-chunks", "full", "--init-checkpoint", str(warm)]
        )
    if pool:
        args.extend(["--opponent-pool", str(pool)])
    if rate:
        args.extend(["--learning-rate", str(rate)])
    launch(job, "train", args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["rivals", "warmup", "league", "experts"])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--pool", type=Path)
    p.add_argument("--steps", type=int, default=65536)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    root = a.runs.resolve()
    if a.stage in ("league", "experts") and not a.pool:
        p.error("League training requires --pool")
    with ThreadPoolExecutor(max_workers=a.jobs) as workers:
        if a.stage == "rivals":
            tasks = [
                workers.submit(
                    train,
                    root,
                    f"rivals/{Path(t).stem}-{s}",
                    t,
                    [
                        TEAM,
                        "teams/reg-mc-regionals/MC646.txt",
                        "teams/reg-mc-regionals/MC460.txt",
                        "teams/reg-mc-regionals/MC650.txt",
                    ],
                    s,
                    a.steps,
                )
                for t in ROSTERS
                for s in (71, 73)
            ]
        elif a.stage == "experts":
            pool = json.loads(a.pool.read_text())
            tasks = []
            for t in ROSTERS:
                stem = Path(t).stem
                file = root / f"expert-{stem}-pool.json"
                file.write_text(json.dumps({t: pool[t]}, indent=2) + "\n")
                for s in (2, 5, 13):
                    warm = (
                        ARCHIVE / f"seed-{s}/fixed/train/checkpoint-262144.pt"
                        if s in (2, 5)
                        else None
                    )
                    tasks.append(
                        workers.submit(
                            train,
                            root,
                            f"experts/{stem}-{s}",
                            TEAM,
                            [t],
                            s + 200,
                            a.steps,
                            warm,
                            file,
                            0.00015 if warm else None,
                        )
                    )
        else:
            tasks = []
            for s in (2, 3, 4, 5):
                warm = ARCHIVE / f"seed-{s}/fixed/train/checkpoint-262144.pt"
                tasks.append(
                    workers.submit(
                        train,
                        root,
                        f"{a.stage}/warm-{s}",
                        TEAM,
                        ROSTERS,
                        s + 110,
                        a.steps,
                        warm,
                        a.pool.resolve() if a.pool else None,
                        0.0001,
                    )
                )
            for s in (11, 13, 17, 19):
                tasks.append(
                    workers.submit(
                        train,
                        root,
                        f"{a.stage}/mlp-{s}",
                        TEAM,
                        ROSTERS,
                        s,
                        a.steps,
                        None,
                        a.pool.resolve() if a.pool else None,
                    )
                )
        for task in tasks:
            task.result()
