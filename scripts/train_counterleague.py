"""Adapt the previous champion to four rosters and train best responses to frozen opponents."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opponent_diversity import ROOT, launch
from train_fixed_league import ARCHIVE, TEAM, train

PREVIOUS = ROOT / "results/sketch5-fixed-league"
ORIGINAL = next(iter(json.loads((PREVIOUS / "single.json").read_text()).values()))
TEAMS = tuple(f"teams/reg-mc-regionals/MC{i}.txt" for i in (463, 646, 751, 781))


def save(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Run configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def adapt(root, pool, seed, rate, steps):
    job = root / "adapt" / str(seed)
    job.mkdir(parents=True)
    launch(
        job,
        "train",
        [
            "--learner-teams",
            *TEAMS,
            "--opponent-teams",
            *pool,
            "--opponent-pool",
            str(root / "adapt-pool.json"),
            "--opponent",
            "mixture",
            "--transformer-chunks",
            "full",
            "--hidden-size",
            "256",
            "--init-checkpoint",
            ORIGINAL,
            "--learning-rate",
            str(rate),
            "--seed",
            str(seed),
            "--reward-mode",
            "win-loss",
            "--steps",
            str(steps),
            "--save-initial",
            "--save-at",
            "16384",
            "32768",
            str(steps),
            "--run-dir",
            str(job / "train"),
        ],
    )


def mirror_jobs(executor, root, team, defender, steps):
    name = Path(team).stem
    pool = root / f"mirror-{name}.json"
    save(pool, {team: [defender]})
    starts = {
        331: None,
        337: None,
        347: ARCHIVE / "seed-2/fixed/train/checkpoint-262144.pt",
        349: ROOT / defender,
    }
    return [
        executor.submit(
            train,
            root,
            f"counter/{name}-{seed}",
            team,
            [team],
            seed,
            steps,
            warm,
            pool,
            0.0002 if warm else None,
        )
        for seed, warm in starts.items()
    ]


def shared(root, pool, seed, warm, steps):
    job = root / "shared" / str(seed)
    job.mkdir(parents=True)
    architecture = (
        ["--transformer-chunks", "full", "--init-checkpoint", warm, "--learning-rate", "0.0002"]
        if warm
        else []
    )
    launch(
        job,
        "train",
        [
            "--learner-teams",
            *pool,
            "--opponent-teams",
            *pool,
            "--mirror-teams",
            "--opponent-pool",
            str(root / "shared-pool.json"),
            "--opponent",
            "mixture",
            "--hidden-size",
            "256",
            *architecture,
            "--seed",
            str(seed),
            "--reward-mode",
            "win-loss",
            "--steps",
            str(steps),
            "--save-at",
            "16384",
            "32768",
            str(steps),
            "--run-dir",
            str(job / "train"),
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["start", "counter", "fixed-experts"])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--adapt-steps", type=int, default=65536)
    p.add_argument("--counter-steps", type=int, default=32768)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    root = a.runs.resolve()
    assert hashlib.sha256((ROOT / ORIGINAL).read_bytes()).hexdigest() == (
        "6e5a00ba994721c23afcb3100ab2277601dab688d097fbe15ab2874bbbb01bad"
    )
    with ThreadPoolExecutor(max_workers=a.jobs) as exe:
        tasks = []
        if a.stage == "start":
            pool = json.loads((PREVIOUS / "pool.json").read_text())
            pool[TEAM] = [ORIGINAL]
            save(root / "adapt-pool.json", pool)
            for seed, rate in [(291, 0.0001), (293, 0.0001), (295, 0.0003), (297, 0.0003)]:
                tasks.append(exe.submit(adapt, root, pool, seed, rate, a.adapt_steps))
            tasks.extend(mirror_jobs(exe, root, TEAM, ORIGINAL, a.counter_steps))
        elif a.stage == "fixed-experts":
            pool = json.loads((root / "shared-pool.json").read_text())
            for t in TEAMS:
                stem = Path(t).stem
                file = root / f"mirror-{stem}.json"
                assert json.loads(file.read_text()) == {t: pool[t]}
                for seed, warm in [
                    (401, None),
                    (409, ROOT / ORIGINAL),
                    (419, ARCHIVE / "seed-5/fixed/train/checkpoint-262144.pt"),
                ]:
                    tasks.append(
                        exe.submit(
                            train,
                            root,
                            f"fixed-experts/{stem}-{seed}",
                            TEAM,
                            [t],
                            seed,
                            a.counter_steps,
                            warm,
                            file,
                            0.0002 if warm else None,
                        )
                    )
        else:
            lock = json.loads((root / "defender.json").read_text())
            cp = lock["model"]
            assert hashlib.sha256((ROOT / cp).read_bytes()).hexdigest() == lock["sha256"]
            for team in TEAMS:
                tasks.extend(mirror_jobs(exe, root, team, cp, a.counter_steps))
            pool = {t: [cp] for t in TEAMS}
            pool[TEAM] = [ORIGINAL]
            save(root / "shared-pool.json", pool)
            for seed, warm in [(353, None), (359, cp), (367, ORIGINAL)]:
                tasks.append(exe.submit(shared, root, pool, seed, warm, a.adapt_steps))
            for seed, warm in [
                (373, None),
                (379, ROOT / ORIGINAL),
                (383, ARCHIVE / "seed-2/fixed/train/checkpoint-262144.pt"),
            ]:
                tasks.append(
                    exe.submit(
                        train,
                        root,
                        f"fixed-counter/{seed}",
                        TEAM,
                        list(pool),
                        seed,
                        a.adapt_steps,
                        warm,
                        root / "shared-pool.json",
                        0.0002 if warm else None,
                    )
                )
        for task in tasks:
            task.result()
