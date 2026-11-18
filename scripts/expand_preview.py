"""Run an enlarged MC461 preview sweep and prospective paired confirmation on Sketch5."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluate_challenger_transfer import digest, read
from preview_sweep import LINEUPS, ROOT, cell, initial_manifest, save
from preview_sweep import RUN as PRIOR_RUN

RUN = ROOT / "runs/sketch5-preview-expanded-v1"
SOURCE = Path("teams/reg-mc-preview-expanded")
PREVIOUS = ROOT / "results/sketch5-preview"
OUT = ROOT / "results/sketch5-preview-expanded"
MARGINS = (0.0, 0.025, 0.05, 0.10, 0.20, 1.0)


def manifest():
    spec = initial_manifest()
    prev = read(PREVIOUS / "manifest.json")
    assert spec["parent"] == prev["parent"] and spec["opponents"] == prev["opponents"]
    assert all(spec["sha256"][p] == h for p, h in prev["sha256"].items())
    assert (
        digest(PREVIOUS / "preview-head.pt")
        == read(PREVIOUS / "selection-lock.json")["head_sha256"]
    )
    original = spec["teams"]["train"]
    rows = read(ROOT / SOURCE / "source.json")
    additional = [str(SOURCE / (r["team_id"] + ".txt")) for r in rows]
    spec["original_train"] = original
    spec["teams"]["train"] = additional
    spec["sha256"].update({p: digest(p) for p in additional})
    spec["prior_head_sha256"] = digest(PREVIOUS / "preview-head.pt")
    spec["prior_training_sha256"] = digest(PRIOR_RUN / "training-sweep.csv")
    assert (
        spec["prior_training_sha256"]
        == read(PREVIOUS / "selection-lock.json")["training_table_sha256"]
    )
    spec["counts"] = {"train": 12, "confirm": 256}
    # Avoid every logged simulator seed from the two stages that trained the frozen actor.
    seen = set()
    for path in (
        "runs/sketch5-fixed-league-v1/league/warm-2/train/episodes.jsonl",
        "runs/sketch5-move-slots-v1/seed-2/fixed/train/episodes.jsonl",
    ):
        with (ROOT / path).open() as stream:
            seen.update(json.loads(line)["simulator_seed"] for line in stream)
    spec["base"] = {}
    for stage, start, teams in (
        ("train", 791000000, additional),
        ("confirm", 851000000, spec["teams"]["test"]),
    ):
        n = spec["counts"][stage]
        while any(
            start + cell_index * 1000 + k in seen
            for cell_index in range(len(teams) * len(spec["opponents"]))
            for k in range(n)
        ):
            start += 1000000
        spec["base"][stage] = start
    spec["fallback_margins"] = MARGINS
    return spec


def sweep(root, jobs):
    path = root / "manifest.json"
    if not path.exists():
        save(path, manifest())
    spec = read(path)
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    bots = [(j, k, v) for j, (k, v) in enumerate(spec["opponents"].items())]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = [
            pool.submit(cell, root, "train", spec, team, i, lineup, bots)
            for i, team in enumerate(spec["teams"]["train"])
            for lineup in LINEUPS
        ]
        for n, job in enumerate(as_completed(tasks), 1):
            job.result()
            if n % 250 == 0 or n == len(tasks):
                print(f"New sweep: {n}/{len(tasks)} cells", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=RUN)
    p.add_argument("--jobs", type=int, default=40)
    a = p.parse_args()
    sweep(a.root, a.jobs)
