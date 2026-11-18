"""Repeat the fixed-team fine-tuning recipe with an M-B legal variant and open sheets."""

import json
from concurrent.futures import ThreadPoolExecutor

from opponent_diversity import ROOT, SEEDS, launch

TEAM = "teams/reg-mb-variant/MC461-MB.txt"
POOL = "teams/reg-mb-variant/training-pool.json"
FORMAT = "gen9championsvgc2026regmb"
RUN = ROOT / "runs/sketch5-vgcbench-mb-v1"


def train(seed):
    dest = RUN / f"seed-{seed}"
    if (dest / "train/summary.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    warm = ROOT / f"runs/sketch5-move-slots-v1/seed-{seed}/fixed/train/checkpoint-262144.pt"
    launch(
        dest,
        "train",
        [
            "--team",
            TEAM,
            "--opponent-teams",
            *json.loads((ROOT / POOL).read_text()),
            "--format",
            FORMAT,
            "--open-team-sheets",
            "--opponent",
            "mixture",
            "--opponent-pool",
            POOL,
            "--reward-mode",
            "win-loss",
            "--steps",
            "65536",
            "--seed",
            str(110 + seed),
            "--hidden-size",
            "256",
            "--transformer-chunks",
            "full",
            "--learning-rate",
            "0.0001",
            "--init-checkpoint",
            str(warm),
            "--transfer-champions-regulation",
            "--save-initial",
            "--save-at",
            "16384",
            "65536",
            "--run-dir",
            str(dest / "train"),
        ],
    )


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        list(pool.map(train, SEEDS))
