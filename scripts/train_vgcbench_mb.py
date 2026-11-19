"""Repeat the fixed-team fine-tuning recipe with an M-B legal variant and open sheets."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from opponent_diversity import ROOT, SEEDS, launch

TEAM = "teams/reg-mb-variant/MC461-MB.txt"
POOL = "teams/reg-mb-variant/training-pool.json"
FORMAT = "gen9championsvgc2026regmb"
RUN = ROOT / "runs/sketch5-vgcbench-mb-v1"


BC_RUN = ROOT / "runs/sketch5-vgcbench-bc-v1"


def train(seed, bc=False):
    dest = (BC_RUN if bc else RUN) / f"seed-{seed}"
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
            *(["vgcbench-bc"] if bc else ["mixture", "--opponent-pool", POOL]),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc", action="store_true", help="Train against the frozen behavior-cloned bot"
    )
    args = parser.parse_args()
    if args.bc:
        from evaluate_vgcbench_mb import lock, protocol

        spec = protocol()
        assert not (set(spec["training_pool"]) & set(spec["teams"]))
        spec["primary"] = (
            "BC-trained vs league-trained PPO at 65536 decisions against BC on 12 held-out teams"
        )
        spec["training_opponent"] = "bc-learned"
        spec["training_teams"] = list(spec.pop("training_pool"))
        BC_RUN.mkdir(parents=True, exist_ok=True)
        lock(BC_RUN / "protocol.json", spec)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        list(pool.map(train, SEEDS, [args.bc] * len(SEEDS)))
