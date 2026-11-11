"""Plot recorded training performance; this is not held-out evaluation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", default=100, type=int)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run / "episodes.jsonl").read_text().splitlines()]
    if not 1 <= args.window <= len(records):
        parser.error("Window must be between one and the number of recorded episodes")
    wins = np.array([r["outcome"] == "win" for r in records], dtype=float)
    rate = np.convolve(wins, np.ones(args.window) / args.window, mode="valid")
    steps = [r["step"] for r in records][args.window - 1 :]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(steps, rate, color="#166a97", linewidth=2)
    ax.set(
        xlabel="Learner decision steps",
        ylabel=f"Win rate, last {args.window} episodes",
        ylim=(0, 1),
        title="PPO pilot: training performance (one seed)",
    )
    ax.grid(alpha=0.2)
    fig.supxlabel(
        "Training opponents: 50/50 damage and switch. Not a generalization result.", fontsize=9
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
