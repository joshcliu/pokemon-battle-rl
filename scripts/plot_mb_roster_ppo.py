"""Plot the matched win-rate differences against shared PPO for each deployment setting."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1] / "results/sketch5-mb-roster-ppo"
    with (root / "effects.csv").open() as f:
        data = {(r["arm"], r["baseline"], r["cell"]): r for r in csv.DictReader(f)}
    fig, ax = plt.subplots(figsize=(8.5, 4.1), layout="constrained")
    cells = ["familiar", "familiar_vs_unseen", "balanced"]
    for arm, label, offset, color, marker in [
        ("specialist", "Per-team PPO", -0.11, "#286383", "o"),
        ("moe", "Top-2 MoE", 0.11, "#a35424", "s"),
    ]:
        for i, cell in enumerate(cells):
            r = data[arm, "shared", cell]
            ax.plot([float(r["lo95"]), float(r["hi95"])], [i + offset] * 2, color=color, lw=2)
            ax.plot(
                float(r["delta_pp"]),
                i + offset,
                marker,
                color=color,
                label=label if i == 0 else None,
            )
    ax.axvline(0, color="#555", ls="--", lw=1)
    ax.set_yticks(
        range(3),
        [
            "Familiar teams\nFamiliar opponents",
            "Familiar teams\nUnseen opponents",
            "Held-out teams\nBoth roster directions",
        ],
    )
    ax.set_ylim(2.5, -0.55)
    ax.set_xlabel("Win-rate difference from shared PPO (percentage points)")
    ax.set_title("Specialists and routed experts against shared PPO", loc="left", weight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.41), ncol=2, frameon=False)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(root / "paired-wins.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
