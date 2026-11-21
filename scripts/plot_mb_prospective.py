"""Plot win rates and crossed 95% intervals from both prospective tests."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1] / "results"
FIRST = ROOT / "sketch5-mb-policy-iteration" / "confirmation"
SECOND = ROOT / "sketch5-mb-opponent-search" / "confirmation"


def read(folder):
    with (folder / "results.csv").open() as f:
        return {(r["split"], r["method"]): r for r in csv.DictReader(f)}


def main():
    old = json.loads((FIRST / "candidate.json").read_text())["method"]
    new = json.loads((SECOND / "candidate.json").read_text())["method"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.65), sharex=True, sharey=True, layout="constrained"
    )
    labels = (
        "Tournament teams\nvs. released BC",
        "Original Sun Room\nvs. released BC",
        "Tournament teams\nvs. independent PPO",
    )
    colors = ("#526676", "#227769", "#823155")
    markers = ("o", "s", "D")
    order = ("native", old, new)
    for round_id, (ax, folder) in enumerate(zip(axes, (FIRST, SECOND), strict=True)):
        rows = read(folder)
        policies = order[: 2 + round_id]
        for j, mode in enumerate(policies):
            offset = (j - (len(policies) - 1) / 2) * 0.20
            for i, split in enumerate(("primary", "custom", "transfer")):
                r = rows[split, mode]
                avg = float(r["win_pct"])
                lo = float(r["lo95"])
                hi = float(r["hi95"])
                y = 2 - i + offset
                ax.errorbar(
                    avg,
                    y,
                    xerr=[[avg - lo], [hi - avg]],
                    color=colors[j],
                    marker=markers[j],
                    capsize=3,
                    markersize=6,
                    lw=1.7,
                )
        ax.axvline(80, color="#737373", ls="--", lw=1.1, alpha=0.75)
        ax.set_title(
            f"Round {round_id + 1}   ·   16 reserved rosters",
            fontsize=12.5,
            fontweight="bold",
            loc="left",
        )
        ax.set_yticks([2, 1, 0], labels)
        ax.set_ylim(-0.55, 2.55)
        ax.set_xlim(35, 99)
        ax.set_xticks([40, 50, 60, 70, 80, 90], ["40%", "50%", "60%", "70%", "80%", "90%"])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.grid(axis="x", alpha=0.15)
    handles = [
        Line2D([], [], color=color, marker=marker, lw=1.7, label=label)
        for color, marker, label in zip(
            colors,
            markers,
            ("Original PPO", "First-round planner", "Second-round planner"),
            strict=True,
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Win rate on previously unseen tournament teams", fontsize=17, fontweight="bold")
    axes[1].set_xlabel("Vertical line marks the 80% target", fontsize=9)
    axes[0].set_xlabel("Intervals resample teams, policy seeds and game replicates", fontsize=9)
    dest = SECOND.parent / "prospective-win-rates.png"
    fig.savefig(dest, dpi=180, facecolor="white")
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
