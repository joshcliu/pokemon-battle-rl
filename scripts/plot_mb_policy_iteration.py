"""Plot crossed win-rate intervals for the main battle and preview policy ablations."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/sketch5-mb-policy-iteration"
METHODS = {
    "native": "Sampled PPO",
    "map": "Most probable PPO move",
    "es-initial": "Hand-set mechanics",
    "es-trained": "Learned mechanics",
    "search-map": "Preview search + PPO",
    "search-es-trained": "Preview search + learned mechanics",
}


def main():
    with (OUT / "development.csv").open() as f:
        scores = {(r["split"], r["method"]): r for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    ys = np.arange(len(METHODS))[::-1]
    for ax, (split, title) in zip(
        axes,
        [
            ("validation", "Released BC"),
            ("transfer", "Independent PPO"),
            ("custom-dev", "Sun Room vs. BC"),
        ],
        strict=True,
    ):
        ax.axvline(80, color="#af4b36", ls="--", lw=1)
        for y, method in zip(ys, METHODS, strict=True):
            if (split, method) not in scores:
                continue
            r = scores[split, method]
            estimate, lo, hi = (float(r[k]) for k in ("win_pct", "lo95", "hi95"))
            seeds = [float(s) for s in r["seeds_pct"].split("/")]
            ax.scatter(seeds, [y + 0.13] * len(seeds), s=8, color="#95abb7", edgecolors="none")
            color = "#11605b" if method == "search-es-trained" else "#334f65"
            ax.errorbar(
                estimate,
                y,
                xerr=[[estimate - lo], [hi - estimate]],
                fmt="o",
                ms=5,
                capsize=2.5,
                color=color,
                elinewidth=1.5,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([40, 50, 60, 70, 80])
        ax.set_xlim(32, 83)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.tick_params(labelsize=9, axis="both", length=0, pad=6)
    axes[0].set_yticks(ys, list(METHODS.values()))
    fig.supxlabel("Win rate (%)", fontsize=10, y=0.10)
    fig.text(
        0.5,
        0.025,
        "Dots above each estimate show the four training seeds. Bars show 95% intervals across rosters, seeds and games.",
        ha="center",
        fontsize=9,
        color="#455562",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.99))
    fig.savefig(OUT / "policy-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
