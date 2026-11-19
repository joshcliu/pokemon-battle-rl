"""Plot the reciprocal reward factorial and its paired uncertainty intervals."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

OUT = Path(__file__).resolve().parents[1] / "results/sketch5-mb-reward-factorial"


def main():
    with (OUT / "scores.csv").open() as f:
        rates = {
            (r["algorithm"], r["mode"]): float(r["win_pct"])
            for r in csv.DictReader(f)
            if r["cell"] == "balanced"
        }
    with (OUT / "effects.csv").open() as f:
        effects = {(r["algorithm"], r["effect"]): r for r in csv.DictReader(f)}
    fig, (left, middle, right) = plt.subplots(
        1, 3, figsize=(11.7, 3.65), width_ratios=[1, 1, 1.95], layout="constrained"
    )
    limit = max(5, max(abs(v - 50) for v in rates.values()) + 1)
    for ax, algo in [(left, "ppo"), (middle, "awac")]:
        z = np.array(
            [[rates[algo, "none"], rates[algo, "hp"]], [rates[algo, "alive"], rates[algo, "both"]]]
        )
        ax.imshow(
            z,
            cmap="BrBG",
            norm=TwoSlopeNorm(vmin=50 - limit, vcenter=50, vmax=50 + limit),
            aspect="equal",
        )
        for i, j in np.ndindex(z.shape):
            ax.text(
                j,
                i,
                f"{z[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if abs(z[i, j] - 50) > 0.76 * limit else "#172126",
            )
        ax.set_title(algo.upper(), weight="bold", loc="left")
        ax.set_xticks([0, 1], ["Off", "On"])
        ax.set_xlabel("HP feedback")
        ax.set_yticks([0, 1], ["Off", "On"])
        ax.set_ylabel("Opposing-KO feedback")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    names = ["HP main effect", "opponent KO main effect", "HP x KO interaction"]
    for arm, offset, color, marker in [
        ("ppo", -0.12, "#00756b", "o"),
        ("awac", 0.12, "#955537", "s"),
    ]:
        for i, n in enumerate(names):
            r = effects[arm, n]
            x = float(r["delta_pp"])
            lo = float(r["lo95"])
            hi = float(r["hi95"])
            right.plot([lo, hi], [i + offset, i + offset], color=color)
            right.plot(x, i + offset, marker, color=color, label=arm.upper() if i == 0 else None)
    right.axvline(0, ls="--", lw=1, color="#666")
    right.set_yticks(range(3), ["HP", "Opposing KOs", "Interaction"])
    right.invert_yaxis()
    right.margins(y=0.35)
    right.tick_params(axis="y", length=0)
    right.set_xlabel("Paired win-rate change (percentage points)")
    right.set_title("Factor effects with 95% intervals", loc="left", weight="bold", fontsize=12)
    right.spines[["top", "left", "right"]].set_visible(False)
    right.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.savefig(OUT / "reward-factorial.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
