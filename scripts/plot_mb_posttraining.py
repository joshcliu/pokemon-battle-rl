"""Plot win rates across the four roster-transfer conditions and the reciprocal pilot comparison."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPORT = Path(__file__).resolve().parents[1] / "results/sketch5-mb-posttraining-pilot"


def main():
    with (REPORT / "win-rates.csv").open() as f:
        values = {(r["arm"], r["cell"]): float(r["win_pct"]) for r in csv.DictReader(f)}
    cells = ["own_train_foe_train", "own_train_foe_test", "own_test_foe_train", "own_test_foe_test"]
    arms = [
        "history",
        "bc",
        "native",
        "flat",
        "structured",
        "bc-native",
        "native-preview",
        "preview-ppo",
        "preview-grpo",
    ]
    names = [
        "Previous history PPO",
        "Released BC · fixed four",
        "BC → battle PPO",
        "Battle PPO + flat head",
        "Battle PPO + action scorer",
        "Released BC · native preview",
        "Battle PPO + native preview",
        "Frozen BC + PPO preview",
        "Frozen BC + GRPO preview",
    ]
    mat = np.array([[values[a, c] for c in cells] for a in arms])
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.8, 5.55), gridspec_kw={"width_ratios": [1.7, 1]}, layout="constrained"
    )
    cmap = LinearSegmentedColormap.from_list(
        "rates", ["#b66c50", "#ead7ce", "#f9f8f2", "#c2dcd5", "#539f94"]
    )
    a.imshow(mat, aspect="auto", vmin=20, vmax=80, cmap=cmap)
    for i in range(len(arms)):
        for j in range(len(cells)):
            a.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=9.5)
    a.set_yticks(range(len(arms)), names, fontsize=9)
    a.set_xticks(
        range(4), ["Familiar\nFamiliar", "Familiar\nNew", "New\nFamiliar", "New\nNew"], fontsize=9
    )
    a.set_xlabel("Our roster / opposing roster", labelpad=10)
    a.set_title(
        "Win rate against the frozen BC bot", loc="left", weight="bold", fontsize=12, pad=12
    )
    a.tick_params(length=0)
    for y in (0.5, 4.5):
        a.axhline(y, color="white", lw=2.3)
    for spine in a.spines.values():
        spine.set_visible(False)
    with (REPORT / "reciprocal.csv").open() as f:
        table = {r["arm"]: r for r in csv.DictReader(f)}
    order = ["bc", "native-preview", "preview-ppo", "preview-grpo"]
    labels = ["BC, fixed four", "Battle PPO, BC preview", "BC, PPO preview", "BC, GRPO preview"]
    for k, key in enumerate(order):
        r = table[key]
        delta = float(r["delta_native_bc_pp"])
        lo = float(r["lo95"])
        hi = float(r["hi95"])
        b.plot([lo, hi], [k, k], color="#427a72", lw=2)
        b.plot(delta, k, "o", color="#17584e", ms=6)
    b.axvline(0, ls="--", lw=1, color="#555555")
    b.set_yticks(range(4), labels, fontsize=9)
    b.invert_yaxis()
    b.margins(y=0.28)
    b.set_xlabel(
        "Win-rate change (percentage points)\nrelative to BC with its native preview", fontsize=9
    )
    b.set_title("After swapping rosters", loc="left", weight="bold", fontsize=12, pad=12)
    b.grid(axis="x", alpha=0.2)
    b.spines[["top", "right", "left"]].set_visible(False)
    b.tick_params(axis="y", length=0)
    fig.savefig(REPORT / "transfer-and-reciprocal.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
