"""Show roster-balanced transfer and paired routing effects at both PPO budgets."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results/sketch5-mb-long-ppo"


def main():
    with (OUT / "scores.csv").open() as f:
        scores = {
            (r["arm"], int(r["steps"])): float(r["win_pct"])
            for r in csv.DictReader(f)
            if r["variant"] == "normal" and r["cell"] == "heldout"
        }
    with (OUT / "effects.csv").open() as f:
        effects = {(r["arm"], r["effect"]): r for r in csv.DictReader(f) if r["cell"] == "heldout"}
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(10.2, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]}
    )
    labels = dict(
        shared="Shared PPO",
        moe="4 experts · top 2",
        moe1="4 experts · top 1",
        moe8="8 experts · top 2",
    )
    colors = dict(shared="#444a50", moe="#007b76", moe1="#a45228", moe8="#6555a4")
    for a in labels:
        ax.plot(
            [0, 1], [scores[a, t] for t in (65536, 262144)], "-o", color=colors[a], label=labels[a]
        )
    ax.axhline(50, ls=":", lw=1, color="#757575")
    ax.set_xticks([0, 1], ["65,536", "262,144"])
    ax.set(xlabel="Training decisions", ylabel="Win rate (%)")
    ax.set_title("Held-out teams", loc="left", weight="bold")
    ax.legend(fontsize=8, frameon=False)
    for y, a in enumerate(("moe", "moe8", "moe1")):
        for j, (t, off) in enumerate(((65536, -0.13), (262144, 0.13))):
            r = effects[a, f"identity at {t}"]
            pos = y + off
            bx.plot([float(r["lo95"]), float(r["hi95"])], [pos, pos], lw=1.6, color=colors[a])
            bx.scatter(
                float(r["delta_pp"]),
                pos,
                marker="o" if j == 0 else "s",
                facecolor="white" if j == 0 else colors[a],
                edgecolor=colors[a],
                s=35,
                zorder=3,
            )
    bx.axvline(0, ls="--", lw=1, color="#777")
    bx.set_yticks(range(3), [labels[a] for a in ("moe", "moe8", "moe1")])
    bx.set_ylim(2.45, -0.5)
    bx.set_xlabel("Normal minus rotated win rate (points)")
    bx.set_title("Expert identity · paired 95% intervals", loc="left", weight="bold")
    bx.plot([], [], "o", mfc="white", mec="#444", label="65k")
    bx.plot([], [], "s", color="#444", label="262k")
    bx.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    for panel in (ax, bx):
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "heldout-routing.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    main()
