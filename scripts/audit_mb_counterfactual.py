"""Quantify training label noise and opportunities within matched public decision states."""

import numpy as np
from mb_policy_iteration_eval import OUT, RUN, SEEDS
from vgc_posttrain import atomic_rows


def main():
    summary = []
    for seed in SEEDS:
        counts = []
        for block in range(40):
            p = RUN / "branches" / f"seed-{seed}" / f"{block:04}.npz"
            with np.load(p) as chunk:
                for ret, valid, depth in zip(
                    chunk["returns"], chunk["valid"], chunk["depth"], strict=True
                ):
                    y = (ret[valid] == 1).astype(float)
                    gap = y.max(0) - y[0]
                    rng = np.random.default_rng([1303, seed, block, len(counts)])
                    cross = []
                    for k in (0, 1):
                        best = rng.choice(np.flatnonzero(y[:, k] == y[:, k].max()))
                        cross.append(y[best, 1 - k] - y[0, 1 - k])
                    counts.append(
                        (
                            len(y) * 2,
                            depth,
                            gap.mean(),
                            (y.max(0) != y.min(0)).mean(),
                            (y[:, 0] != y[:, 1]).mean(),
                            np.mean(cross),
                        )
                    )
        c = np.asarray(counts)
        summary.append(
            dict(
                seed=seed,
                states=len(c),
                continuations=int(c[:, 0].sum()),
                mean_prefix_decisions=c[:, 1].mean(),
                oracle_gain_within_sample=c[:, 2].mean(),
                fraction_matched_replicates_with_action_effect=c[:, 3].mean(),
                repeat_disagreement_for_same_action=c[:, 4].mean(),
                gain_select_one_repeat_score_other_same_hidden_world=c[:, 5].mean(),
            )
        )
    atomic_rows(OUT / "branch-diagnostics.csv", summary)
    print(
        "Audited",
        sum(r["states"] for r in summary),
        "multi-action training roots and",
        sum(r["continuations"] for r in summary),
        "continuations",
        flush=True,
    )


if __name__ == "__main__":
    main()
