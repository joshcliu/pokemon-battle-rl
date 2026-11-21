"""Measure warmed move selection and simulation budgets on the training roster pool."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from mb_policy_iteration_eval import OUT, SEEDS, SPEC, decide, verify
from mb_residual_preview import pilot
from vgc_posttrain import Match, atomic_rows, batch_obs


@torch.no_grad()
def main(games):
    verify()
    match = Match()
    samples = []
    for mode in ("es-trained", "belief-es", "wide-belief-es", "multistep-es"):
        for seed in SEEDS:
            model = pilot(seed)
            for g in range(games + 1):
                own, foe = SPEC["own_train"][seed + g], SPEC["foe_train"][g]
                match.reset(
                    own, foe, 965_000_000 + 100 * seed + g, ("p1", "p2")[g % 2], preview="native"
                )
                done = False
                while not done:
                    start = time.perf_counter()
                    obs = batch_obs(match.obs())
                    a = decide(model, obs, mode, seed, match.view)
                    duration = (time.perf_counter() - start) * 1000
                    if g:
                        samples.append(dict(method=mode, seed=seed, ms=duration))
                    done, _ = match.step(a)
    out = []
    for mode in dict.fromkeys(r["method"] for r in samples):
        times = [r["ms"] for r in samples if r["method"] == mode]
        out.append(
            dict(
                method=mode,
                training_only_games=games * len(SEEDS),
                decisions=len(times),
                median_ms=np.median(times),
                p95_ms=np.percentile(times, 95),
                max_ms=max(times),
            )
        )
    atomic_rows(OUT / "move-latency.csv", out)
    print(out, flush=True)


@torch.no_grad()
def preview():
    from mb_public_preview_search import plan, public_fixture

    verify()
    out = []
    for seed in SEEDS:
        own, foe = SPEC["own_train"][seed], SPEC["foe_train"][seed + 1]
        public = public_fixture(own, foe)
        pilot(seed)
        start = time.perf_counter()
        result = plan(own, public, seed)
        duration = time.perf_counter() - start
        assert sum(result["counts"].values()) == 64
        out.append(
            dict(
                seed=seed,
                own=Path(own).stem,
                foe=Path(foe).stem,
                simulated_games=64,
                uncached_seconds=duration,
            )
        )
        print(seed, duration, flush=True)
    atomic_rows(OUT / "preview-latency.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games-per-seed", type=int, default=4)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    if args.preview:
        preview()
    else:
        main(args.games_per_seed)
