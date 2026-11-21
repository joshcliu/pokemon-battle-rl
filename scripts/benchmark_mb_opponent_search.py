"""Measure deploy-time costs on identical public positions from training-only games."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from mb_midgame_plan import choose
from mb_opponent_belief import OUT, RUN, private_prior
from mb_opponent_search import CONFIGS, act
from mb_policy_iteration_eval import RUN as OLD_RUN
from mb_policy_iteration_eval import SEEDS
from mb_public_policy_search import optimize
from mb_public_preview_search import key_for, public_fixture
from mb_residual_preview import pilot
from mb_rollout_branches import pairing
from vgc_posttrain import Match, atomic_rows, batch_obs


@torch.no_grad()
def main():
    samples = []
    preview = []
    for s in SEEDS:
        model = pilot(s)
        for g in range(2):
            own, foe, seat = pairing(s, 4900 + g)
            public = public_fixture(own, foe)
            key = key_for(own, public, s)
            adaptive_key = hashlib.sha256((key + "-specialize-v1").encode()).hexdigest()
            cached = (OLD_RUN / "public-adaptation" / f"{adaptive_key}.json").exists()
            base_cached = (OLD_RUN / "public-preview" / f"{key}.json").exists()
            prior_cached = (RUN / "public-prior" / f"{key}.json").exists()
            start = time.perf_counter()
            plan = optimize(own, public, s)
            middle = time.perf_counter()
            prior = private_prior(own, public, s)
            end = time.perf_counter()
            preview.append(
                dict(
                    seed=s,
                    own=Path(own).stem,
                    foe=Path(foe).stem,
                    old_cached=cached,
                    base_cached=base_cached,
                    prior_cached=prior_cached,
                    planning_seconds=middle - start,
                    opponent_prior_seconds=end - middle,
                )
            )
            match = Match()
            match.reset(own, foe, 137000000 + s * 100 + g, seat, lineup=plan["selected"])
            done = False
            step = 0
            while not done:
                for mode in ("incumbent", *CONFIGS):
                    start = time.perf_counter()
                    ob = batch_obs(match.obs())
                    if mode == "incumbent":
                        command = choose(model, ob, match.view, s, False)[0]
                    else:
                        act(model, ob, match.view, s, prior, mode)
                    elapsed = time.perf_counter() - start
                    if step > 0:
                        samples.append(
                            dict(mode=mode, seed=s, game=g, request=step, ms=elapsed * 1000)
                        )
                done, _ = match.step(command)
                step += 1
            print(s, g, step, flush=True)
    rows = []
    for mode in ("incumbent", *CONFIGS):
        t = [r["ms"] for r in samples if r["mode"] == mode]
        rows.append(
            dict(
                mode=mode,
                games=2 * len(SEEDS),
                matched_decisions=len(t),
                median_ms=np.median(t),
                p95_ms=np.percentile(t, 95),
            )
        )
    atomic_rows(OUT / "move-latency.csv", rows)
    atomic_rows(OUT / "preview-latency.csv", preview)
    print(rows, flush=True)


if __name__ == "__main__":
    main()
