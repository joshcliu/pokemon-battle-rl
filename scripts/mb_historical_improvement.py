"""Re-score the improved policies on the exact battles behind the historical 59.67% figure."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import FINAL, read, schedule
from mb_policy_iteration_eval import OUT, SEEDS, decide, verify
from mb_public_preview_search import public_fixture, public_sets, read_plan
from mb_residual_preview import OUT as PREVIEW
from mb_residual_preview import ROSTERS, pilot
from report_mb_long_ppo import bootstrap
from vgc_posttrain import Match, atomic_rows, batch_obs


@torch.no_grad()
def play(task):
    seed, mode, entry = task
    _, _, own, foe, games, _, direction = entry
    public = public_fixture(own, foe) if mode == "search-es-trained" else None
    picked = read_plan(own, public, seed) if public is not None else None
    model, match = pilot(seed), Match()
    rows = []
    for sim, side in games:
        if picked:

            def picker(view):
                assert public_sets(view) == public
                return picked["selected"]

            match.reset(own, foe, sim, side, preview_picker=picker)
        else:
            match.reset(own, foe, sim, side, preview="native")
        torch.manual_seed(sim + 700_000_000)
        done = False
        while not done:
            a = decide(model, batch_obs(match.obs()), "es-trained", seed, match.view)
            done, r = match.step(a)
        rows.append(
            dict(
                method=mode,
                training_seed=seed,
                cell=direction,
                own=Path(own).stem,
                foe=Path(foe).stem,
                seed=sim,
                side=side,
                lineup=match.lineup,
                opponent_lineup=match.opponent.lineup,
                reward=r,
                turns=match.battle.turn,
            )
        )
    return rows


def report():
    ref = {
        (int(r["training_seed"]), r["cell"], int(r["seed"])): r
        for r in read(PREVIEW / "games.csv.gz")
        if r["variant"] == "native"
    }
    assert len(ref) == 4096 and sum(r["reward"] == "1" for r in ref.values()) == 2444
    groups = [[Path(t).stem for t in ROSTERS[k]] for k in ("own_test", "foe_test")]
    table = {}
    totals = {}
    for mode in ("native", "es-trained", "search-es-trained"):
        if mode == "native":
            rows = list(ref.values())
        else:
            paths = [OUT / "historical" / f"{mode}-{s}.csv" for s in SEEDS]
            if not all(p.exists() for p in paths):
                continue
            rows = [r for p in paths for r in read(p)]
        x = np.full((4, 4, 8, 2, 16), np.nan)
        for r in rows:
            seed = int(r["training_seed"])
            sim = int(r["seed"])
            cell = r["cell"]
            d = int(cell == "reverse")
            own, foe = (r["foe"], r["own"]) if d else (r["own"], r["foe"])
            i, j = groups[0].index(own), groups[1].index(foe)
            refrow = ref[seed, cell, sim]
            assert all(r[k] == refrow[k] for k in ("own", "foe", "side", "opponent_lineup"))
            if mode != "search-es-trained":
                assert r["lineup"] == refrow["lineup"]
            idx = (SEEDS.index(seed), i, j, d, sim % 100)
            assert np.isnan(x[idx])
            x[idx] = r["reward"] == "1"
        assert np.isfinite(x).all()
        table[mode] = x.mean(3)
        totals[mode] = (int(x.sum()), [100 * x[:, :, :, d].mean() for d in (0, 1)])
    effects = {f"{k} minus native": x - table["native"] for k, x in table.items() if k != "native"}
    intervals = bootstrap({**table, **effects})
    rows = []
    for k, x in {**table, **effects}.items():
        lo, hi = intervals[k]
        rows.append(
            dict(
                method_or_contrast=k,
                wins=totals[k][0] if k in totals else "",
                games=4096,
                estimate=x.mean() * 100,
                lo95=lo,
                hi95=hi,
                seed_rates=" / ".join(f"{z.mean() * 100:.2f}" for z in x),
                directions=" / ".join(f"{z:.2f}" for z in totals[k][1]) if k in totals else "",
            )
        )
    atomic_rows(OUT / "historical.csv", rows)
    print(rows, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    verify()
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for mode in ("es-trained", "search-es-trained"):
            for seed in SEEDS:
                f = OUT / "historical" / f"{mode}-{seed}.csv"
                if f.exists():
                    continue
                entries = list(schedule("moe", seed, FINAL, "normal", diverse=True))
                rows = [
                    r for grp in pool.map(play, [(seed, mode, e) for e in entries]) for r in grp
                ]
                assert len(rows) == 1024
                atomic_rows(f, rows)
                print(mode, seed, len(rows), sum(r["reward"] == 1 for r in rows), flush=True)
            report()
