"""Freeze the first qualifying policy before scoring any of the reserved rosters."""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from mb_policy_iteration_eval import OUT, RUN, SEEDS, SPEC, schedule, verify
from mb_residual_preview import DIVERSE_RUN
from report_mb_policy_iteration import interval
from vgc_posttrain import ROOT, atomic_rows

DEST = OUT / "confirmation"
LOCK = DEST / "candidate.json"
SPLITS = ("primary", "custom", "transfer")
BASES = {"primary": 640_000_000, "custom": 650_000_000, "transfer": 660_000_000}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def dev_record(mode):
    seen = set()
    total = 0
    files = {}
    for seed in SEEDS:
        path = OUT / "scores" / f"validation-{mode}-{seed}.csv"
        expected = {
            (i, j, d, k): (Path(a).stem, Path(b).stem, start + k, ("p1", "p2")[k % 2])
            for i, j, d, a, b, start in schedule("validation")
            for k in range(8)
        }
        for r in rows(path):
            key = tuple(int(r[k]) for k in ("i", "j", "direction", "k"))
            assert expected.pop(key) == (r["own"], r["foe"], int(r["sim"]), r["side"])
            assert r["method"] == mode and int(r["training_seed"]) == seed
            assert r["reward"] in ("-1", "0", "1")
            seen.add((seed, *key))
            total += r["reward"] == "1"
        assert not expected
        files[str(path.relative_to(ROOT))] = digest(path)
    assert len(seen) == 2048
    return total, files


def fingerprint(mode):
    from eval_mb_long_ppo import FINAL, path_for

    from pokemon_rl.vgcbench import ZIP

    battle = mode.removeprefix("search-")
    scripts = [
        "vgc_posttrain",
        "mb_policy_iteration_eval",
        "mb_policy_transfer",
        "mb_mechanics_es",
        "mb_residual_preview",
        "mb_roster_ppo",
        "eval_mb_long_ppo",
        "mb_policy",
        "report_mb_policy_iteration",
    ]
    if mode.startswith(("search-", "adaptive-")):
        scripts.append("mb_public_preview_search")
    if battle in (
        "belief-es",
        "wide-belief-es",
        "multistep-es",
        "endgame-es",
        "adaptive-belief-es",
    ):
        scripts += ["mb_endgame_plan", "mb_endgame_value", "mb_opening_planner"]
    if battle in ("belief-es", "wide-belief-es", "multistep-es", "adaptive-belief-es"):
        scripts.append("mb_midgame_plan")
    if battle == "multistep-es":
        scripts.append("mb_multistep_value")
    if battle == "context-ppo":
        scripts.append("mb_context_ppo")
    if battle.startswith(("joint-ppo-", "distilled-ppo-")):
        scripts.append("mb_joint_mechanics_ppo")
    if battle.startswith("adaptive-"):
        scripts += ["mb_public_policy_search", "mb_offline_lineup_value"]
    files = [OUT / "protocol.json", ZIP, path_for("shared", 2, FINAL)]
    files.extend(ROOT / "scripts" / f"{name}.py" for name in scripts)
    files += [
        Path(__file__).resolve(),
        ROOT / "simulator/public_endgame.cjs",
        ROOT / "simulator/bridge.cjs",
        ROOT / "src/pokemon_rl/vgcbench.py",
        ROOT / "src/pokemon_rl/battle.py",
        ROOT / "src/pokemon_rl/league.py",
    ]
    for seed in SEEDS:
        files += [
            DIVERSE_RUN / f"moe-{seed}" / "policy.pt",
            RUN / "es" / f"seed-{seed}" / "weights.json",
        ]
        if battle in ("belief-es", "wide-belief-es", "multistep-es", "adaptive-belief-es"):
            files.append(RUN / "midgame" / f"seed-{seed}" / "value.npy")
        if battle == "endgame-es":
            files.append(RUN / "endgame" / f"seed-{seed}" / "value.npy")
        if battle == "multistep-es":
            files.append(RUN / "multistep" / f"seed-{seed}" / "horizon.npz")
        if battle == "context-ppo":
            files.append(RUN / "context-ppo" / f"seed-{seed}" / "progress.pt")
        if battle.startswith("adaptive-"):
            files.append(RUN / "lineup-value" / f"seed-{seed}.npz")
        if battle.startswith(("joint-ppo-", "distilled-ppo-")):
            arm, steps = battle.rsplit("-", 1)
            files.append(RUN / arm / f"seed-{seed}" / f"policy-{steps}.pt")
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def freeze(mode):
    supported = {
        "es-trained",
        "adaptive-es",
        "adaptive-belief-es",
        "belief-es",
        "multistep-es",
        "wide-belief-es",
        "endgame-es",
        "context-ppo",
        "joint-ppo-16384",
        "joint-ppo-65536",
        "distilled-ppo-16384",
    }
    if mode.removeprefix("search-") not in supported:
        raise ValueError("This policy requires its own validated confirmation entry point")
    wins, dev = dev_record(mode)
    if wins * 5 < 2048 * 4:
        raise ValueError(f"{mode} won {wins}/2048 on development; the prespecified gate is 80%")
    verify()
    proposed = dict(
        method=mode,
        development_wins=wins,
        development_games=2048,
        seeds=SEEDS,
        games_per_direction=16,
        simulator_seed_bases=BASES,
        development_files=dev,
        frozen_assets=fingerprint(mode),
    )
    DEST.mkdir(parents=True, exist_ok=True)
    if not LOCK.exists():
        with tempfile.NamedTemporaryFile(dir=DEST, mode="w", delete=False) as tmp:
            json.dump(proposed, tmp, sort_keys=True, indent=2)
            tmp.write("\n")
        try:
            os.link(tmp.name, LOCK)
        except FileExistsError:
            pass
        finally:
            Path(tmp.name).unlink()
    frozen = json.loads(LOCK.read_text())
    if frozen != proposed:
        raise RuntimeError(
            "The confirmation pool is already committed to a different policy or version"
        )
    print("Frozen", mode, wins, "/ 2048", flush=True)


def dimensions(split):
    if split == "custom":
        return [SPEC["custom"]], SPEC["confirmation_a"] + SPEC["confirmation_b"], 1
    return SPEC["confirmation_a"], SPEC["confirmation_b"], 2


def cells(split):
    a, b, directions = dimensions(split)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            for d in range(directions):
                for block in (0, 8):
                    own, foe = (x, y) if not d else (y, x)
                    start = BASES[split] + 10_000 * i + 100 * j + 20 * d + block
                    yield (i, j, d, own, foe, start), block


def play(task):
    split, mode, seed, cell, block = task
    i, j, d, own, foe, start = cell
    dest = DEST / "games" / split / mode / f"s{seed}-{i}-{j}-{d}-{block}.csv"
    if not dest.exists():
        if split == "transfer":
            from mb_policy_transfer import score

            game_rows = score((seed, mode, cell))
        elif mode.startswith("adaptive-"):
            from mb_public_policy_search import play as adaptive_play

            game_rows = adaptive_play((seed, cell), method=mode)
        elif mode.startswith("search-"):
            from mb_public_preview_search import score

            game_rows = score((seed, cell, mode.removeprefix("search-")))
        else:
            from mb_policy_iteration_eval import score

            game_rows = score((mode, seed, cell))
        assert len(game_rows) == 8
        for r in game_rows:
            r["k"] += block
        atomic_rows(dest, game_rows)
    records = rows(dest)
    assert len(records) == 8
    for k, r in enumerate(records):
        assert (int(r["training_seed"]), r["method"], int(r["k"])) == (seed, mode, k + block)
        assert (r["own"], r["foe"], r["side"], int(r["sim"])) == (
            Path(own).stem,
            Path(foe).stem,
            ("p1", "p2")[k % 2],
            start + k,
        )
        assert r["reward"] in ("-1", "0", "1") and int(r["turns"]) > 0
        if split == "transfer":
            assert r["opponent"] == "shared-2-262144-normal"
    return records


def evaluate(jobs):
    frozen = json.loads(LOCK.read_text())
    mode = frozen["method"]
    freeze(mode)
    output = []
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in SPLITS:
            a, b, directions = dimensions(split)
            baseline = None
            fours = {}
            for label in ("native", mode):
                x = np.full((len(SEEDS), len(a), len(b), directions, 16), np.nan)
                tasks = [(split, label, s, c, block) for s in SEEDS for c, block in cells(split)]
                for group in pool.map(play, tasks):
                    for r in group:
                        key = (
                            SEEDS.index(int(r["training_seed"])),
                            *(int(r[k]) for k in ("i", "j", "direction", "k")),
                        )
                        assert np.isnan(x[key])
                        if label == "native":
                            fours[key] = r["lineup"], r["opp_lineup"]
                        else:
                            assert r["opp_lineup"] == fours[key][1]
                            if not label.startswith(("search-", "adaptive-")):
                                assert r["lineup"] == fours[key][0]
                        x[key] = int(r["reward"] == "1")
                assert np.isfinite(x).all()
                mean = x.mean(3)
                if label == "native":
                    baseline = mean
                lo, hi = interval(mean)
                delta = mean - baseline
                dl, dh = interval(delta)
                output.append(
                    dict(
                        split=split,
                        method=label,
                        wins=int(x.sum()),
                        games=x.size,
                        win_pct=100 * mean.mean(),
                        lo95=lo,
                        hi95=hi,
                        delta_pp=100 * delta.mean(),
                        diff_lo95=dl,
                        diff_hi95=dh,
                        directions=" / ".join(
                            f"{100 * x[:, :, :, d].mean():.2f}" for d in range(directions)
                        ),
                        seeds=" / ".join(f"{100 * q.mean():.2f}" for q in mean),
                    )
                )
                atomic_rows(DEST / "results.csv", output)
                print(split, label, int(x.sum()), x.size, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("eligibility", "freeze", "score"))
    p.add_argument("--mode")
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    if args.stage == "score":
        evaluate(args.jobs)
    elif not args.mode:
        p.error("Supply --mode for the development gate")
    elif args.stage == "freeze":
        freeze(args.mode)
    else:
        w, _ = dev_record(args.mode)
        print(args.mode, w, "/ 2048; eligible:", w * 5 >= 2048 * 4)
