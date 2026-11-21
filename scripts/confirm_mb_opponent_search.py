"""Lock, run and audit the second prospective matrix, reporting both incumbent controls."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from confirm_mb_policy_iteration import LOCK as PREVIOUS_LOCK
from confirm_mb_policy_iteration import digest
from confirm_mb_policy_iteration import freeze as check_incumbent
from mb_opponent_belief import OUT, ROOT, RUN
from mb_opponent_search import CONFIGS
from mb_opponent_search import score as challenger
from mb_policy_iteration_eval import OUT as FIRST
from mb_policy_iteration_eval import SEEDS, schedule
from mb_policy_iteration_eval import score as basic
from mb_policy_transfer import score as transfer_score
from mb_public_policy_search import play as incumbent
from report_mb_policy_iteration import interval
from vgc_posttrain import atomic_rows

SPEC = json.loads((OUT / "protocol.json").read_text())
LOCK = OUT / "confirmation" / "candidate.json"
OLD = "adaptive-belief-es"
SPLITS = ("primary", "custom", "transfer")


def read(p):
    with p.open() as handle:
        return list(csv.DictReader(handle))


def dev(mode):
    ref = {}
    wins = 0
    hashes = {}
    for s in SEEDS:
        f = OUT / "scores" / f"validation-{mode}-{s}.csv"
        hashes[str(f.relative_to(ROOT))] = digest(f)
        control = read(FIRST / "scores" / f"validation-{OLD}-{s}.csv")
        expected = {
            (i, j, d, k): (Path(a).stem, Path(b).stem, start + k, ("p1", "p2")[k % 2])
            for i, j, d, a, b, start in schedule("validation")
            for k in range(8)
        }
        for r in control:
            key = tuple(int(r[k]) for k in ("i", "j", "direction", "k"))
            ref[s, key] = (r["lineup"], r["opp_lineup"])
        for r in read(f):
            key = tuple(int(r[k]) for k in ("i", "j", "direction", "k"))
            assert expected.pop(key) == (r["own"], r["foe"], int(r["sim"]), r["side"])
            assert ref[s, key] == (r["lineup"], r["opp_lineup"])
            assert (
                r["method"] == mode
                and int(r["training_seed"]) == s
                and r["reward"] in ("-1", "0", "1")
            )
            wins += r["reward"] == "1"
        assert not expected
    return wins, hashes


def freeze(mode):
    selection = json.loads((OUT / "selection.json").read_text())
    ranking = selection["methods_in_tie_break_order"]
    assert set(ranking) == set(CONFIGS)
    records = {m: dev(m) for m in ranking}
    winner = max(ranking, key=lambda m: records[m][0])
    if mode is None:
        mode = winner
    if mode != winner:
        raise ValueError(f"The prespecified development winner is {winner}")
    n, files = records[mode]
    if n < selection["minimum_development_wins"]:
        raise ValueError(f"{mode} scored {n}/2048; required development win rate is 82%")
    check_incumbent(OLD)
    for p, h in SPEC["fresh_roster_sha256"].items():
        assert digest(ROOT / p) == h
    extra = [
        OUT / "protocol.json",
        OUT / "selection.json",
        PREVIOUS_LOCK,
        Path(__file__),
        ROOT / "simulator/public_reply_search.cjs",
        ROOT / "scripts/mb_opponent_belief.py",
        ROOT / "scripts/mb_opponent_search.py",
        ROOT / "scripts/mb_coordinate_search.py",
    ]
    if "critic" in mode:
        extra += [
            ROOT / "scripts/mb_world_value.py",
            *(OUT / "weights" / f"critic-{s}.pt" for s in SEEDS),
        ]
    assets = dict(json.loads(PREVIOUS_LOCK.read_text())["frozen_assets"])
    assets.update({str(p.relative_to(ROOT)): digest(p) for p in extra})
    lock = dict(
        method=mode,
        development_wins=n,
        development_games=2048,
        development_files=files,
        all_development_wins={m: r[0] for m, r in records.items()},
        frozen_assets=assets,
        training_seeds=SEEDS,
        simulation_bases=SPEC["simulation_bases"],
        controls=["native", OLD],
        retired_test=str(PREVIOUS_LOCK.parent.relative_to(ROOT)),
    )
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if not LOCK.exists():
        with tempfile.NamedTemporaryFile("w", dir=LOCK.parent, delete=False) as f:
            json.dump(lock, f, indent=2, sort_keys=True)
            f.write("\n")
        try:
            os.link(f.name, LOCK)
        except FileExistsError:
            pass
        finally:
            Path(f.name).unlink()
    assert json.loads(LOCK.read_text()) == lock, (
        "A different second-round candidate has already been frozen"
    )
    print("Frozen", mode, n, "/ 2048", flush=True)


def teams(split):
    a, b = SPEC["confirmation_a"], SPEC["confirmation_b"]
    return ([SPEC["custom"]], a + b, 1) if split == "custom" else (a, b, 2)


def cells(split):
    a, b, directions = teams(split)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            for d in range(directions):
                own, foe = (y, x) if d else (x, y)
                start = SPEC["simulation_bases"][split] + 10000 * i + 100 * j + 20 * d
                yield i, j, d, own, foe, start


def shard(split, mode, seed, i, j, d):
    return RUN / "confirmation-games" / split / mode / f"s{seed}-{i}-{j}-{d}.csv"


def play(task):
    split, mode, seed, cell = task
    i, j, d, own, foe, start = cell
    dest = shard(split, mode, seed, i, j, d)
    if not dest.exists():
        if mode not in ("native", OLD):
            rows = challenger((seed, cell, mode, 16), transfer=(split == "transfer"))
        else:
            rows = []
            for block in (0, 8):
                part = (i, j, d, own, foe, start + block)
                if split == "transfer":
                    batch = transfer_score((seed, mode, part))
                elif mode == OLD:
                    batch = incumbent((seed, part), method=OLD)
                else:
                    batch = basic((mode, seed, part))
                for r in batch:
                    r["k"] += block
                rows.extend(batch)
        assert len(rows) == 16
        atomic_rows(dest, rows)
    return dest


def matched(task):
    split, mode, seed, cell = task
    for policy in ("native", OLD, mode):
        play((split, policy, seed, cell))


def report():
    mode = json.loads(LOCK.read_text())["method"]
    summary = []
    matchups = []
    games = []
    for split in SPLITS:
        a, b, directions = teams(split)
        vectors = {}
        fours = {}
        for m in ("native", OLD, mode):
            x = np.full((len(SEEDS), len(a), len(b), directions, 16), np.nan)
            for si, s in enumerate(SEEDS):
                for i, j, d, own, foe, start in cells(split):
                    rows = read(shard(split, m, s, i, j, d))
                    assert len(rows) == 16
                    for k, r in enumerate(rows):
                        assert (int(r["training_seed"]), r["method"], int(r["sim"]), r["side"]) == (
                            s,
                            m,
                            start + k,
                            ("p1", "p2")[k % 2],
                        )
                        assert (r["own"], r["foe"]) == (Path(own).stem, Path(foe).stem)
                        assert (int(r["i"]), int(r["j"]), int(r["direction"]), int(r["k"])) == (
                            i,
                            j,
                            d,
                            k,
                        )
                        assert r["reward"] in ("-1", "0", "1") and int(r["turns"]) > 0
                        if split == "transfer":
                            assert r["opponent"] == "shared-2-262144-normal"
                        for col in ("lineup", "opp_lineup"):
                            assert len(r[col]) == len(set(r[col])) == 4 and set(r[col]) <= set(
                                "123456"
                            )
                        key = (si, i, j, d, k)
                        if m == "native":
                            fours[key] = [r["opp_lineup"]]
                        else:
                            assert r["opp_lineup"] == fours[key][0]
                            if m == OLD:
                                fours[key].append(r["lineup"])
                            else:
                                assert r["lineup"] == fours[key][1]
                        x[key] = int(r["reward"] == "1")
                        games.append(
                            dict(
                                split=split,
                                **{
                                    c: r[c]
                                    for c in (
                                        "method",
                                        "training_seed",
                                        "i",
                                        "j",
                                        "direction",
                                        "k",
                                        "own",
                                        "foe",
                                        "sim",
                                        "side",
                                        "lineup",
                                        "opp_lineup",
                                        "reward",
                                        "turns",
                                    )
                                },
                            )
                        )
            assert np.isfinite(x).all()
            avg = x.mean(3)
            vectors[m] = avg
            lo, hi = interval(avg)
            estimates = dict(
                split=split,
                method=m,
                wins=int(x.sum()),
                games=x.size,
                win_pct=100 * x.mean(),
                lo95=lo,
                hi95=hi,
                directions=" / ".join(
                    f"{100 * x[:, :, :, d].mean():.2f}" for d in range(directions)
                ),
                seeds=" / ".join(f"{100 * z.mean():.2f}" for z in x),
            )
            for name, control in (("baseline", "native"), ("incumbent", OLD)):
                delta = avg - vectors[control] if control in vectors else None
                estimates[name + "_gain"] = 100 * delta.mean() if delta is not None else ""
                bounds = interval(delta) if delta is not None else ("", "")
                estimates[name + "_lo95"], estimates[name + "_hi95"] = bounds
            summary.append(estimates)
            for i, j, d, own, foe, _ in cells(split):
                z = x[:, i, j, d]
                matchups.append(
                    dict(
                        split=split,
                        method=m,
                        own=Path(own).stem,
                        foe=Path(foe).stem,
                        wins=int(z.sum()),
                        games=z.size,
                    )
                )
    archive = LOCK.parent / "games.csv.gz"
    with (
        archive.with_suffix(".tmp").open("wb") as raw,
        gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf8", newline="") as f,
    ):
        writer = csv.DictWriter(f, fieldnames=list(games[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(games)
    archive.with_suffix(".tmp").replace(archive)
    assert len(games) == 52224
    atomic_rows(LOCK.parent / "results.csv", summary)
    atomic_rows(LOCK.parent / "matchups.csv", matchups)
    print(
        "Audited", len(games), "games", hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True
    )
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fixture", "freeze", "score", "report"))
    p.add_argument("--mode", choices=CONFIGS)
    p.add_argument("--jobs", default=16, type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = p.parse_args()
    if args.stage == "fixture":
        with tempfile.TemporaryDirectory(dir=RUN) as scratch:
            RUN = Path(scratch)
            own, foe = SPEC["own_train"][0], SPEC["foe_train"][0]
            for i, split in enumerate(SPLITS):
                cell = (0, 0, 0, own, foe, 666100000 + 100 * i)
                matched((split, args.mode, 2, cell))
                records = [read(shard(split, m, 2, 0, 0, 0)) for m in ("native", OLD, args.mode)]
                for native, old, new in zip(*records, strict=True):
                    for col in ("side", "sim", "opp_lineup"):
                        assert native[col] == old[col] == new[col]
                    assert old["lineup"] == new["lineup"]
                print("Training fixture verified", split, len(records[0]), flush=True)
    elif args.stage == "freeze":
        freeze(args.mode)
    else:
        mode = json.loads(LOCK.read_text())["method"]
        freeze(mode)
        if args.stage == "report":
            report()
        else:
            assert set(args.seeds) <= set(SEEDS)
            with ProcessPoolExecutor(
                args.jobs, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                # Group each matchup with its controls so both expensive pilot
                # evaluations reuse the same public matchup plan.
                for split in args.splits:
                    tasks = [(split, mode, s, c) for s in args.seeds for c in cells(split)]
                    for k, _ in enumerate(pool.map(matched, tasks)):
                        if (k + 1) % 32 == 0:
                            print(split, k + 1, "of", len(tasks), "matched groups", flush=True)
            if set(args.seeds) == set(SEEDS) and set(args.splits) == set(SPLITS):
                report()
