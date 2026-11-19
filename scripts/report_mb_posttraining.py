"""Audit all pilot outcomes and compute win rates and paired roster/seed bootstrap intervals."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vgc_posttrain import REPORT, ROOT, RUN, atomic_rows


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def archive(path, records):
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def interval(changes):
    # Crossed bootstrap of own roster, opposing roster, model seed and individual games within seat.
    # Identical resampled indices for both policies preserve the paired comparison.
    rng = np.random.default_rng(141592)
    n, a, b, k = changes.shape
    draws = 6000
    train = rng.integers(n, size=(draws, n, 1, 1, 1))
    own = rng.integers(a, size=(draws, 1, a, 1, 1))
    foe = rng.integers(b, size=(draws, 1, 1, b, 1))
    games = np.arange(k) % 2 + 2 * rng.integers(k // 2, size=(draws, 1, a, b, k))
    estimate = changes[train, own, foe, games].mean(axis=(1, 2, 3, 4))
    return np.percentile(estimate * 100, [2.5, 97.5])


def main():
    spec = json.loads((REPORT / "protocol.json").read_text())
    seeds = spec["training_seeds"]
    for file, digest in spec["hashes"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest
    cells = [o + "_" + f for o in ("own_train", "own_test") for f in ("foe_train", "foe_test")]
    arms = [
        "bc",
        "bc-native",
        "history",
        "native",
        "flat",
        "structured",
        "native-preview",
        "preview-ppo",
        "preview-grpo",
    ]
    arrays = {}
    games = []
    board_keys = {}
    foe_lineups = {}
    rates = []
    seed_rates = []
    roster_rates = []
    for arm in arms:
        blocks = []
        for s in [None] if arm in ("bc", "bc-native") else seeds:
            label = arm if s is None else f"{arm}-{s}"
            data = rows(REPORT / f"matrix-{label}.csv")
            assert len(data) == 1024 and len({r["seed"] for r in data}) == 1024
            block = []
            for cell in cells:
                part = sorted([r for r in data if r["cell"] == cell], key=lambda r: int(r["seed"]))
                assert len(part) == 256
                own_key, foe_key = cell.split("_foe_")
                foe_key = "foe_" + foe_key
                assert {r["own"] for r in part} == {Path(t).stem for t in spec[own_key]}
                assert {r["foe"] for r in part} == {Path(t).stem for t in spec[foe_key]}
                keys = [(r["own"], r["foe"], r["seed"], r["side"]) for r in part]
                assert keys == board_keys.setdefault(cell, keys)
                assert [r["opponent_lineup"] for r in part] == foe_lineups.setdefault(
                    cell, [r["opponent_lineup"] for r in part]
                )
                assert all(
                    r["side"] == ("p1", "p2")[i % 2] and int(r["reward"]) in (-1, 0, 1)
                    for i, r in enumerate(part)
                )
                for j in range(0, 256, 8):
                    group = part[j : j + 8]
                    assert len({(r["own"], r["foe"]) for r in group}) == 1
                    roster_rates.append(
                        dict(
                            arm=arm,
                            seed=s,
                            cell=cell,
                            own=group[0]["own"],
                            foe=group[0]["foe"],
                            wins=sum(r["reward"] == "1" for r in group),
                            games=len(group),
                        )
                    )
                wins = np.array([r["reward"] == "1" for r in part], dtype=float).reshape(4, 8, 8)
                seed_rates.append(
                    dict(
                        arm=arm,
                        seed=s,
                        cell=cell,
                        wins=int(wins.sum()),
                        games=wins.size,
                        win_pct=100 * wins.mean(),
                    )
                )
                block.append(wins)
            blocks.append(block)
            games.extend(dict(label=label, **r) for r in data)
        arrays[arm] = np.array(blocks)
        for i, cell in enumerate(cells):
            v = arrays[arm][:, i]
            low, high = interval(v)
            rates.append(
                dict(
                    arm=arm,
                    cell=cell,
                    wins=int(v.sum()),
                    games=v.size,
                    win_pct=100 * v.mean(),
                    lo95=low,
                    hi95=high,
                )
            )
    archive(REPORT / "matrix-games.csv.gz", games)
    atomic_rows(REPORT / "win-rates.csv", rates)
    atomic_rows(REPORT / "per-seed.csv", seed_rates)
    atomic_rows(REPORT / "per-matchup.csv", roster_rates)
    preview_choices = []
    for arm in ("preview-ppo", "preview-grpo"):
        for s in seeds:
            for own in spec["own_test"]:
                name = Path(own).stem
                plays = [
                    r
                    for r in games
                    if r["label"] == f"{arm}-{s}"
                    and r["cell"] == "own_test_foe_test"
                    and r["own"] == name
                ]
                choices = {r["foe"]: r["lineup"] for r in plays}
                assert len(choices) == 8 and all(r["lineup"] == choices[r["foe"]] for r in plays)
                preview_choices.append(
                    dict(
                        arm=arm,
                        seed=s,
                        own=name,
                        distinct_lineups=len(set(choices.values())),
                        opponents=len(choices),
                        selections=json.dumps(choices, sort_keys=True),
                    )
                )
    atomic_rows(REPORT / "preview-choices.csv", preview_choices)
    prefs = rows(REPORT / "target-preferences.csv")
    targets = []
    for arm in ("bc", "native", "flat", "structured"):
        group = [r for r in prefs if r["policy"].split("-")[0] == arm]
        pair_count = len({(r["board"], r["hi"], r["lo"]) for r in group})
        targets.append(
            dict(
                arm=arm,
                cases=pair_count,
                boards=len({r["board"] for r in group}),
                prob_higher_multiplier_pct=100
                * np.mean([float(r["p_higher_multiplier"]) for r in group]),
            )
        )
    atomic_rows(REPORT / "target-summary.csv", targets)
    changes = []
    contrasts = [(x, "bc") for x in arms if x != "bc"] + [
        ("flat", "native"),
        ("structured", "native"),
        ("structured", "flat"),
        ("native-preview", "native"),
        ("native-preview", "bc-native"),
        ("preview-grpo", "preview-ppo"),
        ("preview-grpo", "bc-native"),
        ("preview-ppo", "bc-native"),
    ]
    for arm, control in contrasts:
        for i, cell in enumerate(cells):
            v = arrays[arm][:, i] - arrays[control][:, i]
            lo, hi = interval(v)
            changes.append(
                dict(
                    arm=arm,
                    control=control,
                    cell=cell,
                    delta_pp=100 * v.mean(),
                    lo95=lo,
                    hi95=hi,
                    seeds_improved=int((v.mean(axis=(1, 2, 3)) > 0).sum()),
                    seeds=len(v),
                )
            )
    atomic_rows(REPORT / "paired-effects.csv", changes)
    reverse_rows, reciprocal = [], []
    reciprocal_arrays = {}
    for arm in ["bc", "bc-native", "native-preview", "preview-ppo", "preview-grpo"]:
        parts = []
        for s in [None] if arm in ("bc", "bc-native") else seeds:
            label = arm if s is None else f"{arm}-{s}"
            rs = sorted(rows(REPORT / f"reverse-{label}.csv"), key=lambda r: int(r["seed"]))
            expected = board_keys["own_test_foe_test"]
            assert len(rs) == len(expected)
            assert all(
                (r["foe"], r["own"], r["seed"], "p1" if r["side"] == "p2" else "p2") == key
                for r, key in zip(rs, expected, strict=True)
            )
            assert [r["opponent_lineup"] for r in rs] == foe_lineups.setdefault(
                "reverse", [r["opponent_lineup"] for r in rs]
            )
            parts.append(np.array([r["reward"] == "1" for r in rs], dtype=float).reshape(4, 8, 8))
            reverse_rows.extend(dict(label=label, **r) for r in rs)
        fw = arrays[arm][:, -1]
        rev = np.stack(parts)
        both = (fw + rev) / 2
        reciprocal_arrays[arm] = both
        low, high = interval(both)
        reciprocal.append(
            dict(
                arm=arm,
                forward_wins=int(fw.sum()),
                reverse_wins=int(rev.sum()),
                games_per_direction=fw.size,
                forward_pct=100 * fw.mean(),
                reverse_pct=100 * rev.mean(),
                balanced_pct=100 * both.mean(),
                balanced_lo95=low,
                balanced_hi95=high,
            )
        )
    for r in reciprocal:
        delta = reciprocal_arrays[r["arm"]] - reciprocal_arrays["bc-native"]
        lo, hi = interval(delta)
        r.update(delta_native_bc_pp=100 * delta.mean(), lo95=lo, hi95=hi)
    atomic_rows(REPORT / "reciprocal.csv", reciprocal)
    archive(REPORT / "reciprocal-games.csv.gz", reverse_rows)
    baseline = rows(REPORT / "same-team-bc.csv")
    old = ROOT / "results/sketch5-vgcbench-bc/games.csv.gz"
    with gzip.open(old, "rt") as f:
        original = {
            (r["roster"], r["seed"], r["side"]): r["opponent_lineup"]
            for r in csv.DictReader(f)
            if r["family"] == "bc-learned"
        }
    devs = {
        "bc-fixed": [r for r in baseline if r["selector"] == "fixed"],
        "bc-native": [r for r in baseline if r["selector"] == "native"],
    }
    for s in seeds:
        devs[f"native-{s}"] = rows(REPORT / f"dev-native-{s}.csv")
    dev = []
    devgames = []
    fixed = None
    for label, rs in devs.items():
        rs = sorted(rs, key=lambda r: int(r["seed"]))
        assert len(rs) == 768 and all(
            r["opponent_lineup"] == original[r["foe"], r["seed"], r["side"]] for r in rs
        )
        v = np.array([r["reward"] == "1" for r in rs], dtype=float).reshape(1, 1, 12, 64)
        if fixed is None:
            fixed = v
        lo, hi = interval(v - fixed)
        dev.append(
            dict(
                label=label,
                wins=int(v.sum()),
                games=v.size,
                win_pct=100 * v.mean(),
                delta_fixed_pp=100 * (v - fixed).mean(),
                lo95=lo,
                hi95=hi,
            )
        )
        devgames.extend(dict(label=label, **r) for r in rs)
    nat = [
        np.array(
            [r["reward"] == "1" for r in sorted(devs[f"native-{s}"], key=lambda r: int(r["seed"]))],
            dtype=float,
        )
        for s in seeds
    ]
    nat = np.array(nat).reshape(len(seeds), 1, 12, 64)
    lo, hi = interval(nat - fixed)
    dev.append(
        dict(
            label="native-PPO-mean",
            wins=int(nat.sum()),
            games=nat.size,
            win_pct=100 * nat.mean(),
            delta_fixed_pp=100 * (nat - fixed).mean(),
            lo95=lo,
            hi95=hi,
        )
    )
    atomic_rows(REPORT / "development-rates.csv", dev)
    archive(REPORT / "development-games.csv.gz", devgames)
    ckpts = []
    logs = []
    previews = []
    for arm in ("native", "flat", "structured", "preview-ppo", "preview-grpo"):
        for s in seeds:
            root = RUN / f"{arm}-{s}"
            preview = arm.startswith("preview-")
            file = root / ("head.pt" if preview else "policy.pt")
            ck = torch.load(file, map_location="cpu", weights_only=True)
            ckpts.append(
                dict(
                    arm=arm,
                    seed=s,
                    path=str(file.relative_to(ROOT)),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    parameters=ck.get("parameters", ""),
                )
            )
            if preview:
                rec = rows(root / "rollouts.csv")
                assert len(rec) == 384
                for g in range(32):
                    p = [r for r in rec if int(r["group"]) == g]
                    assert (
                        len(p) == 12
                        and len({r["foe"] for r in p}) == 1
                        and len({r["own"] for r in p}) == 1
                    )
                    for k in (0, 1):
                        assert len({r["opponent_lineup"] for r in p[k::2]}) == 1
                previews.extend(dict(arm=arm, training_seed=s, **r) for r in rec)
            else:
                rec = rows(root / "training.csv")
                assert int(rec[-1]["steps"]) == 4096
                logs.extend(dict(arm=arm, seed=s, **r) for r in rec)
                assert {r["own"] for r in rows(root / "episodes.csv")} == {
                    Path(t).stem for t in spec["own_train"]
                }
                assert {r["foe"] for r in rows(root / "episodes.csv")} == {
                    Path(t).stem for t in spec["foe_train"]
                }
    atomic_rows(REPORT / "checkpoints.csv", ckpts)
    atomic_rows(REPORT / "training.csv", logs)
    archive(REPORT / "preview-rollouts.csv.gz", previews)
    print(
        "Audited",
        len(games),
        "matrix games,",
        len(devgames),
        "development games and",
        len(previews),
        "preview training games.",
        flush=True,
    )


if __name__ == "__main__":
    main()
