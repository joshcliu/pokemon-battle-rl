"""Audit and summarize matched 2×2 reward experiments for native PPO and AWAC."""

import csv
import gzip
import hashlib
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from eval_mb_rewards import schedule
from mb_reward_factorial import OUT, RUN, SPEC, verify_assets
from poke_env.teambuilder import Teambuilder
from report_mb_posttraining import interval
from vgc_posttrain import ROOT, atomic_rows

SEEDS = SPEC["training_seeds"]
ALGOS = SPEC["optimizers"]
MODES = SPEC["rewards"]
CELLS = ["familiar", "forward", "reverse"]
RAW_CELLS = ["own_train_foe_train", "own_test_foe_test", "reciprocal"]


def read(file):
    with file.open() as f:
        return list(csv.DictReader(f))


def archive(file, records):
    with gzip.open(file, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def low_high(values):
    low, high = interval(values)
    return dict(lo95=low, hi95=high)


def audit_training():
    checkpoints, updates, episodes = [], [], []
    hashes = {}
    paired = {}
    for a, m, s in product(ALGOS, MODES, SEEDS):
        root = RUN / f"{a}-{m}-{s}"
        file = root / "policy.pt"
        ckpt = torch.load(file, map_location="cpu", weights_only=True)
        assert (ckpt["algorithm"], ckpt["reward"], ckpt["seed"], ckpt["steps"]) == (
            a,
            m,
            s,
            SPEC["steps"],
        )
        assert ckpt["first_rollout"] == hashes.setdefault(s, ckpt["first_rollout"])
        if a == "ppo" and m == "none":
            old = torch.load(
                ROOT / f"runs/sketch5-mb-posttraining-pilot/native-{s}/policy.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert set(old["state"]) == set(ckpt["state"]) and all(
                torch.equal(old["state"][k], v) for k, v in ckpt["state"].items()
            )
        meta = dict(algorithm=a, mode=m, training_seed=s)
        checkpoints.append(
            dict(
                **meta,
                path=str(file.relative_to(ROOT)),
                sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                first_256_actions_outcomes_sha256=ckpt["first_rollout"],
            )
        )
        logs = read(root / "training.csv")
        assert [int(r["steps"]) for r in logs] == list(range(256, SPEC["steps"] + 1, 256))
        assert (sum(float(r["hp_abs"]) for r in logs) > 0) == (m in ("hp", "both"))
        assert (sum(float(r["ko_abs"]) for r in logs) > 0) == (m in ("alive", "both"))
        if a == "awac":
            assert (
                float(logs[0]["old_replay_fraction"]) == 0
                and float(logs[-1]["old_replay_fraction"]) > 0.75
            )
            assert all(float(r["weight_sd"]) > 0 for r in logs)
        updates.extend(
            {**meta, **{"q_loss": "", "weight_sd": "", "old_replay_fraction": ""}, **r}
            for r in logs
        )
        rs = read(root / "episodes.csv")
        assert len(rs) == int(logs[-1]["games"])
        for r in rs:
            key = s, r["seed"]
            value = r["own"], r["foe"], r["side"]
            assert value == paired.setdefault(key, value)
        assert {r["own"] for r in rs} == {Path(p).stem for p in SPEC["own_train"]}
        assert {r["foe"] for r in rs} == {Path(p).stem for p in SPEC["foe_train"]}
        episodes.extend(dict(**meta, **r) for r in rs)
    atomic_rows(OUT / "checkpoints.csv", checkpoints)
    atomic_rows(OUT / "training.csv", updates)
    archive(OUT / "training-games.csv.gz", episodes)


def score_games():
    result, raw, by_seed, by_pair, previews = [], [], [], [], {}
    arms = [("bc", "none"), *product(ALGOS, MODES)]
    scores, swept_scores = {}, {}
    for a, m in arms:
        arrays = defaultdict(list)
        stats = defaultdict(list)
        for s in [None] if a == "bc" else SEEDS:
            label = "bc" if s is None else f"{a}-{m}-{s}"
            rows = read(RUN / "scores" / f"{label}.csv")
            expected = {
                (cell, sim_seed): (Path(own).stem, Path(foe).stem, side)
                for _, _, own, foe, seeds, _, cell in schedule(label)
                for sim_seed, side in seeds
            }
            assert len(rows) == len(expected) and {
                (r["cell"], int(r["seed"])) for r in rows
            } == set(expected)
            for r in rows:
                key = r["cell"], int(r["seed"])
                assert r["policy"] == label and r["selector"] == "native" and r["mode"] == "sample"
                assert expected[key] == (r["own"], r["foe"], r["side"])
                assert 0 <= float(r["foe_hp_remaining"]) <= 1 and int(r["foe_alive"]) in range(5)
                assert (r["lineup"], r["opponent_lineup"]) == previews.setdefault(
                    key, (r["lineup"], r["opponent_lineup"])
                )
            raw.extend(dict(algorithm=a, mode_name=m, training_seed=s, **r) for r in rows)
            for cell, raw_cell in zip(CELLS, RAW_CELLS, strict=True):
                rs = sorted(
                    [r for r in rows if r["cell"] == raw_cell], key=lambda r: int(r["seed"])
                )
                assert len(rs) == 256
                wins = np.asarray([r["reward"] == "1" for r in rs], dtype=float).reshape(4, 8, 8)
                arrays[cell].append(wins)
                stats[cell].extend(rs)
                by_seed.append(
                    dict(
                        algorithm=a,
                        mode=m,
                        seed=s,
                        cell=cell,
                        wins=int(wins.sum()),
                        games=wins.size,
                        win_pct=100 * wins.mean(),
                    )
                )
                for i in range(0, len(rs), 8):
                    batch = rs[i : i + 8]
                    assert len({(r["own"], r["foe"]) for r in batch}) == 1
                    by_pair.append(
                        dict(
                            algorithm=a,
                            mode=m,
                            seed=s,
                            cell=cell,
                            own=batch[0]["own"],
                            foe=batch[0]["foe"],
                            wins=sum(r["reward"] == "1" for r in batch),
                            games=len(batch),
                        )
                    )
        arrays = {c: np.stack(v) for c, v in arrays.items()}
        arrays["balanced"] = (arrays["forward"] + arrays["reverse"]) / 2
        stats["balanced"] = stats["forward"] + stats["reverse"]
        scores[a, m] = arrays["balanced"]
        swept = [
            np.asarray(
                [r["reward"] == "-1" and r["foe_alive"] == "4" for r in stats[direction]],
                dtype=float,
            ).reshape(-1, 4, 8, 8)
            for direction in ("forward", "reverse")
        ]
        swept_scores[a, m] = (swept[0] + swept[1]) / 2

        for cell, v in arrays.items():
            rs = stats[cell]
            loss = [r for r in rs if r["reward"] == "-1"]
            sweeps = sum(r["foe_alive"] == "4" for r in loss)
            result.append(
                dict(
                    algorithm=a,
                    mode=m,
                    cell=cell,
                    games=len(rs),
                    wins=sum(r["reward"] == "1" for r in rs),
                    win_pct=100 * v.mean(),
                    **low_high(v),
                    losses=len(loss),
                    draws=sum(r["reward"] == "0" for r in rs),
                    sweeps=sweeps,
                    sweeps_pct_of_games=100 * sweeps / len(rs),
                    sweeps_pct_of_losses=100 * sweeps / len(loss),
                    foe_alive_per_loss=np.mean([int(r["foe_alive"]) for r in loss]),
                    foe_hp_fraction_per_loss=np.mean([float(r["foe_hp_remaining"]) for r in loss]),
                )
            )
    archive(OUT / "games.csv.gz", raw)
    atomic_rows(OUT / "scores.csv", result)
    atomic_rows(OUT / "per-seed.csv", by_seed)
    atomic_rows(OUT / "per-pair.csv", by_pair)
    atomic_rows(OUT / "effects.csv", factorial_effects(scores))
    atomic_rows(OUT / "sweep-effects.csv", factorial_effects(swept_scores))
    return by_pair, len(raw)


def factorial_effects(scores):
    contrasts = []

    def contrast(name, algo, v):
        contrasts.append(
            dict(
                algorithm=algo,
                effect=name,
                delta_pp=100 * v.mean(),
                **low_high(v),
                seeds_positive=int((v.mean(axis=(1, 2, 3)) > 0).sum()),
                seeds=len(v),
            )
        )

    for a in ALGOS:
        n, h, k, b = [scores[a, m] for m in MODES]
        contrast("HP main effect", a, ((h - n) + (b - k)) / 2)
        contrast("opponent KO main effect", a, ((k - n) + (b - h)) / 2)
        contrast("HP x KO interaction", a, b - h - k + n)
        for m in MODES:
            contrast(m + " minus frozen BC", a, scores[a, m] - scores["bc", "none"])
            if m != "none":
                contrast(m + " minus none", a, scores[a, m] - n)
    diffs = [scores["awac", m] - scores["ppo", m] for m in MODES]
    for m, d in zip(MODES, diffs, strict=True):
        contrast(m + " | AWAC minus PPO", "cross", d)
    contrast("AWAC minus PPO averaged over rewards", "cross", np.mean(diffs, axis=0))
    n, h, k, b = diffs
    contrast("AWAC x HP interaction", "cross", ((h - n) + (b - k)) / 2)
    contrast("AWAC x KO interaction", "cross", ((k - n) + (b - h)) / 2)
    contrast("AWAC x HP x KO interaction", "cross", b - h - k + n)
    return contrasts


def team_matrix(rows):
    pool = {
        k: [Path(p).stem for p in SPEC[k]]
        for k in ("own_train", "foe_train", "own_test", "foe_test")
    }
    layout = [
        ("Training and familiar evaluation", "own_train", "foe_train"),
        ("Held-out forward", "own_test", "foe_test"),
        ("Held-out reverse", "foe_test", "own_test"),
    ]
    lines = [
        "# Exact team assignments",
        "",
        "| Schedule | Our policy | Frozen BC opponent |",
        "|---|---|---|",
    ]
    for title, o, f in layout:
        lines.append(f"| {title} | {', '.join(pool[o])} | {', '.join(pool[f])} |")
    lines += [
        "",
        "Training samples the two rosters independently. Evaluation crosses every listed pairing. In training, the learner leads with slots 1–2 and brings 3–4 in back. The opponent uses its published preview. In evaluation, both sides use that frozen preview. Each checkpoint plays eight battles per roster pairing per direction, four in each simulator seat. The ordered lineup digits for both sides appear in [every game record](games.csv.gz).",
        "",
        "| Pool | Team | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, pool_name in [
        ("own_train", "Learner training"),
        ("foe_train", "Opponent training"),
        ("own_test", "Learner held-out"),
        ("foe_test", "Opponent held-out"),
    ]:
        for p in SPEC[k]:
            path = ROOT / p
            mons = Teambuilder.parse_showdown_team(path.read_text().strip())
            if p.startswith(".cache/"):
                url = f"https://github.com/cameronangliss/vgc-bench/blob/{SPEC['source']}/teams/reg_mb/{path.name}"
            else:
                url = "../../" + p
            lines.append(
                "| "
                + " | ".join(
                    [
                        pool_name,
                        f"[{path.stem}]({url})",
                        *[mon.species or mon.nickname for mon in mons],
                    ]
                )
                + " |"
            )
    for a, m in [("bc", "none"), *product(ALGOS, MODES)]:
        lines += ["", f"## {a.upper()} · {m}", ""]
        for cell, (title, o, f) in zip(CELLS, layout, strict=True):
            matches = defaultdict(lambda: [0, 0])
            for r in rows:
                if (r["algorithm"], r["mode"], r["cell"]) == (a, m, cell):
                    pair = matches[r["own"], r["foe"]]
                    pair[0] += r["wins"]
                    pair[1] += r["games"]
            lines += [
                "",
                f"### {title.split(' and ')[-1].capitalize()} — wins / games",
                "",
                "| Learner ↓ / opponent → | " + " | ".join(pool[f]) + " |",
                "|---|" + "---:|" * len(pool[f]),
            ]
            for own in pool[o]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            own,
                            *[f"{matches[own, foe][0]}/{matches[own, foe][1]}" for foe in pool[f]],
                        ]
                    )
                    + " |"
                )
    (OUT / "team-matrix.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    verify_assets()
    audit_training()
    by_pair, n = score_games()
    team_matrix(by_pair)
    replays = read(RUN / "scores/reproduced.csv")
    assert len(replays) == 2 * (1 + len(ALGOS) * len(MODES) * len(SEEDS))
    atomic_rows(OUT / "reproduced.csv", replays)
    print(f"Audited {n} games, {len(replays)} replay reproductions and 32 checkpoints.")
