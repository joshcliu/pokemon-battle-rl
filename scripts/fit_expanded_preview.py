"""Fit a preview head and lock a training-only fallback before held-out confirmation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from expand_preview import MARGINS, PREVIOUS, PRIOR_RUN, RUN
from preview_head import EPOCHS, FIT_SEED, audit_sweep, describe, fit, load_model
from preview_sweep import LINEUPS, play, save

POLICIES = ("original-four", "previous-head", "expanded-head", "expanded-fallback")
DEFAULT = "1234"
DEFAULT_INDEX = LINEUPS.index(DEFAULT)


@torch.inference_mode()
def prediction(model, public):
    values = model(describe(tuple(public), model.vocabulary)[None])[0].sigmoid()
    best = int(values.argmax())
    return LINEUPS[best], float(values[best]), float(values[DEFAULT_INDEX])


def gated(choice, advantage, margin):
    return choice if advantage >= margin else DEFAULT


def fold(val, teams, previews, wins, n):
    torch.set_num_threads(1)
    indices = [i for i in range(len(teams)) if i != val]
    scores = []
    for epoch, model in fit(previews, torch.tensor(wins / n, dtype=torch.float32), indices, EPOCHS):
        choice, best, default = prediction(model, previews[val])
        scores.append(
            dict(
                team=teams[val],
                epoch=epoch,
                lineup=choice,
                best_pred=best,
                default_pred=default,
                picked_wins=int(wins[val, LINEUPS.index(choice)]),
                default_wins=int(wins[val, DEFAULT_INDEX]),
                battles=n,
            )
        )
    return scores


def train(root, jobs):
    torch.set_num_threads(1)
    if (root / "selection-lock.json").exists():
        raise FileExistsError("Selection already locked")
    spec = read(root / "manifest.json")
    assert digest(PRIOR_RUN / "training-sweep.csv") == spec["prior_training_sha256"]
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    teams, previews, wins, table = audit_sweep(PRIOR_RUN, "train")
    new_teams, new_previews, new_wins, new_table = audit_sweep(root, "train")
    assert teams == spec["original_train"] and new_teams == spec["teams"]["train"]
    teams += new_teams
    previews += new_previews
    wins = np.vstack([wins, new_wins])
    n = len(spec["opponents"]) * spec["counts"]["train"]
    full_table = table + new_table
    assert len(set(teams)) == len(set(previews)) == len(teams)
    assert not set(teams) & set(spec["teams"]["test"])
    assert all(r["battles"] == n for r in full_table)
    write_csv(root / "training-sweep.csv", full_table)
    write_csv(
        root / "training-previews.csv",
        [dict(team=t, public_six="|".join(p)) for t, p in zip(teams, previews, strict=True)],
    )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fold, i, teams, previews, wins, n) for i in range(len(teams))]
        cv = [r for f in futures for r in f.result()]
    epoch = max(EPOCHS, key=lambda e: sum(r["picked_wins"] for r in cv if r["epoch"] == e))
    selected_cv = [r for r in cv if r["epoch"] == epoch]

    def fallback_wins(margin):
        return sum(
            r["picked_wins"]
            if gated(r["lineup"], r["best_pred"] - r["default_pred"], margin) != DEFAULT
            else r["default_wins"]
            for r in selected_cv
        )

    # Larger threshold breaks ties for this secondary policy.
    margin = max(MARGINS, key=lambda m: (fallback_wins(m), m))
    write_csv(root / "cv.csv", cv)
    write_csv(
        root / "fallback-cv.csv",
        [
            dict(epoch=epoch, margin=m, wins=fallback_wins(m), battles=len(teams) * n)
            for m in MARGINS
        ],
    )
    _, model = next(
        fit(
            previews, torch.tensor(wins / n, dtype=torch.float32), list(range(len(teams))), (epoch,)
        )
    )
    dest = root / "preview-head.pt"
    torch.save(dict(vocabulary=model.vocabulary, state=model.state_dict(), lineups=LINEUPS), dest)
    restored = load_model(dest)
    assert all(model.choose(p) == restored.choose(tuple(reversed(p))) for p in previews)
    save(
        root / "selection-lock.json",
        dict(
            train_teams=teams,
            heldout=spec["teams"]["test"],
            fit_seed=FIT_SEED,
            candidate_epochs=EPOCHS,
            epoch=epoch,
            fallback_margin=margin,
            head_sha256=digest(dest),
            frozen_sha256=spec["sha256"][spec["parent"]],
            training_table_sha256=digest(root / "training-sweep.csv"),
            cv_sha256=digest(root / "cv.csv"),
        ),
    )
    print(
        f"Locked {len(teams)}-team head after {epoch} epochs, fallback threshold {margin}",
        flush=True,
    )


@lru_cache
def models(root):
    torch.set_num_threads(1)
    return load_model(root / "preview-head.pt"), load_model(PREVIOUS / "preview-head.pt")


@lru_cache(maxsize=256)
def pick(root, label, margin, visible):
    if label == "original-four":
        return DEFAULT
    new, old = models(root)
    if label == "previous-head":
        return old.choose(visible)
    choice, best, default = prediction(new, visible)
    if label == "expanded-head":
        return choice
    if label == "expanded-fallback":
        return gated(choice, best - default, margin)
    raise ValueError(label)


def confirm_cell(root, spec, margin, team, i, label, family, opponent, j):
    dest = root / "confirm" / Path(team).stem / f"{label}-{family}.csv.gz"
    if dest.exists():
        return
    rows = play(
        "confirm",
        team,
        i,
        label,
        family,
        opponent,
        j,
        spec,
        root,
        partial(pick, root, label, margin),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.gz")
    write_csv(tmp, rows)
    tmp.replace(dest)


def confirm(root, jobs):
    spec, lock = read(root / "manifest.json"), read(root / "selection-lock.json")
    assert digest(root / "preview-head.pt") == lock["head_sha256"]
    assert digest(PREVIOUS / "preview-head.pt") == spec["prior_head_sha256"]
    assert all(digest(p) == h for p, h in spec["sha256"].items())
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = [
            pool.submit(
                confirm_cell,
                root,
                spec,
                lock["fallback_margin"],
                team,
                i,
                label,
                family,
                opponent,
                j,
            )
            for i, team in enumerate(spec["teams"]["test"])
            for label in POLICIES
            for j, (family, opponent) in enumerate(spec["opponents"].items())
        ]
        for n, job in enumerate(as_completed(tasks), 1):
            job.result()
            if n % 50 == 0 or n == len(tasks):
                print(f"Confirmation: {n}/{len(tasks)} cells", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "confirm"])
    p.add_argument("--root", type=Path, default=RUN)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    (train if a.stage == "train" else confirm)(a.root, a.jobs)
