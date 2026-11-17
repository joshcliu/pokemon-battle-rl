"""Select, freeze, and evaluate best responses to a multi-roster Transformer."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_long_rewards import write_csv
from fixed_league import name, run
from report_fixed_league import export_weights, games, interval
from screen_league import fit_router
from train_counterleague import ORIGINAL, ROOT, TEAM, TEAMS, save
from train_fixed_league import ARCHIVE

ALL_TEAMS = (*TEAMS, TEAM)
PUBLIC = ROOT / "results/sketch5-counterleague"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def tasks_for(label, learner, team, other, other_team, family, base, n, roster=None):
    condition = {"candidate": label, "roster": Path(roster or team).stem, "family": family}
    return [
        (condition, learner, team, other, other_team, base, start, min(40, n - start))
        for start in range(0, n, 40)
    ]


def field(root, out, models, base, n, jobs):
    pool = read(root / "adapt-pool.json")
    matchups = [(t, opp) for t, opponents in pool.items() for opp in opponents]
    tasks = []
    for label, cp in models.items():
        for i, team in enumerate(TEAMS):
            for j, (other_team, opp) in enumerate(matchups):
                family = f"{Path(other_team).stem}/{Path(opp).stem}"
                tasks.extend(
                    tasks_for(
                        label,
                        cp,
                        team,
                        opp,
                        other_team,
                        family,
                        base + (i * len(matchups) + j) * 1000,
                        n,
                    )
                )
    scores = run(out, jobs, tasks)
    save(out / "models.json", models)
    return scores


def choose_defender(root, jobs):
    models = {
        f"adapt-{seed}-{step}": name(root / f"adapt/{seed}/train/checkpoint-{step}.pt")
        for seed in (291, 293, 295, 297)
        for step in (32768, 65536)
    }
    scores = field(root, root / "defender-selection", models, 42000000, 32, jobs)
    wins = Counter()
    for r in scores:
        wins[r["candidate"]] += r["wins"]
    best = min(models, key=lambda k: (-wins[k], k))
    save(
        root / "defender.json",
        {
            "label": best,
            "model": models[best],
            "sha256": digest(models[best]),
            "original": ORIGINAL,
            "original_sha256": digest(ORIGINAL),
            "training_teams": TEAMS,
        },
    )


def matchups(out, assignments, pool, base, n, jobs, fixed_team=False, sample_opponent=False):
    tasks = []
    for label, (cp, teams) in assignments.items():
        for t in teams:
            tasks.extend(
                tasks_for(
                    label,
                    cp,
                    TEAM if fixed_team else t,
                    pool[t][0],
                    t,
                    "fixed" if fixed_team else "mirror",
                    base + ALL_TEAMS.index(t) * 10000,
                    n,
                    roster=t,
                )
            )
    if sample_opponent:
        tasks = [(*task, None, True) for task in tasks]
    scores = run(out, jobs, tasks)
    save(out / "models.json", {k: v[0] for k, v in assignments.items()})
    return scores


def baselines(root, jobs):
    adapted = read(root / "defender.json")["model"]
    pool = {t: [adapted if t in TEAMS else ORIGINAL] for t in ALL_TEAMS}
    tasks = []
    for i, t in enumerate(ALL_TEAMS):
        foe = pool[t][0]
        for label, actor in (
            ("self", foe),
            ("old-champion", ORIGINAL),
            ("tactical", "tactical"),
            ("damage", "damage"),
        ):
            tasks.extend(tasks_for(label, actor, t, foe, t, "mirror", 44000000 + 10000 * i, 160))
    run(root / "baseline", jobs, tasks)
    matchups(
        root / "fixed-baseline",
        {"old-champion": (ORIGINAL, ALL_TEAMS)},
        pool,
        45000000,
        160,
        jobs,
        fixed_team=True,
    )


def sampled_diagnostic(public, jobs):
    lock = read(public / "locked.json")
    candidates = read(public / "candidates.json")
    pool = read(public / "opponents.json")
    for fixed, out in [(False, "sampled-mirrors"), (True, "sampled-fixed")]:
        models = {
            k: (v, ALL_TEAMS) for k, v in candidates.items() if (k == lock["fixed_team"]) == fixed
        }
        matchups(
            public / out,
            models,
            pool,
            60000000 + 50000 * fixed,
            240,
            jobs,
            fixed_team=fixed,
            sample_opponent=True,
        )


def screen(root, jobs):
    pool = read(root / "shared-pool.json")
    assignments = {}
    for t in ALL_TEAMS:
        team = Path(t).stem
        for seed in (331, 337, 347, 349):
            for steps in (16384, 32768):
                cp = root / f"counter/{team}-{seed}/train/checkpoint-{steps}.pt"
                assignments[f"{team}-{seed}-{steps}"] = (name(cp), [t])
    for s in (353, 359, 367):
        for steps in (32768, 65536):
            cp = root / f"shared/{s}/train/checkpoint-{steps}.pt"
            assignments[f"shared-{s}-{steps}"] = (name(cp), ALL_TEAMS)
    scores = matchups(root / "counter-selection", assignments, pool, 46000000, 80, jobs)
    totals = Counter()
    for r in scores:
        if r["candidate"].startswith("shared"):
            totals[r["candidate"]] += r["wins"]
    router = fit_router(root, ["counter-selection"], root / "counter-validation")
    shortlist = {k: assignments[k][0] for k in sorted(totals, key=lambda k: (-totals[k], k))[:3]}
    shortlist["opening-router"] = router
    save(root / "shortlist.json", shortlist)
    fixed = {
        f"fixed-{seed}-{steps}": (
            name(root / f"fixed-counter/{seed}/train/checkpoint-{steps}.pt"),
            ALL_TEAMS,
        )
        for seed in (373, 379, 383)
        for steps in (32768, 65536)
    }
    matchups(root / "fixed-selection", fixed, pool, 47000000, 80, jobs, fixed_team=True)


def weights(model):
    if model.endswith(".json"):
        router = read(ROOT / model)
        return sorted({router["default"], *router["routes"].values()})
    return [model]


def choose_final_base(root):
    # Avoid even accidental physical simulator-seed reuse, including ancestor training.
    used = set()
    dirs = [root, ARCHIVE, ROOT / "runs/sketch5-fixed-league-v1"]
    for directory in dirs:
        for log in directory.rglob("episodes.jsonl"):
            with log.open() as f:
                used.update(json.loads(line)["simulator_seed"] for line in f)
    base = 52000000
    while any(base + 10000 * i + j in used for i in range(2 * len(ALL_TEAMS)) for j in range(800)):
        base += 100000
    return base


def validate(root, jobs):
    pool = read(root / "shared-pool.json")
    models = read(root / "shortlist.json")
    scores = matchups(
        root / "counter-validation",
        {k: (cp, ALL_TEAMS) for k, cp in models.items()},
        pool,
        48000000,
        240,
        jobs,
    )

    def rank(label):
        wins = [r["wins"] for r in scores if r["candidate"] == label]
        return -min(wins), -sum(wins), label

    best = min(models, key=rank)
    single = min((k for k in models if models[k].endswith(".pt")), key=rank)
    cp = read(root / "defender.json")["model"]
    assert cp == pool[TEAMS[0]][0] and all(pool[t] == [cp] for t in TEAMS)
    assert pool[TEAM] == [ORIGINAL]
    chosen = {label: models[label] for label in (best, single)}
    fixed_models = read(root / "fixed-selection/models.json")
    fixed_scores = matchups(
        root / "fixed-validation",
        {k: (v, ALL_TEAMS) for k, v in fixed_models.items()},
        pool,
        49000000,
        240,
        jobs,
        fixed_team=True,
    )
    scores.extend(fixed_scores)
    fixed_best = min(fixed_models, key=rank)
    chosen[fixed_best] = fixed_models[fixed_best]
    files = {cp, ORIGINAL, *ALL_TEAMS, *chosen.values()}
    for model in chosen.values():
        files.update(weights(model))
    lock = {
        "primary": best,
        "single": single,
        "fixed_team": fixed_best,
        "candidates": chosen,
        "pool": pool,
        "sha256": {p: digest(p) for p in sorted(files)},
        "games_per_team": 800,
        "base": choose_final_base(root),
        "team_order": ALL_TEAMS,
    }
    save(root / "locked.json", lock)


def confirm(root, public, jobs):
    lock = read(root / "locked.json")
    assert all(digest(p) == h for p, h in lock["sha256"].items())
    assert lock["base"] == choose_final_base(root)
    assets = public / "weights"
    assets.mkdir(parents=True, exist_ok=False)
    cp = read(root / "defender.json")["model"]
    models = lock["candidates"]
    sources = sorted({p for model in models.values() for p in weights(model)})
    mapping = {source: assets / f"counter-{i}.pt" for i, source in enumerate(sources, 1)}
    mapping[cp] = assets / "defender.pt"
    records = [export_weights(src, dest) for src, dest in mapping.items()]
    write_csv(public / "weights.csv", records)
    exported = {}
    for label, model in models.items():
        if model.endswith(".json"):
            spec = read(ROOT / model)
            spec = {
                "routes": {k: name(mapping[v]) for k, v in spec["routes"].items()},
                "default": name(mapping[spec["default"]]),
            }
            save(public / "router.json", spec)
            exported[label] = name(public / "router.json")
        else:
            exported[label] = name(mapping[model])
    pool = {t: [ORIGINAL if t == TEAM else name(mapping[cp])] for t in ALL_TEAMS}
    save(public / "opponents.json", pool)
    save(public / "candidates.json", exported)
    save(public / "locked.json", lock)
    dest = public / "confirmation"
    n = lock["games_per_team"]
    fixed = lock["fixed_team"]
    matchups(
        dest,
        {k: (model, ALL_TEAMS) for k, model in exported.items() if k != fixed},
        pool,
        lock["base"],
        n,
        jobs,
    )
    fixed_dest = public / "fixed-confirmation"
    matchups(
        fixed_dest,
        {fixed: (exported[fixed], ALL_TEAMS)},
        pool,
        lock["base"] + 50000,
        n,
        jobs,
        fixed_team=True,
    )
    all_rows = games(dest) + games(fixed_dest)
    summaries = []
    for label, model in exported.items():
        base = lock["base"] + (50000 if label == fixed else 0)
        expected = {
            (t, base + 10000 * i + j, "p1" if j % 2 == 0 else "p2")
            for i, t in enumerate(ALL_TEAMS)
            for j in range(n)
        }
        rows = [r for r in all_rows if r["candidate"] == label]
        assert len(rows) == len(expected)
        assert {(r["opponent_team"], int(r["seed"]), r["side"]) for r in rows} == expected
        assert all(
            r["team"] == (TEAM if label == fixed else r["opponent_team"])
            and r["opponent"] == pool[r["opponent_team"]][0]
            and r["learner"] == model
            for r in rows
        )
        assert all(r["outcome"] in ("win", "loss", "tie", "cutoff") for r in rows)
        for group, subset in [
            ("all", rows),
            ("four-adapted", [r for r in rows if r["opponent_team"] != TEAM]),
            *[(Path(t).stem, [r for r in rows if r["opponent_team"] == t]) for t in ALL_TEAMS],
        ]:
            summaries.append({"candidate": label, "group": group, **interval(subset)})
    write_csv(public / "summary.csv", summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "defender",
            "zero-shot",
            "calibrate",
            "baselines",
            "screen",
            "validate",
            "confirm",
            "sampled",
        ],
    )
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--public", type=Path, default=PUBLIC)
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    root = a.runs.resolve()
    if a.stage == "defender":
        choose_defender(root, a.jobs)
    elif a.stage == "zero-shot":
        field(root, root / "zero-shot", {"original": ORIGINAL}, 43000000, 96, a.jobs)
    elif a.stage == "calibrate":
        field(
            root,
            root / "calibration",
            {"adapted": read(root / "defender.json")["model"]},
            43000000,
            96,
            a.jobs,
        )
    elif a.stage == "baselines":
        baselines(root, a.jobs)
    elif a.stage == "screen":
        screen(root, a.jobs)
    elif a.stage == "validate":
        validate(root, a.jobs)
    elif a.stage == "sampled":
        sampled_diagnostic(a.public.resolve(), a.jobs)
    else:
        confirm(root, a.public.resolve(), a.jobs)
