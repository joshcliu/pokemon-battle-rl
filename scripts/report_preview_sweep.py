"""Audit all lineup choices and summarize matched held-out outcomes, including roster variation."""

import argparse
import csv
import gzip
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_challenger_transfer import paired, score
from analyze_long_rewards import write_csv
from evaluate_challenger_transfer import digest, read
from preview_head import load_model
from preview_sweep import BASE, COUNT, LINEUPS, OUT, RUN, read_games, save

NAMES = ("Meowstic-F", "Kommo-o", "Indeedee-M", "Sneasler", "Kingambit", "Charizard")
POLICIES = ("learned", "fixed", "default", "random", "sweep-winner")


def pair_name(code):
    return " + ".join(NAMES[int(c) - 1] for c in code)


def archive(path, sources):
    with gzip.open(path, "wt", newline="") as dest:
        writer = None
        for src in sources:
            rows = read_games(src)
            if writer is None:
                writer = csv.DictWriter(dest, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
            writer.writerows(rows)


def plot(path, effects):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [r for r in effects if r["split"] == "test" and r["baseline"] == "fixed"]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for label, offset, color, marker, legend in (
        ("fixed", -0.13, "#165b6c", "o", "vs. training-selected fixed lineup"),
        ("default", 0.13, "#9c4828", "s", "vs. original first four"),
    ):
        for i, r in enumerate(
            r for r in effects if r["split"] == "test" and r["baseline"] == label
        ):
            delta = r["delta_pp"]
            ax.errorbar(
                delta,
                i + offset,
                xerr=[[delta - r["low95_pp"]], [r["high95_pp"] - delta]],
                fmt=marker,
                capsize=3,
                color=color,
                ms=7 if r["team"] == "overall" else 5,
                label=legend if i == 0 else None,
            )
    ax.set(
        yticks=range(len(values)),
        yticklabels=[r["team"] if r["team"] != "overall" else "Seven-team average" for r in values],
        xlabel="Preview head advantage (percentage points, paired 95% interval)",
        title="MC461 on the seven held-out opposing rosters",
    )
    ax.axvline(0, color="gray", lw=1)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.invert_yaxis()
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def report(root, out):
    out.mkdir(parents=True, exist_ok=True)
    spec, lock = read(root / "manifest.json"), read(root / "selection-lock.json")
    assert lock["head_sha256"] == digest(root / "preview-head.pt")
    assert lock["training_table_sha256"] == digest(root / "training-sweep.csv")
    assert spec["teams"]["train"] == lock["train_teams"]
    assert not set(spec["teams"]["train"]) & set(spec["teams"]["test"])
    winners = read(root / "sweep-winners.json")
    head = load_model(root / "preview-head.pt")
    tables, leads, team_results, scores, effects = [], [], [], [], []
    previews = {}
    methods = defaultdict(list)
    all_teams = spec["teams"]["train"] + spec["teams"]["test"]
    for i, t in enumerate(all_teams):
        team = Path(t).stem
        stage = "train" if t in spec["teams"]["train"] else "test"
        for label in POLICIES:
            directory = root / "confirm" / team
            rows = []
            for j, (family, bot) in enumerate(spec["opponents"].items()):
                games = read_games(directory / f"{label}-{family}.csv.gz")
                n = COUNT["confirm"]
                base = BASE["confirm"] + (i * len(spec["opponents"]) + j) * 1000
                assert len(games) == n and {(int(r["seed"]), r["side"]) for r in games} == {
                    (base + k, "p1" if k % 2 == 0 else "p2") for k in range(n)
                }
                assert all(
                    r["team"] == spec["team"]
                    and r["learner"] == spec["parent"]
                    and r["opponent_team"] == t
                    and r["opponent"] == bot
                    and r["family"] == family
                    and r["candidate"] == label
                    and r["roster"] == team
                    for r in games
                )
                rows.extend(games)
            visible = {r["preview"] for r in rows}
            assert len(visible) == 1
            public = tuple(visible.pop().split("|"))
            assert previews.setdefault(team, public) == public and tuple(sorted(public)) == public
            chosen = {r["lineup"] for r in rows}
            assert chosen <= set(LINEUPS)
            expected = {
                "learned": head.choose(public),
                "fixed": lock["fixed"],
                "default": "1234",
                "sweep-winner": winners[t],
            }
            if label != "random":
                assert chosen == {expected[label]}
            else:
                for r in rows:
                    rng = np.random.default_rng(np.random.SeedSequence([int(r["seed"]), 7329]))
                    assert r["lineup"] == LINEUPS[int(rng.integers(len(LINEUPS)))]
            methods[stage, team, label] = rows
            methods[stage, "overall", label].extend(rows)
    for stage in ("train", "test"):
        with (root / f"{'training' if stage == 'train' else 'test'}-sweep.csv").open() as f:
            sweep = list(csv.DictReader(f))
        for t in spec["teams"][stage]:
            team = Path(t).stem
            ranked = sorted(
                (r for r in sweep if r["team"] == team),
                key=lambda r: (-int(r["wins"]), r["lineup"]),
            )
            assert len(ranked) == 90 and ranked[0]["lineup"] == winners[t]
            top = ranked[0]
            best_by_lead = {}
            for place, r in enumerate(ranked, 1):
                a = r["lineup"]
                detail = dict(
                    **r,
                    leads=pair_name(a[:2]),
                    back=pair_name(a[2:]),
                    rank=place,
                    win_pct=100 * int(r["wins"]) / int(r["battles"]),
                )
                tables.append(detail)
                best_by_lead.setdefault(a[:2], detail)
            leads.extend(best_by_lead.values())
            head_pick = head.choose(previews[team])
            summary = dict(
                split=stage,
                team=team,
                public_six="|".join(previews[team]),
                sweep_winner=top["lineup"],
                winning_leads=pair_name(top["lineup"][:2]),
                winning_back=pair_name(top["lineup"][2:]),
                tied_first=sum(r["wins"] == top["wins"] for r in ranked),
                screen_wins=int(top["wins"]),
                screen_battles=int(top["battles"]),
                head_pick=head_pick,
                head_leads=pair_name(head_pick[:2]),
                head_back=pair_name(head_pick[2:]),
            )
            for label in POLICIES:
                summary[f"{label}_pct"] = score(methods[stage, team, label])["win_pct"]
            team_results.append(summary)
        for team in [*(Path(t).stem for t in spec["teams"][stage]), "overall"]:
            for label in POLICIES:
                scores.append(
                    dict(split=stage, team=team, policy=label, **score(methods[stage, team, label]))
                )
            for baseline in POLICIES[1:]:
                effects.append(
                    dict(
                        split=stage,
                        team=team,
                        baseline=baseline,
                        **paired(methods[stage, team, "learned"], methods[stage, team, baseline]),
                    )
                )
    ts = [
        r["delta_pp"]
        for r in effects
        if r["split"] == "test" and r["team"] != "overall" and r["baseline"] == "fixed"
    ]
    samples = np.random.default_rng(49033).choice(ts, (50000, len(ts))).mean(axis=1)
    save(
        out / "roster-sensitivity.json",
        dict(
            teams=len(ts),
            mean_delta_pp=np.mean(ts).item(),
            bootstrap95_pp=np.quantile(samples, [0.025, 0.975]).tolist(),
            positive=sum(d > 0 for d in ts),
            negative=sum(d < 0 for d in ts),
            equal=sum(d == 0 for d in ts),
            bootstrap_seed=49033,
        ),
    )
    fixed_ranks = []
    for stage in ("train", "test"):
        totals = defaultdict(lambda: [0, 0])
        for r in tables:
            if r["split"] == stage:
                totals[r["lineup"]][0] += int(r["wins"])
                totals[r["lineup"]][1] += int(r["battles"])
        for idx, (a, (w, n)) in enumerate(
            sorted(totals.items(), key=lambda item: (-item[1][0], item[0])), 1
        ):
            fixed_ranks.append(
                dict(
                    split=stage,
                    rank=idx,
                    lineup=a,
                    leads=pair_name(a[:2]),
                    back=pair_name(a[2:]),
                    wins=w,
                    battles=n,
                    win_pct=100 * w / n,
                )
            )
    write_csv(out / "fixed-lineup-rankings.csv", fixed_ranks)
    write_csv(out / "lineup-rankings.csv", tables)
    write_csv(out / "best-back-by-lead.csv", leads)
    write_csv(out / "matchup-guide.csv", team_results)
    write_csv(out / "confirmation-scores.csv", scores)
    write_csv(out / "paired.csv", effects)
    for filename in ("preview-head.pt", "manifest.json", "selection-lock.json", "training-cv.csv"):
        shutil.copyfile(root / filename, out / filename)
    archive(
        out / "sweep-games.csv.gz",
        (p for stage in ("train", "test") for p in sorted((root / stage).glob("*/*.csv.gz"))),
    )
    archive(out / "confirmation-games.csv.gz", sorted((root / "confirm").glob("*/*.csv.gz")))
    plot(out / "held-out-effects.png", effects)
    print([r for r in scores if r["split"] == "test" and r["team"] == "overall"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=RUN)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    report(a.root, a.out)
