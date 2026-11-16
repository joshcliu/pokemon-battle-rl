"""Verify the frozen learner-roster experiment and score its paired evaluations."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from analyze_long_rewards import load, settings, write_csv
from analyze_rewards import records
from opponent_diversity import ROOT, SEEDS, STEPS, verified_mixture
from roster_transfer import BATTLES, EVAL_SEED, POLICIES, STYLES, split

from pokemon_rl.policy import Policy

SUN = "teams/sun.txt"
OPPONENT_TEAMS = ["teams/balance.txt", SUN]
OUTCOMES = ("win", "loss", "tie", "truncated")


def same_runtime(manifest, reference):
    for field in ("host", "simulator", "python", "numpy", "torch"):
        assert manifest[field] == reference[field]
    assert manifest["host"].split(".")[0] == "sketch5"
    for path, detail in manifest["teams"].items():
        assert detail["sha256"] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def training_audit(root, seed, prior, roster_paths):
    train = root / f"seed-{seed}/train"
    reference = load(prior.parent / "manifest.json")
    manifest = load(train / "manifest.json")
    same_runtime(manifest, reference)
    a, b = manifest["args"], reference["args"]
    except_keys = {"learner_teams", "run_dir", "save_initial", "save_at"}
    assert settings(a, except_keys) == settings(b, except_keys)
    assert (a["seed"], a["reward_mode"], a["opponent"], a["steps"]) == (
        seed,
        "win-loss",
        "mixture",
        STEPS,
    )
    assert a["learner_teams"] == roster_paths and a["opponent_teams"] == OPPONENT_TEAMS
    assert b["team"] == SUN
    assert set(manifest["teams"]) == set(roster_paths + OPPONENT_TEAMS)
    episodes = records(train / "episodes.jsonl")
    previous = records(prior.parent / "episodes.jsonl")
    for old, new in zip(previous, episodes):
        for field in ("episode", "simulator_seed", "opponent", "opponent_team_index"):
            assert old[field] == new[field]
    summary = load(train / "summary.json")
    assert summary["steps"] == STEPS and summary["episodes"] == len(episodes)
    counts = Counter(r["learner_team"] for r in episodes)
    assert set(counts) == set(roster_paths)
    draws = [
        {"seed": seed, "roster": Path(t).stem, "training_battles": n}
        for t, n in sorted(counts.items())
    ]
    checkpoints = dict(
        zip(POLICIES, [prior, train / "checkpoint.pt", train / "checkpoint-0.pt"], strict=True)
    )
    provenance = []
    for policy, path in checkpoints.items():
        model = torch.load(path, map_location="cpu", weights_only=True)
        assert model["seed"] == seed and model["steps"] == (0 if policy == "untrained" else STEPS)
        if policy == "untrained":
            torch.manual_seed(seed)
            original = Policy(model["obs_size"], model["hidden_size"]).state_dict()
            assert all(torch.equal(original[k], v) for k, v in model["policy"].items())
        provenance.append(
            {
                "seed": seed,
                "policy": policy,
                "steps": model["steps"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": (reference if policy == "sun" else manifest)["git_commit"],
            }
        )
    return manifest, checkpoints, draws, provenance


def plot_matrix(matrix, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.4), layout="constrained")
    values = [[matrix[p][c] for c in ("sun", "held_out")] for p in ("sun", "tournament")]
    ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1], ["Sun", "7 held-out tournament rosters"])
    ax.set_yticks([0, 1], ["Sun only", "15 tournament rosters"])
    ax.set(
        xlabel="Roster played at evaluation",
        ylabel="Training roster pool",
        title="Win rate against the damage and switching bots",
    )
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(
                x,
                y,
                f"{value:.2f}%",
                color="white" if value >= 58 else "#112239",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )
    fig.savefig(output / "transfer-matrix.png", dpi=180)
    plt.close(fig)


def analyze(root, prior, output):
    torch.set_num_threads(1)
    pool = split()
    rosters = [SUN, *pool["test"]]
    old = verified_mixture(prior)
    games, scores, draws, checkpoints = [], [], [], []
    values, schedule = {}, {}
    eval_commits = set()
    for seed in SEEDS:
        manifest, paths, sampled, hashes = training_audit(root, seed, old[seed], pool["train"])
        draws.extend(sampled)
        checkpoints.extend(hashes)
        for policy in POLICIES:
            for roster in rosters:
                name = Path(roster).stem
                d = root / f"seed-{seed}/eval" / policy / name / "games"
                ev = load(d / "manifest.json")
                same_runtime(ev, manifest)
                eval_commits.add(ev["git_commit"])
                args = ev["args"]
                assert (args["seed"], args["battles"], args["deterministic"]) == (
                    EVAL_SEED,
                    BATTLES,
                    False,
                )
                assert args["team"] == roster and args["opponent_teams"] == OPPONENT_TEAMS
                assert (
                    args["opponents"] == list(STYLES)
                    and Path(args["checkpoint"]).resolve() == paths[policy]
                )
                assert not (d / "errors.jsonl").exists()
                summary = load(d / "summary.json")
                rates = []
                for style in STYLES:
                    key = {
                        "seed": seed,
                        "policy": policy,
                        "learner_roster": name,
                        "opponent": style,
                    }
                    rows = records(d / f"{style}.jsonl")
                    assert [r["seed"] for r in rows] == list(range(EVAL_SEED, EVAL_SEED + BATTLES))
                    count = Counter(r["outcome"] for r in rows)
                    assert set(count) <= set(OUTCOMES) and count == Counter(
                        summary[style]["outcomes"]
                    )
                    for r in rows:
                        assert r["opponent"] == style and r["learner_team"] == roster
                        pair = r["simulator_seed"], r["opponent_team_index"]
                        assert schedule.setdefault(r["seed"], pair) == pair
                        games.append(
                            key
                            | {
                                "eval_seed": r["seed"],
                                "simulator_seed": pair[0],
                                "opponent_team": Path(OPPONENT_TEAMS[pair[1]]).stem,
                                "outcome": r["outcome"],
                                "turns": r["turn"],
                            }
                        )
                    # Use all scheduled battles. Report truncations separately.
                    rate = 100 * count["win"] / BATTLES
                    scores.append(key | {k: count[k] for k in OUTCOMES} | {"win_pct": rate})
                    rates.append(rate)
                values[seed, policy, roster] = mean(rates)
    per_seed = []
    for seed in SEEDS:
        for policy in POLICIES:
            per_seed.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "sun": values[seed, policy, SUN],
                    "held_out": mean(values[seed, policy, r] for r in pool["test"]),
                }
            )
    matrix = {
        p: {c: mean(r[c] for r in per_seed if r["policy"] == p) for c in ("sun", "held_out")}
        for p in POLICIES
    }
    per_roster = [
        {"roster": Path(roster).stem}
        | {p: mean(values[s, p, roster] for s in SEEDS) for p in POLICIES}
        for roster in rosters
    ]
    assert len(eval_commits) == 1
    result = {
        "matrix_win_pct": matrix,
        "evaluation_commit": next(iter(eval_commits)),
        "scheduled_games": len(games),
        "outcomes": dict(Counter(r["outcome"] for r in games)),
        "opponent_teams_per_100": dict(
            Counter(Path(OPPONENT_TEAMS[i]).stem for _, i in schedule.values())
        ),
        "training_rosters": len(pool["train"]),
        "held_out_rosters": len(pool["test"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    for file, rows in [
        ("per-seed.csv", per_seed),
        ("per-roster.csv", per_roster),
        ("scores.csv", scores),
        ("checkpoints.csv", checkpoints),
        ("training-draws.csv", draws),
        ("evaluation-games.csv.gz", games),
    ]:
        write_csv(output / file, rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_matrix(matrix, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reuse-sun-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs.resolve(), args.reuse_sun_from.resolve(), args.output)
