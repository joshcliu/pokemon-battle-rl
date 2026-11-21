"""Audit frozen-pilot preview post-training, paired matchups and opponent masking."""

import csv
import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from eval_mb_long_ppo import FINAL, read, schedule
from mb_residual_preview import (
    OUT,
    ROSTERS,
    SEEDS,
    SPEC,
    VARIANTS,
    acting_head,
    folder,
    game,
    training_schedule,
)
from mb_roster_ppo import DIVERSE_OUT
from report_mb_long_ppo import bootstrap, digest
from vgc_posttrain import atomic_rows

NAME = dict(
    native="Native BC",
    learned="BC + preview PPO",
    blind_native="Native BC, opponent hidden",
    blind_learned="BC + preview PPO, opponent hidden",
)


def compressed(name, rows):
    with gzip.open(OUT / name, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def audit_training():
    all_rows, logs, weights = [], [], []
    pilots = {
        int(r["training_seed"]): r["checkpoint_sha256"]
        for r in read(DIVERSE_OUT / "checkpoints.csv")
        if r["arm"] == "moe"
    }
    (OUT / "weights").mkdir(exist_ok=True)
    for seed in SEEDS:
        path = folder(seed) / "progress.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert (ck["seed"], ck["update"]) == (seed, SPEC["updates"])
        assert len(ck["logs"]) == SPEC["updates"]
        assert ck["head"]["adjust.weight"].abs().sum() > 0
        tasks = list(training_schedule(seed))
        exposure = Counter()
        seen = set()
        for u, row in enumerate(ck["logs"]):
            batch = read(folder(seed) / f"rollouts-{u:03}.csv")
            assert row["update"] == u + 1 and row["battles"] == (u + 1) * 128 and len(batch) == 128
            assert sum(g["reward"] == "1" for g in batch) == row["wins"]
            assert all(np.isfinite(v) for v in row.values())
            for k in range(0, len(batch), 4):
                group = batch[k : k + 4]
                group_id = u * 32 + k // 4
                _, _, _, own, foe, side = tasks[group_id]
                assert len({r["opponent_lineup"] for r in group}) == 1
                for j, g in enumerate(group):
                    assert (int(g["group"]), int(g["sample"]), g["own"], g["foe"], g["side"]) == (
                        group_id,
                        j,
                        Path(own).stem,
                        Path(foe).stem,
                        side,
                    )
                    assert int(g["seed"]) == 310_000_000 + 10000 * seed + group_id
                    assert len(set(g["lineup"])) == 4 and g["reward"] in ("1", "0", "-1")
                    assert g["preview_seed"] not in seen
                    seen.add(g["preview_seed"])
                    exposure[g["own"], g["foe"], side] += 1
                    all_rows.append(dict(training_seed=seed, **g))
            logs.append(dict(training_seed=seed, **row))
        assert (
            len(exposure) == 256
            and set(exposure.values()) == {64}
            and len(seen) == SPEC["battles_per_seed"]
        )
        export = OUT / "weights" / f"seed-{seed}.pt"
        torch.save(dict(seed=seed, pilot_sha256=pilots[seed], head=ck["head"]), export)
        weights.append(
            dict(
                training_seed=seed,
                training_battles=len(seen),
                pilot_sha256=pilots[seed],
                resume_sha256=digest(path),
                adapter_sha256=digest(export),
            )
        )
    atomic_rows(OUT / "training.csv", logs)
    atomic_rows(OUT / "checkpoints.csv", weights)
    compressed("training-games.csv.gz", all_rows)


def gather():
    archive = {
        (int(r["training_seed"]), r["cell"], int(r["seed"])): r
        for r in read(DIVERSE_OUT / "games.csv.gz")
        if r["training_teams"] == "16" and r["arm"] == "moe" and r["variant"] == "normal"
    }
    assert len(archive) == 4096
    table = {}
    all_rows = []
    comparisons = []
    replays = []
    team_counts = Counter()
    matchup_rows = []
    for k, seed in enumerate(SEEDS):
        native = {}
        lookup = {}
        plan = list(schedule("moe", seed, FINAL, "normal", diverse=True))
        for v in VARIANTS:
            rows = read(folder(seed) / f"eval-{v}.csv")
            assert len(rows) == 1024
            lookup[v] = {(r["cell"], int(r["seed"])): r for r in rows}
            assert len(lookup[v]) == len(rows)
            for direction in ("forward", "reverse"):
                wins = []
                for _, _, own, foe, games, _, cell in plan:
                    if cell != direction:
                        continue
                    pair_wins = 0
                    for sim, side in games:
                        r = lookup[v][cell, sim]
                        prior = archive[seed, cell, sim]
                        assert (r["own"], r["foe"], r["side"]) == (
                            Path(own).stem,
                            Path(foe).stem,
                            side,
                        )
                        assert (int(r["training_seed"]), r["variant"]) == (seed, v)
                        assert all(
                            prior[f] == r[f] for f in ("own", "foe", "side", "opponent_lineup")
                        )
                        assert (
                            len(r["lineup"]) == 4
                            and len(set(r["lineup"])) == 4
                            and set(r["lineup"]) <= set("123456")
                        )
                        assert r["reward"] in ("1", "0", "-1")
                        if v == "native":
                            assert all(prior[f] == r[f] for f in ("lineup", "reward", "turns"))
                            native[cell, sim] = r
                        if native[cell, sim]["lineup"] == r["lineup"]:
                            assert (native[cell, sim]["reward"], native[cell, sim]["turns"]) == (
                                r["reward"],
                                r["turns"],
                            )
                        win = r["reward"] == "1"
                        wins.append(float(win))
                        pair_wins += win
                        team_counts[v, r["own"], "wins"] += win
                        team_counts[v, r["own"], "games"] += 1
                        all_rows.append(r)
                    matchup_rows.append(
                        dict(
                            training_seed=seed,
                            variant=v,
                            cell=cell,
                            own=Path(own).stem,
                            foe=Path(foe).stem,
                            games=len(games),
                            wins=pair_wins,
                        )
                    )
                table.setdefault((v, direction), np.empty((4, 4, 8, 16)))[k] = np.reshape(
                    wins, (4, 8, 16)
                )
            # Independent replay for each condition, seat and direction.
            for task in (plan[(7 * k) % 32], plan[32 + (11 * k) % 32]):
                _, _, own, foe, games, _, cell = task
                for sim, side in games[:2]:
                    m = acting_head(seed, SPEC["updates"]) if v.endswith("learned") else None
                    fresh, _ = game(seed, own, foe, sim, side, v, sim + 500_000_000, m)
                    saved = lookup[v][cell, sim]
                    assert all(str(val) == saved[field] for field, val in fresh.items())
                    replays.append(dict(training_seed=seed, variant=v, cell=cell, **fresh))
        for label, v, b in (
            ("visible", "learned", "native"),
            ("hidden", "blind_learned", "blind_native"),
        ):
            for key, row in lookup[v].items():
                base = lookup[b][key]
                a, z = row["lineup"], base["lineup"]
                comparisons.append(
                    dict(
                        training_seed=seed,
                        view=label,
                        cell=key[0],
                        seed=key[1],
                        own=row["own"],
                        foe=row["foe"],
                        ordered_changed=int(a != z),
                        leads_changed=int(set(a[:2]) != set(z[:2])),
                        four_changed=int(set(a) != set(z)),
                        win_delta=int(row["reward"] == "1") - int(base["reward"] == "1"),
                    )
                )
    for v in VARIANTS:
        table[v, "heldout"] = (table[v, "forward"] + table[v, "reverse"]) / 2
    compressed("games.csv.gz", all_rows)
    compressed("lineup-changes.csv.gz", comparisons)
    atomic_rows(OUT / "replays.csv", replays)
    atomic_rows(OUT / "per-matchup.csv", matchup_rows)
    return table, team_counts, comparisons


def plot_effects(deltas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in deltas if r["cell"] == "heldout"][:3]
    fig, ax = plt.subplots(figsize=(7.7, 3.4))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.31)
    for i, r in enumerate(rows):
        ax.plot([r["lo95"], r["hi95"]], [i, i], color="#176c71", lw=2)
        ax.scatter(
            r["delta_pp"],
            i,
            color="#176c71",
            s=40,
            zorder=3,
            label="Pooled estimate and 95% interval" if i == 0 else None,
        )
        ax.scatter(
            [r[f"seed_{seed}_pp"] for seed in SEEDS],
            i + np.linspace(0.12, 0.24, len(SEEDS)),
            marker="|",
            s=46,
            color="#52555b",
            label="Individual training seeds" if i == 0 else None,
        )
    ax.axvline(0, color="#777", lw=1, ls="--", zorder=0)
    ax.set_yticks(
        range(3),
        [
            "Learned − native\nopponent visible",
            "Learned − native\nopponent hidden",
            "Change in gain\nvisible − hidden",
        ],
    )
    ax.set_ylim(2.55, -0.55)
    ax.set_xlabel("Paired change in held-out win rate (percentage points)")
    ax.set_title("Effect of PPO on team preview", loc="left", weight="bold")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "preview-effects.png", dpi=165)
    plt.close(fig)


def summarize(table, counts, changes):
    effects = {}
    for cell in ("forward", "reverse", "heldout"):
        visible = table["learned", cell] - table["native", cell]
        hidden = table["blind_learned", cell] - table["blind_native", cell]
        for key, d in (
            ("Learned minus native", visible),
            ("Learned minus native with opponent hidden", hidden),
            ("Change in gain when opponent is visible", visible - hidden),
            (
                "Visible minus hidden for learned picker",
                table["learned", cell] - table["blind_learned", cell],
            ),
        ):
            effects[key, cell] = d
    intervals = bootstrap({**table, **effects})
    scores = []
    deltas = []
    for (v, cell), x in table.items():
        n = x.size * (2 if cell == "heldout" else 1)
        lo, hi = intervals[v, cell]
        scores.append(
            dict(
                variant=v,
                cell=cell,
                games=n,
                wins=round(n * x.mean()),
                win_pct=100 * x.mean(),
                lo95=lo,
                hi95=hi,
            )
        )
    for (effect, cell), x in effects.items():
        lo, hi = intervals[effect, cell]
        deltas.append(
            dict(
                effect=effect,
                cell=cell,
                delta_pp=100 * x.mean(),
                lo95=lo,
                hi95=hi,
                **{
                    f"seed_{s}_pp": p
                    for s, p in zip(SEEDS, 100 * x.mean(axis=(1, 2, 3)), strict=True)
                },
            )
        )
    atomic_rows(OUT / "scores.csv", scores)
    atomic_rows(OUT / "effects.csv", deltas)
    plot_effects(deltas)
    groups = []
    for view in ("visible", "hidden"):
        rows = [r for r in changes if r["view"] == view]
        groups.append(
            dict(
                view=view,
                games=len(rows),
                **{
                    k: sum(r[k] for r in rows)
                    for k in ("ordered_changed", "leads_changed", "four_changed")
                },
                gained_wins=sum(r["win_delta"] == 1 for r in rows),
                lost_wins=sum(r["win_delta"] == -1 for r in rows),
            )
        )
    atomic_rows(OUT / "lineup-summary.csv", groups)
    per_team = []
    for v, own, f in sorted(counts):
        if f == "games":
            n = counts[v, own, f]
            w = counts[v, own, "wins"]
            per_team.append(dict(variant=v, own=own, games=n, wins=w, win_pct=100 * w / n))
    atomic_rows(OUT / "per-team.csv", per_team)
    lines = [
        "# PPO for team preview with a frozen battle policy",
        "",
        "All four preview adapters trained on 16,384 battles on Sketch5. Each used the matching frozen battle pilot from the 16-team, 262,144-decision top-2 MoE study. Each selector played 4,096 held-out games across training seeds 2–5. The native baseline reproduced all 4,096 archived lineups, opponent lineups, outcomes and battle lengths.",
        "",
        "| Preview picker | Forward | Reverse | Held-out wins | Held-out rate (95% interval) |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        r = next(r for r in scores if r["variant"] == v and r["cell"] == "heldout")
        lines.append(
            f"| {NAME[v]} | {100 * table[v, 'forward'].mean():.2f}% | {100 * table[v, 'reverse'].mean():.2f}% | {r['wins']} / {r['games']} | {r['win_pct']:.2f}% ({r['lo95']:.2f} to {r['hi95']:.2f}) |"
        )
    lines += [
        "",
        "| Matched held-out difference (percentage points) | Estimate (95% interval) | Seed 2 | Seed 3 | Seed 4 | Seed 5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in deltas:
        if r["cell"] == "heldout":
            seeds = " | ".join(f"{r[f'seed_{s}_pp']:+.2f}" for s in SEEDS)
            lines.append(
                f"| {r['effect']} | {r['delta_pp']:+.2f} ({r['lo95']:+.2f} to {r['hi95']:+.2f}) | {seeds} |"
            )
    lines += [
        "",
        "![Paired preview effects with confidence intervals and individual seeds](preview-effects.png)",
        "",
        "## Lineup choices",
        "",
        "Each change compares the learned and native pickers under the same visibility and preview sampling seed. Lead-pair and four-member changes ignore order within the respective sets.",
        "",
        "| Opponent at preview | Ordered lineup changed | Lead pair changed | Four members changed | Wins gained | Wins lost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in groups:
        lines.append(
            f"| {r['view'].title()} | {100 * r['ordered_changed'] / r['games']:.1f}% | {100 * r['leads_changed'] / r['games']:.1f}% | {100 * r['four_changed'] / r['games']:.1f}% | {r['gained_wins']} | {r['lost_wins']} |"
        )
    lines += [
        "",
        "## Team matrix",
        "",
        "Every training matchup received 128 battles per seed, split evenly between seats. Every held-out pair played 16 games in each direction per selector and seed.",
        "",
        "| Schedule | Our teams | Frozen BC opponent teams |",
        "|---|---|---|",
    ]
    for title, a, b in (
        ("Training", "own_train", "foe_train"),
        ("Held-out forward", "own_test", "foe_test"),
        ("Held-out reverse", "foe_test", "own_test"),
    ):
        own = ", ".join(Path(x).stem for x in ROSTERS[a])
        foe = ", ".join(Path(x).stem for x in ROSTERS[b])
        lines.append(f"| {title} | {own} | {foe} |")
    lines += [
        "",
        "## Method",
        "",
        "At each native preview phase, the frozen BC actor reads the public board and produces its 256-dimensional actor representation. A 64-unit tanh layer and a zero-initialized output layer add six legal-slot offsets to each of the two positions. The native masks enforce four distinct Pokémon. Both trained and native selectors sample ordered lead and reserve pairs. A separate scalar value head reads the initial matchup. The roster bank and BC battle opponent match the previous study.",
        "",
        "Each group samples four lineups independently against the same opponent preview, simulator seed, seat and battle sampling seeds. Their terminal rewards are +1, 0 and −1 for win, draw and loss. PPO uses the joint log probability of all four picks and clips the ratio at 0.2. Its baseline averages the predicted matchup value and the mean return of the other three lineups in the group. Each update uses 128 games with four epochs, minibatches of 64 and an entropy coefficient of 0.002. Only the adapter and preview value head receive updates.",
        "",
        "For the blinded control we zero all six opposing Pokémon tokens before either the native actor or the adapter sees them. The opposing BC picker and both in-battle policies retain normal information. The change in gain compares learned minus native win rate with and without this mask. Masking is an inference-time intervention on a normally trained picker. Paired intervals resample training seed, both roster axes and game seeds within seats (6,000 draws). These teams have appeared in earlier evaluations. The study measures one frozen opponent.",
        "",
        "[Pilot and full team sheets](../sketch5-mb-team-diversity/README.md) · [Locked protocol](protocol.json) · [Win rates](scores.csv) · [Paired effects](effects.csv) · [Per-team scores](per-team.csv) · [Per-matchup scores](per-matchup.csv) · [Training updates](training.csv) · [Lineup changes](lineup-summary.csv) · [Game records](games.csv.gz) · [Replay checks](replays.csv) · [Checkpoint hashes](checkpoints.csv)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(
        "Verified 65,536 training battles, 16,384 evaluation games, 4,096 archived native games and 64 replay checks.",
        flush=True,
    )


if __name__ == "__main__":
    from probe_mb_residual_preview import probe

    torch.set_num_threads(1)
    audit_training()
    summarize(*gather())
    probe()
