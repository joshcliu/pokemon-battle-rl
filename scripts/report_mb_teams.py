"""Publish the exact M-B pilot assignments, six-Pokémon rosters and paired scores."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from itertools import product
from pathlib import Path

from poke_env.teambuilder import Teambuilder

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/sketch5-mb-posttraining-pilot"
SOURCE = ROOT / ".cache/vgcbench-external/source"


def matchup_table(archive, labels, learner, opponent, games_per_pair):
    counts = Counter()
    wins = Counter()
    pairs = set(product(learner, opponent))
    with gzip.open(REPORT / archive, "rt") as f:
        for row in csv.DictReader(f):
            pair = row["own"], row["foe"]
            if row["label"] in labels and pair in pairs:
                counts[pair] += 1
                wins[pair] += row["reward"] == "1"
    assert all(counts[pair] == games_per_pair for pair in pairs)
    lines = [
        "| Our roster ↓ / BC roster → | " + " | ".join(opponent) + " |",
        "|---|" + "---:|" * len(opponent),
    ]
    for own in learner:
        scores = [f"{wins[own, foe]}/{counts[own, foe]}" for foe in opponent]
        lines.append("| " + " | ".join([own, *scores]) + " |")
    return lines


def write_team_matrix(source=SOURCE):
    spec = json.loads((REPORT / "protocol.json").read_text())
    dev = json.loads((ROOT / "results/sketch5-vgcbench-bc/protocol.json").read_text())
    pools = {
        key: [Path(p).stem for p in spec[key]]
        for key in ("own_train", "own_test", "foe_train", "foe_test")
    }
    lines = [
        "# Which teams each bot played",
        "",
        "In battle PPO training, we independently sampled one learner roster and one frozen BC opponent roster per game. Our learner brought slots 1–4 in order. The BC opponent selected its leads and reserves from its six at team preview.",
        "",
        "| Assignment | Our PPO roster pool | Frozen BC opponent roster pool |",
        "|---|---|---|",
    ]
    schedules = [
        ("Training and familiar/familiar evaluation", "own_train", "foe_train"),
        ("Familiar learner / unseen opponent", "own_train", "foe_test"),
        ("Unseen learner / familiar opponent", "own_test", "foe_train"),
        ("Both unseen", "own_test", "foe_test"),
        ("Both unseen, assignments reversed", "foe_test", "own_test"),
    ]
    for label, own, foe in schedules:
        lines.append(f"| {label} | {', '.join(pools[own])} | {', '.join(pools[foe])} |")
    mc461 = Path(dev["learner_team"]).stem
    dev_foes = [Path(p).stem for p in dev["teams"]]
    lines += [
        f"| MC461 development | {mc461} | {', '.join(dev_foes)} |",
        "",
        "Each evaluation row crosses every learner roster with every opposing roster. Matrix evaluation played eight games per pair per checkpoint, four in each simulator seat. The MC461 development evaluation played 64 per pair per checkpoint.",
        "",
        "## Roster sheets in slot order",
        "",
        "In the fixed-four runs, slots 1–2 led and slots 3–4 were in back. Adaptive previews could choose any four. The individual game records store both ordered selections as slot numbers. [Matrix games](matrix-games.csv.gz) · [Development games](development-games.csv.gz) · [Reciprocal games](reciprocal-games.csv.gz)",
        "",
        "| Pool | Roster and full set sheet | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    source_paths = {
        r["team"]: r["sha256"]
        for r in json.loads((ROOT / "teams/vgcbench-mb-transfer/source.json").read_text())["teams"]
    }
    for pool, files in [
        ("PPO training", spec["own_train"]),
        ("BC training opponent", spec["foe_train"]),
        ("Held-out learner", spec["own_test"]),
        ("Held-out opponent", spec["foe_test"]),
        ("BC development opponent", dev["teams"]),
    ]:
        for filename in files:
            path = Path(filename)
            if path.is_relative_to(".cache/vgcbench-external/source"):
                upstream = path.relative_to(".cache/vgcbench-external/source")
                disk = source / upstream
                link = (
                    f"https://github.com/cameronangliss/vgc-bench/blob/{spec['source']}/{upstream}"
                )
            else:
                disk = ROOT / path
                link = os.path.relpath(disk, REPORT)
            raw = disk.read_bytes()
            expected = (
                spec["hashes"][filename] if filename in spec["hashes"] else source_paths[path.name]
            )
            assert hashlib.sha256(raw).hexdigest() == expected, disk
            team = [
                m.species or m.nickname
                for m in Teambuilder.parse_showdown_team(raw.decode().strip())
            ]
            assert len(team) == 6
            lines.append("| " + " | ".join([pool, f"[{path.stem}]({link})", *team]) + " |")
    lines += [
        "",
        "## Wins and games for each roster pair",
        "",
        "Rows name the team our PPO battle policy piloted. Columns name the frozen BC opponent’s team. Counts pool the four training seeds. The matrix and reversed games used the released BC preview on both sides. In the development games our PPO learner brought its fixed first four.",
        "",
    ]
    main_labels = {f"native-preview-{s}" for s in spec["training_seeds"]}
    for index, (title, own, foe) in enumerate(schedules):
        archive = "reciprocal-games.csv.gz" if own == "foe_test" else "matrix-games.csv.gz"
        lines += [
            "### " + ("Both familiar" if index == 0 else title),
            "",
            *matchup_table(
                archive,
                main_labels,
                pools[own],
                pools[foe],
                len(main_labels) * spec["eval_games_per_pair"],
            ),
            "",
        ]
    dev_labels = {f"native-{s}" for s in spec["training_seeds"]}
    lines += [
        "### MC461 development",
        "",
        *matchup_table(
            "development-games.csv.gz",
            dev_labels,
            [mc461],
            dev_foes,
            len(dev_labels) * spec["development_games_per_team"],
        ),
        "",
    ]
    (REPORT / "team-matrix.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Pinned VGC-Bench checkout")
    write_team_matrix(parser.parse_args().source)
