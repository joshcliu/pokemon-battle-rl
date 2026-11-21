"""Independently audit and archive the completed, frozen prospective evaluation."""

import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results/sketch5-mb-policy-iteration"
DEST = OUT / "confirmation"
SPLITS = ("primary", "custom", "transfer")
LABELS = {
    "primary": "Reciprocal vs. released BC",
    "custom": "Sun Room vs. released BC",
    "transfer": "Reciprocal vs. independent PPO",
}


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    lock = json.loads((DEST / "candidate.json").read_text())
    spec = json.loads((OUT / "protocol.json").read_text())
    method = lock["method"]
    results = {(r["split"], r["method"]): r for r in read(DEST / "results.csv")}
    assert set(results) == {(s, m) for s in SPLITS for m in ("native", method)}
    all_rows = []
    pairs = defaultdict(list)
    seeds = lock["seeds"]
    for split in SPLITS:
        a, b = spec["confirmation_a"], spec["confirmation_b"]
        directions = 2
        if split == "custom":
            a, b, directions = [spec["custom"]], a + b, 1
        baseline = {}
        for mode in ("native", method):
            audited = {}
            for file in sorted((DEST / "games" / split / mode).glob("*.csv")):
                for r in read(file):
                    s, i, j, d, k = (
                        int(r[x]) for x in ("training_seed", "i", "j", "direction", "k")
                    )
                    assert (
                        s in seeds
                        and 0 <= i < len(a)
                        and 0 <= j < len(b)
                        and 0 <= d < directions
                        and 0 <= k < 16
                    )
                    key = (s, i, j, d, k)
                    assert key not in audited and r["method"] == mode and int(r["turns"]) > 0
                    x, y = (a[i], b[j]) if not d else (b[j], a[i])
                    assert (r["own"], r["foe"]) == (Path(x).stem, Path(y).stem)
                    assert (
                        int(r["sim"])
                        == lock["simulator_seed_bases"][split] + 10000 * i + 100 * j + 20 * d + k
                    )
                    assert r["side"] == ("p1", "p2")[k % 2] and r["reward"] in ("-1", "0", "1")
                    assert all(
                        len(r[c]) == len(set(r[c])) == 4 and set(r[c]) <= set("123456")
                        for c in ("lineup", "opp_lineup")
                    )
                    if split == "transfer":
                        assert r["opponent"] == "shared-2-262144-normal"
                    if mode == "native":
                        baseline[key] = r["opp_lineup"]
                    else:
                        assert baseline[key] == r["opp_lineup"]
                    audited[key] = int(r["reward"] == "1")
                    pairs[split, mode, r["own"], r["foe"]].append(audited[key])
                    all_rows.append(dict(split=split, **r))
            summary = results[split, mode]
            assert (
                len(audited)
                == len(a) * len(b) * directions * 16 * len(seeds)
                == int(summary["games"])
            )
            assert sum(audited.values()) == int(summary["wins"])
            assert round(100 * sum(audited.values()) / len(audited), 8) == round(
                float(summary["win_pct"]), 8
            )

    fields = [
        "split",
        "method",
        "training_seed",
        "own",
        "foe",
        "i",
        "j",
        "direction",
        "k",
        "side",
        "sim",
        "lineup",
        "opp_lineup",
        "reward",
        "turns",
    ]
    extra = sorted({k for r in all_rows for k in r} - set(fields))
    archive = DEST / "games.csv.gz"
    tmp = DEST / "games.csv.gz.tmp"
    with (
        tmp.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as f,
    ):
        writer = csv.DictWriter(f, fieldnames=fields + extra, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    tmp.replace(archive)
    with gzip.open(archive, "rt", newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == len(all_rows)
    with (DEST / "matchups.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["split", "method", "learner_team", "opponent_team", "wins", "games"])
        for k, outcomes in sorted(pairs.items()):
            w.writerow([*k, sum(outcomes), len(outcomes)])
    primary = results["primary", method]
    lines = [
        "# Prospective confirmation on 16 reserved tournament rosters",
        "",
        f"The frozen policy won **{int(primary['wins']):,} of {int(primary['games']):,} games ({float(primary['win_pct']):.2f}%, 95% interval {float(primary['lo95']):.2f}%–{float(primary['hi95']):.2f}%)** against the released VGC-Bench behavior-cloned opponent. We selected it at {100 * lock['development_wins'] / lock['development_games']:.2f}% on the separate development matrix and froze its configuration and training seeds before opening these games.",
        "",
        "| Reserved evaluation | Original PPO | Frozen policy (95% interval) | Paired gain in points (95% interval) |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        old, r = results[split, "native"], results[split, method]
        lines.append(
            f"| {LABELS[split]} | {float(old['win_pct']):.2f}% ({old['wins']}/{old['games']}) | {float(r['win_pct']):.2f}% ({float(r['lo95']):.2f}–{float(r['hi95']):.2f}), {r['wins']}/{r['games']} | {float(r['delta_pp']):+.2f} ({float(r['diff_lo95']):+.2f} to {float(r['diff_hi95']):+.2f}) |"
        )

    # Pool the eight opponents for each assigned roster with equal game counts.
    assigned = defaultdict(list)
    for (split, mode, own, _), outcomes in pairs.items():
        if split == "primary":
            assigned[mode, own].extend(outcomes)
    teams = [Path(p).stem for p in (*spec["confirmation_a"], *spec["confirmation_b"])]
    assert all(
        len(assigned[m, t]) == len(seeds) * 8 * 16 for m in ("native", method) for t in teams
    )
    at_target = sum(5 * sum(assigned[method, t]) >= 4 * len(assigned[method, t]) for t in teams)
    improved = sum(sum(assigned[method, t]) > sum(assigned["native", t]) for t in teams)
    lines += [
        "",
        "## Playing each assigned roster",
        "",
        f"The frozen policy reached 80% on {at_target} of the 16 learner rosters and improved on the original PPO with {improved} of 16. Each row covers the eight opposing rosters in the opposite group over four training seeds, with 512 games per policy.",
        "",
        "| Assigned team | Original PPO | Frozen policy | Paired gain in points |",
        "|---|---:|---:|---:|",
    ]
    for t in teams:
        original, candidate = (
            100 * sum(assigned[m, t]) / len(assigned[m, t]) for m in ("native", method)
        )
        lines.append(f"| {t} | {original:.2f}% | {candidate:.2f}% | {candidate - original:+.2f} |")

    def names(group):
        return ", ".join(Path(t).stem for t in group)

    lines += [
        "",
        "## Team matrix",
        "",
        f"- Group A — {names(spec['confirmation_a'])}.",
        f"- Group B — {names(spec['confirmation_b'])}.",
        "- In the reciprocal tests, every A team faced every B team in both directions. Each ordered matchup played 16 games per training seed, split evenly between simulator seats.",
        "- Our original Sun Room roster faced all 16 reserved teams. Its six were Torkoal, Oranguru, Mega Mawile, Venusaur, Scrafty and Sinistcha.",
        "- Training seeds were 2, 3, 4 and 5. Transfer used the independent 262,144-decision shared PPO opponent with native BC preview.",
        "",
        f"The frozen policy's primary win rates for seeds 2–5 were {primary['seeds']}%. Forward and reverse rates were {primary['directions']}%. Paired intervals resample training seeds, both roster axes and game replicates within seat. The custom-team interval conditions on the one original learner roster.",
        "",
        f"All {len(all_rows):,} records passed roster, seat, paired opponent lineup, seed, count and outcome checks. The SHA-256 of the compressed game archive is `{hashlib.sha256(archive.read_bytes()).hexdigest()}`.",
        "",
        "[Frozen files and development gate](candidate.json) · [All game records](games.csv.gz) · [Per-matchup records](matchups.csv) · [Four-seed results](results.csv) · [Training matrix and method](../README.md)",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines))
    print("Verified", len(all_rows), "prospective games and generated the report", flush=True)


if __name__ == "__main__":
    main()
