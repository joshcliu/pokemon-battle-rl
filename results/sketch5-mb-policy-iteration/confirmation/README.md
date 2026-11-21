# Prospective confirmation on 16 reserved tournament rosters

The frozen policy won **6,322 of 8,192 games (77.17%, 95% interval 73.49%–80.81%)** against the released VGC-Bench behavior-cloned opponent. We selected it at 80.71% on the separate development matrix and froze its configuration and training seeds before opening these games.

| Reserved evaluation | Original PPO | Frozen policy (95% interval) | Paired gain in points (95% interval) |
|---|---:|---:|---:|
| Reciprocal vs. released BC | 56.98% (4668/8192) | 77.17% (73.49–80.81), 6322/8192 | +20.19 (+15.34 to +24.74) |
| Sun Room vs. released BC | 54.49% (558/1024) | 76.17% (68.36–83.40), 780/1024 | +21.68 (+11.23 to +32.13) |
| Reciprocal vs. independent PPO | 50.22% (4114/8192) | 74.12% (70.24–77.99), 6072/8192 | +23.90 (+18.86 to +28.64) |

## Playing each assigned roster

The frozen policy reached 80% on 5 of the 16 learner rosters and improved on the original PPO with 16 of 16. Each row covers the eight opposing rosters in the opposite group over four training seeds, with 512 games per policy.

| Assigned team | Original PPO | Frozen policy | Paired gain in points |
|---|---:|---:|---:|
| MB351 | 65.43% | 84.18% | +18.75 |
| MB50 | 67.38% | 81.45% | +14.06 |
| MB111 | 45.12% | 77.15% | +32.03 |
| MB142 | 66.21% | 81.05% | +14.84 |
| MB428 | 56.64% | 73.83% | +17.19 |
| MB559 | 56.84% | 80.47% | +23.63 |
| MB553 | 74.22% | 85.94% | +11.72 |
| MB309 | 54.30% | 73.83% | +19.53 |
| MB438 | 47.07% | 71.48% | +24.41 |
| MB459 | 62.70% | 74.61% | +11.91 |
| MB425 | 39.06% | 75.00% | +35.94 |
| MB95 | 62.11% | 78.91% | +16.80 |
| MB89 | 58.59% | 74.41% | +15.82 |
| MB268 | 61.91% | 78.71% | +16.80 |
| MB170 | 39.06% | 64.06% | +25.00 |
| MB548 | 55.08% | 79.69% | +24.61 |

## Team matrix

- Group A — MB351, MB50, MB111, MB142, MB428, MB559, MB553, MB309.
- Group B — MB438, MB459, MB425, MB95, MB89, MB268, MB170, MB548.
- In the reciprocal tests, every A team faced every B team in both directions. Each ordered matchup played 16 games per training seed, split evenly between simulator seats.
- Our original Sun Room roster faced all 16 reserved teams. Its six were Torkoal, Oranguru, Mega Mawile, Venusaur, Scrafty and Sinistcha.
- Training seeds were 2, 3, 4 and 5. Transfer used the independent 262,144-decision shared PPO opponent with native BC preview.

The frozen policy's primary win rates for seeds 2–5 were 76.81 / 75.83 / 77.25 / 78.81%. Forward and reverse rates were 79.74 / 74.61%. Paired intervals resample training seeds, both roster axes and game replicates within seat. The custom-team interval conditions on the one original learner roster.

All 34,816 records passed roster, seat, paired opponent lineup, seed, count and outcome checks. The SHA-256 of the compressed game archive is `260d042dbeb44a0c0532d1c29e363026108444bfd6acb98373075b0c4a414949`.

[Frozen files and development gate](candidate.json) · [All game records](games.csv.gz) · [Per-matchup records](matchups.csv) · [Four-seed results](results.csv) · [Training matrix and method](../README.md)
