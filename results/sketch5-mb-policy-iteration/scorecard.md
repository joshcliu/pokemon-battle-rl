# Four-seed win rates

Each row includes every scheduled game for four pilots. Intervals resample both roster axes, training seeds and games within seat. The custom-team intervals condition on that one learner team. All tables are development diagnostics. The 16 reserved confirmation teams have a separate schedule.

## Reciprocal development against released BC

| Agent | Wins / games | Win rate (95% interval) | Four seed rates (%) |
|---|---:|---:|---|
| Full-battle belief search + learned mechanics | 1530 / 2048 | 74.71% (69.53–79.69) | 75.98 / 74.80 / 74.61 / 73.44 |
| Preview search + state-dependent PPO | 1499 / 2048 | 73.19% (67.53–79.00) | 71.88 / 72.27 / 74.61 / 74.02 |
| Preview search + learned mechanics | 1421 / 2048 | 69.38% (63.18–75.49) | 71.68 / 69.34 / 69.14 / 67.38 |
| Endgame value search + learned mechanics | 1420 / 2048 | 69.34% (63.23–75.93) | 67.38 / 72.46 / 70.51 / 66.99 |
| State-dependent mechanics PPO | 1404 / 2048 | 68.55% (62.60–74.32) | 67.19 / 69.73 / 69.73 / 67.58 |
| PPO with learned mechanics weights | 1383 / 2048 | 67.53% (61.13–74.51) | 65.43 / 71.68 / 68.75 / 64.26 |
| PPO with hand-set mechanics weights | 1335 / 2048 | 65.19% (58.89–71.73) | 63.67 / 68.95 / 63.28 / 64.84 |
| Preview search + frozen PPO | 1314 / 2048 | 64.16% (57.91–70.56) | 68.16 / 63.48 / 62.50 / 62.50 |
| Fitted mechanics from terminal outcomes | 1280 / 2048 | 62.50% (56.54–68.60) | 60.16 / 62.89 / 64.65 / 62.30 |
| Fitted mechanics with three-step targets | 1268 / 2048 | 61.91% (56.01–67.92) | 60.16 / 62.89 / 63.09 / 61.52 |
| Monte Carlo action value (weight 8) | 1264 / 2048 | 61.72% (55.62–67.43) | 63.67 / 62.30 / 61.72 / 59.18 |
| Monte Carlo action value (weight 2) | 1257 / 2048 | 61.38% (55.62–66.99) | 63.28 / 61.91 / 60.74 / 59.57 |
| Opening search + frozen PPO | 1254 / 2048 | 61.23% (55.22–67.29) | 62.30 / 62.89 / 59.38 / 60.35 |
| group-64-map | 1251 / 2048 | 61.08% (55.81–67.04) | 62.70 / 60.55 / 60.74 / 60.35 |
| group-16-map | 1246 / 2048 | 60.84% (55.13–66.85) | 61.13 / 62.70 / 60.35 / 59.18 |
| Frozen PPO, most probable command | 1244 / 2048 | 60.74% (54.59–66.55) | 62.89 / 61.72 / 59.77 / 58.59 |
| Frozen PPO, temperature 0.5 | 1237 / 2048 | 60.40% (54.93–66.55) | 61.13 / 61.52 / 58.59 / 60.35 |
| Frozen PPO, sampled | 1202 / 2048 | 58.69% (53.47–64.11) | 58.40 / 59.57 / 57.03 / 59.77 |
| group-64-native | 1192 / 2048 | 58.20% (52.15–65.38) | 56.84 / 60.74 / 57.23 / 58.01 |
| group-16-native | 1190 / 2048 | 58.11% (52.93–63.77) | 56.84 / 56.64 / 58.79 / 60.16 |
| Offline lineup value + frozen PPO | 1176 / 2048 | 57.42% (50.73–63.38) | 59.18 / 51.95 / 59.18 / 59.38 |

## Original Sun Room on the 12 development opponents

| Agent | Wins / games | Win rate (95% interval) | Four seed rates (%) |
|---|---:|---:|---|
| Full-battle belief search + learned mechanics | 269 / 384 | 70.05% (58.59–79.95) | 69.79 / 68.75 / 71.88 / 69.79 |
| Offline lineup value + frozen PPO | 264 / 384 | 68.75% (53.65–81.25) | 61.46 / 70.83 / 76.04 / 66.67 |
| Preview search + state-dependent PPO | 261 / 384 | 67.97% (54.43–79.95) | 61.46 / 73.96 / 68.75 / 67.71 |
| Preview search + learned mechanics | 260 / 384 | 67.71% (55.21–79.17) | 65.62 / 67.71 / 70.83 / 66.67 |
| Preview search + frozen PPO | 248 / 384 | 64.58% (51.30–76.30) | 60.42 / 63.54 / 65.62 / 68.75 |
| Endgame value search + learned mechanics | 242 / 384 | 63.02% (49.47–75.78) | 67.71 / 60.42 / 63.54 / 60.42 |
| State-dependent mechanics PPO | 229 / 384 | 59.64% (46.61–72.40) | 60.42 / 63.54 / 56.25 / 58.33 |
| PPO with learned mechanics weights | 227 / 384 | 59.11% (44.27–72.92) | 64.58 / 58.33 / 57.29 / 56.25 |
| PPO with hand-set mechanics weights | 222 / 384 | 57.81% (42.96–71.88) | 61.46 / 61.46 / 50.00 / 58.33 |
| Fitted mechanics with three-step targets | 208 / 384 | 54.17% (39.58–68.49) | 55.21 / 51.04 / 48.96 / 61.46 |
| Frozen PPO, sampled | 207 / 384 | 53.91% (40.10–67.19) | 51.04 / 51.04 / 57.29 / 56.25 |
| group-64-native | 206 / 384 | 53.65% (39.58–66.41) | 53.12 / 51.04 / 60.42 / 50.00 |
| Opening search + frozen PPO | 206 / 384 | 53.65% (39.58–66.67) | 56.25 / 50.00 / 53.12 / 55.21 |
| group-64-map | 201 / 384 | 52.34% (37.24–66.67) | 62.50 / 46.88 / 46.88 / 53.12 |
| Fitted mechanics from terminal outcomes | 200 / 384 | 52.08% (37.50–66.15) | 54.17 / 50.00 / 44.79 / 59.38 |
| Monte Carlo action value (weight 2) | 199 / 384 | 51.82% (36.46–66.41) | 57.29 / 47.92 / 44.79 / 57.29 |
| Frozen PPO, most probable command | 195 / 384 | 50.78% (35.42–65.62) | 58.33 / 47.92 / 42.71 / 54.17 |
| Monte Carlo action value (weight 8) | 194 / 384 | 50.52% (36.46–64.33) | 57.29 / 47.92 / 44.79 / 52.08 |

## Reciprocal development against the independent shared PPO

| Agent | Wins / games | Win rate (95% interval) | Four seed rates (%) |
|---|---:|---:|---|
| Preview search + state-dependent PPO | 1336 / 2048 | 65.23% (59.42–71.24) | 65.82 / 64.26 / 66.80 / 64.06 |
| Preview search + learned mechanics | 1319 / 2048 | 64.40% (57.52–71.09) | 64.65 / 62.50 / 69.34 / 61.13 |
| State-dependent mechanics PPO | 1227 / 2048 | 59.91% (54.00–66.70) | 56.64 / 59.57 / 60.35 / 63.09 |
| PPO with learned mechanics weights | 1165 / 2048 | 56.88% (50.88–62.70) | 54.88 / 56.64 / 60.16 / 55.86 |
| Preview search + frozen PPO | 1125 / 2048 | 54.93% (48.83–61.82) | 58.79 / 52.54 / 54.30 / 54.10 |
| Frozen PPO, most probable command | 1029 / 2048 | 50.24% (44.43–56.69) | 48.63 / 49.22 / 50.20 / 52.93 |
| Frozen PPO, sampled | 983 / 2048 | 48.00% (42.77–53.52) | 47.66 / 48.63 / 47.46 / 48.24 |

[Exact roster matrices and methods](README.md) · [Paired gain intervals](paired-effects.csv) · [Per-matchup wins](matchups.csv) · [Machine-readable totals](development.csv)
