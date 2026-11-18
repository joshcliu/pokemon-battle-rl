# Frozen MC461 cross-play against VGC-Bench opponents

MC461's original four won **541/768 games (70.44%, Wilson 95% interval 67.12–73.56%)** against VGC-Bench's published behavior-cloned player on 12 external teams. The 60-team preview head won 480/768. Its paired change from the original four was **−7.94 percentage points** (95% interval −12.20 to −3.69). The 15-team head led against both damage-oriented heuristics. The choice of evaluation opponent changes which preview strategy looks best. [Scores](win-rates.csv) · [Paired differences](preview-differences.csv)

| Opponent on the 12 external teams | Original four | 15-team head | 60-team head | 60-team head + fallback |
| --- | ---: | ---: | ---: | ---: |
| Random | 96.2% | 96.1% | 96.0% | 96.0% |
| Max base power | 70.8% | 75.3% | 74.9% | 73.8% |
| Simple heuristics | 64.1% | 66.8% | 60.5% | 60.9% |
| Behavior cloning, learned preview | **70.4%** | 64.5% | 62.5% | 64.3% |
| Same weights, listed first four | 76.2% | 73.2% | 73.7% | 76.2% |

Allowing the behavior-cloned opponent to pick its leads and reserves cost the original four 5.73 points (paired 95% interval −9.60 to −1.86). With the 60-team preview head it cost 11.20 points (−15.20 to −7.20). Both arms shared battle seeds, side assignments and the action-sampling stream after preview. The learned opponent chose its own moves in both. [Opponent preview comparison](opponent-preview-effect.csv)

On the seven previous test rosters the four selectors clustered between 67.9% and 69.0% against the behavior-cloned player. On MC405 the original four and the fallback won 63/64 games, the 60-team head won 64/64, and the 15-team head won 40/64. The current interval for the 60-team head's advantage over the original four on MC405 includes zero. [Team-specific results](win-rates.csv) · [Selected lead and reserve slots](team-choices.csv)

## Design

The fixed learner used the exported MC461 battle Transformer, greedy action selection and four previously locked preview policies. Each opponent piloted each roster for 64 games per policy, with both sides equally represented. The three heuristic classes are the ones used by the benchmark. Their source file matches the benchmark fork byte for byte. The published seed-1, epoch-100 BC checkpoint uses its own embedding, action mask, sampled action policy and two-step preview. The benchmark’s open-sheet evaluation and multi-policy ranking address a broader question. This test used closed sheets and Regulation M-C so the original MC461 could play. [VGC-Bench evaluator](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/eval.py) · [Published checkpoint](https://huggingface.co/cameronangliss/vgc-bench-models/blob/204c76741829ca0681629e41382043c385850d5c/results/saves_bc/seed1/100.zip)

Of the 19 opposing rosters, seven came from our previous evaluation and 12 from the authors' public M-B team bank. We selected the latter before running games, preserving the published exports and filtering for legal M-B and M-C teams with distinct base-species sixes and first fours relative to all 67 previous regional rosters, Sun and Balance. The 12 may share individual species or strategies with training. Each percentage weights rosters equally. Paired intervals reflect battle randomness within the fixed matchup and side strata. [Team provenance](../../teams/vgcbench-mb-transfer/source.json) · [Locked protocol](manifest.json)

The audit covers 24,320 policy outcomes from 16,320 distinct simulations. Identical MC461 lineup decisions under the same seed and opponent reused the same game. Every game ended in a win or loss. Ten games replayed exactly, including lineup choices, outcome and finishing turn. [Audit](audit.json) · [Replay check](reproduced-games.csv) · [Game records](games.csv.gz) · [Reproduction commands](../../docs/vgcbench.md)
