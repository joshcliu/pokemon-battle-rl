# AI assistance record

Codex assisted with the initial repository implementation, including literature discovery and summaries, environment and simulator integration, heuristic opponents, PPO code, tests, debugging, experiment commands, and documentation. Codex also ran the local engineering checks and pilot experiments recorded in the repository.

Codex implemented the three reward conditions, ran the four-seed replication on Sketch5, checked the paired evaluation schedules, and drafted the comparison and proposed follow-up experiments.

Codex also ran the matched 65,536- and 262,144-step comparison on Sketch5,
verified checkpoint weights and evaluation schedules, and generated its results table and plot.

Codex ran the three-condition opponent diversity study on Sketch5, verified the
four reused mixture checkpoints, and prepared the per-seed analysis and figure.

Codex sourced, validated and split the published tournament rosters, implemented per-battle learner team sampling, ran the roster-transfer study, and prepared its results. A held-out integration check exposed Shadow Tag’s private re-selection request. Codex updated the simulator bridge and added a regression test for that turn.

Codex reconstructed the opponent study’s early learning curves and paired setup replays on Sketch5, computed win-rate intervals, reviewed divergent games, and tested first-turn action swaps.

Codex implemented PPO fine-tuning from saved Sun weights, ran the tournament adaptation and Sun retention study on Sketch5, verified the matched scratch trajectories, and prepared its curves, paired comparisons and report.

Codex implemented and tested decision-history Transformer policies, ran the one- and eight-chunk PPO comparison on Sketch5, and prepared the held-out-team analysis against the archived feedforward policies.

Codex configured the wider PPO agents, implemented and tested full-battle attention, ran the matched size and history comparisons on Sketch5, and designed and scored the counterfactual targeting probe for the mechanics analysis.

Codex reconstructed intermediate full-history checkpoints, replayed the held-out games on Sketch5, tested target, move-order and visible-history changes, and drafted the action analysis with examples from simulator logs.

Codex implemented per-battle move-slot shuffling and its simulator regression test, ran the paired full-history PPO training and evaluations on Sketch5, and prepared the familiar-team, transfer and unaided-targeting analyses.

Codex built and calibrated the fixed-team opponent league, trained and selected the PPO candidates on Sketch5, implemented the opening-based expert router, and audited the held-out game schedules and exported inference weights.

Codex fine-tuned the Transformer on four additional teams, trained the mirror and fixed-lineup PPO challengers on Sketch5, ran the reserved-seed and sampled-action evaluations, and wrote the counter-training report. Codex verified the exported policies against the training checkpoints by replaying all 16,000 confirmation games.

Codex evaluated the frozen challengers and prior policies on the original seven test teams against the earlier bots, identified the two rosters reused during counter-training, and reported transfer on the five remaining unseen teams. The 156,800-game study ran on Sketch5, with matched seeds, both action-selection modes, paired confidence intervals and reproducible evaluation scripts.

Codex then evaluated the original Transformer and both single-network challengers while holding their own lineup fixed to MC461 and assigning the seven test rosters to the previous bots. All 75,600 games ran on Sketch5. Codex audited the schedule and team assignments, replayed twelve games, and prepared the paired confidence intervals and report.

Codex audited three training trajectories and ran a two-seed PPO opponent-mixture pilot on Sketch5. It evaluated intermediate checkpoints, compared sampled and greedy actions on matched games, plotted the results, and drafted the interpretation and proposed follow-up experiment.

Codex implemented simultaneous closed-sheet lineup selection and its simulator check, ran the 90-lineup sweeps with the MC461 Transformer frozen, fit the preview head from training-roster outcomes, and compared locked choices on Sketch5. It compiled the matchup guide, paired game outcomes and uncertainty estimates.

Codex sourced and validated 45 additional tournament rosters, ran their lineup sweeps on Sketch5, corrected the setup pilot’s handling of a forced recharge, refit the preview head, and compared its held-out picks with the previous head and original four. It generated the paired results, validation analysis, figure and report.

Codex integrated the VGC-Bench behavior-cloned player and heuristic baselines, selected and checked 12 external rosters, and ran matched cross-play on Sketch5. It adapted MC461 to M-B, added open-sheet support, and implemented the four-seed PPO training and evaluation.

Codex added the frozen VGC-Bench behavior-cloned opponent to training, ran the matched four-seed experiment on Sketch5, and prepared the held-out comparison and model exports.

Codex implemented and ran the M-B post-training pilot on Sketch5, including BC-initialized battle PPO, the shared action scorer, paired PPO and group-relative preview training, roster-transfer and reciprocal evaluations, and the targeting and replay audits. Codex prepared the pilot report and figure.

Codex implemented and ran the four-seed HP and opposing-knockout reward ablation with PPO and a joint-action AWAC implementation on Sketch5. It audited the matched roster schedules and exported policies, analyzed reciprocal win rates and sweep losses, and produced the experiment report and figure.

The generated code and research interpretations still require review by Joshua Liu and Carol Li. This file does not represent a claim that either author has completed that review. No text or results should be represented as independently human-authored if they were generated with assistance.

For the final report, update this record with subsequent tools, affected components, human revisions, and verification performed. Cite original research separately from AI assistance. This repository disclosure does not update any previously submitted proposal or replace a course-specific disclosure requirement.
