# Experiment plan

## Question and hypothesis

Does diversity in training opponents improve generalization of a fixed battle policy? We expect a 50/50 mixture of damage-oriented and switch-oriented opponents to reduce specialization, possibly at the cost of performance against either training style alone. A null or negative result is still useful if the setup and limitations are clear.

This first study measures **opponent-style generalization on familiar teams**. It does not measure generalization to new teams, the entire Champions metagame, or human opponents.

## Freeze the task first

Use the locked Champions VGC ruleset, learner `sun.txt`, opponent pool `balance.txt` and `sun.txt`, fixed first-four selection/first-two leads, terminal rewards, and the same feature schema. Keep the policy feedforward for this comparison. Do not change teams, rewards, or observation features between training conditions.

The starter opponents differ in both style and strength. Measure their cross-play results before interpreting the study; we cannot attribute a gain uniquely to diversity if mixing merely changes average training difficulty. Keep the team distribution identical in all conditions.

| Training condition | Opponent selection | Proposed environment-step budget | Training seeds |
| --- | --- | --- | --- |
| Damage only | `damage` every battle | 262,144 | 0, 1, 2 |
| Switch only | `switch` every battle | 262,144 | 0, 1, 2 |
| Mixed | `damage` or `switch`, uniformly per battle | 262,144 | 0, 1, 2 |

Budget is a starting proposal, to be frozen after development learning curves. Match **learner decision steps**, not number of battles or wall-clock time. Log the actual battles and turns too, because the policies can induce different episode lengths. Three seeds are the minimum; five is preferable if time permits.

## Development and final evaluation

- Development: short runs and simulator seeds derived from user seeds 10,000–10,099. Inspect `random`, `damage`, and `switch` opponents. Pilot results are development observations.
- Final: reserve user seeds 500,000–500,099 and the `setup` opponent until hyperparameters, team files, stopping rule, and checkpoint choice are fixed. The supplied sweep runs this final evaluation, so do not run it casually during tuning.
- Use the final checkpoint for every condition. If selecting earlier checkpoints later, choose them on a separate validation set and apply the same rule to every condition.
- Evaluate 100 battles **per opponent, per training seed, per condition** against random, damage, switch, and setup. Use stochastic policy sampling by default, with the same per-battle random seed schedule for each checkpoint. Deterministic evaluation can be a separately labeled sensitivity check.
- Pair simulator seed and opponent-team schedules across conditions. Different action histories still produce different game trajectories; paired seeds do not make the games identical.

The `setup` heuristic is a modest held-out behavior, not a strong adversary. Inspect its definition before freezing the protocol, but do not repeatedly inspect final performance while tuning. Broader evidence needs additional independently designed opponents and reserved team rosters.

## Measurements and analysis

Primary outcome: mean win rate against the reserved setup opponent, averaged across training seeds. Secondary outcomes: each training style, their macro-average, random-opponent performance, and variation across training seeds.

Keep wins, losses, ties, simulator failures, and truncations separate. The harness reports win rate on completed games and truncation rate separately. If truncations occur materially, also report wins divided by all scheduled battles and investigate the failure mode rather than silently dropping it.

The CLI's Wilson intervals quantify battle-sampling uncertainty for one checkpoint. They **do not** quantify variation from retraining. For the report, show every training-seed result, mean and standard deviation across seeds, and paired seed-level differences between mixed and each single-style condition. With only three training seeds, avoid strong significance claims.

For a figure, plot training win rate or return against learner steps, labeled as training performance. Plot final per-opponent evaluation separately. Review several wins and losses per condition to explain behavior; do not select only favorable replays.

## Milestones

1. **Environment:** legal teams, reproducible local battles, no hidden-information shortcut, and 1,000 battles without simulator errors.
2. **Learning pilot:** checkpoint creation/reloading, finite losses, and a development comparison against a random baseline. Improvement is helpful evidence, not required for accepting the infrastructure.
3. **Controlled study:** freeze the protocol, run all conditions and seeds, save raw records, then analyze once.
4. **One extension if justified:** recurrent memory or a stronger opponent population. Keep the original study as a reference and change one variable at a time.

Suggested division: one person owns simulator/features/opponents and replay review; the other owns training/evaluation/analysis. Both review the split and interpretation. Joshua and Carol can choose ownership; no work has been assigned or messaged externally.

## Before spending GPU time

Measure simulator decisions/second, training updates/second, and end-to-end throughput locally. The small MLP may be faster on CPU at this batch size. Scale vectorized environment collection before increasing model size. No paid cloud jobs are part of this repository setup.
