# Reproducing VGC-Bench cross-play on Sketch5

The project uses a pinned upstream player, checkpoint and vocabulary. The M-C test uses our existing simulator and closed sheets. The M-B test trains under M-B on the project simulator, then evaluates under M-B with the benchmark’s pinned simulator and open sheets. Our learner pilots one team and our tests report matchup win rates. The official evaluation constructs a population payoff matrix over multiple agents and runs. [Upstream evaluation](https://github.com/cameronangliss/vgc-bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/eval.py) · [Paper](https://arxiv.org/abs/2506.10326)

Run these commands from a project checkout with its analysis dependencies, Python 3.12 and the simulator already installed. The original training checkpoints live under `runs/sketch5-move-slots-v1/seed-{2,3,4,5}/fixed/train/checkpoint-262144.pt` on Sketch5. To rebuild those initializations, use the repository's `scripts/train_move_slots.py`.

```sh
cache="$PWD/.cache/vgcbench-external"
mkdir -p "$cache"
git clone https://github.com/cameronangliss/vgc-bench.git "$cache/source"
git -C "$cache/source" checkout --detach d79f9532947ac114dce1dda2456a590afcd375b2
curl -fL https://huggingface.co/cameronangliss/vgc-bench-models/resolve/204c76741829ca0681629e41382043c385850d5c/results/saves_bc/seed1/100.zip \
  -o "$cache/vgcbench-bc100-mc461-20261118.zip"
uv pip install --python .venv/bin/python --target "$cache/packages" --no-deps \
  stable-baselines3==2.8.0 gymnasium==1.2.3 pandas==2.2.3 pytz==2025.2 tzdata==2025.2
git clone https://github.com/cameronangliss/pokemon-showdown.git "$cache/showdown"
git -C "$cache/showdown" checkout --detach 913da3602a3aa1db79f9fdc5d5222eaf8d39569d
ln -s "$PWD/.cache/pokemon-showdown/node_modules" "$cache/showdown/node_modules"
(cd "$cache/showdown" && ../../node-runtime/node_modules/node/bin/node build)
```

The historical M-C experiment can run against the 12 committed exports as-is. To recreate that selection, run the team importer with the pinned VGC-Bench source checkout. It reads only the original regional rosters, expanded regional rosters and two fixtures for deduplication. The evaluator checks the released weights, vocabulary, actor source and heuristic implementation before playing. Every eight-game batch is written atomically.

```sh
.cache/node-runtime/node_modules/node/bin/node scripts/import_vgcbench_teams.cjs "$cache/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$cache/packages:$cache/source"
.venv/bin/python scripts/evaluate_vgcbench.py --smoke
.venv/bin/python scripts/evaluate_vgcbench.py --jobs 32
.venv/bin/python scripts/evaluate_vgcbench.py --verify
.venv/bin/python scripts/report_vgcbench.py
```

For M-B, run training in the standard project environment, then run cross-play with the upstream Python overlay. Training uses the four archived initialization seeds and the damage, tactical and frozen PPO league pilots. The 12 public opposing teams and BC weights are reserved for evaluation. The evaluator validates both public sheets before preview and uses the native opponent's selected four. The learner brings the four listed on the legal variant.

```sh
env -u PYTHONPATH .venv/bin/python scripts/train_vgcbench_mb.py
.venv/bin/python scripts/evaluate_vgcbench_mb.py --smoke
for steps in 0 16384 65536; do
  .venv/bin/python scripts/evaluate_vgcbench_mb.py --step "$steps" --jobs 24
  .venv/bin/python scripts/evaluate_vgcbench_mb.py --step "$steps" --verify
done
.venv/bin/python scripts/report_vgcbench_mb.py
```

To evaluate another legal team and an exported PPO checkpoint, use a fresh output directory. This command runs 64 games per team and opponent, writes the individual outcomes and reports a win rate and Wilson interval for each opponent.

```sh
.venv/bin/python scripts/evaluate_vgcbench_mb.py \
  --checkpoint results/sketch5-vgcbench-mb/weights/seed-2.pt \
  --team teams/reg-mb-variant/MC461-MB.txt --out runs/mb-export-repeat --jobs 24
```

## Training against the published behavior-cloned player

With the same upstream Python overlay and single-thread settings as cross-play, the following commands repeat the four-seed opponent comparison. The M-B variant, tournament initialization for each seed, 65,536-step budget, training simulator and four training rosters match the league experiment. The frozen upstream policy samples its own lineup and battle actions. Its random-number stream is separate from PPO. Each evaluation uses the previous 12 opposing rosters and their original side and seed schedule.

```sh
.venv/bin/python scripts/train_vgcbench_mb.py --bc
.venv/bin/python scripts/compare_vgcbench_bc.py evaluate --steps 16384
.venv/bin/python scripts/compare_vgcbench_bc.py evaluate --steps 65536
.venv/bin/python scripts/compare_vgcbench_bc.py report
```

The comparison reads the previous scores from `results/sketch5-vgcbench-mb/games.csv.gz` and checks the original training manifests and initial policy tensors on Sketch5. It writes matched outcomes, per-seed and per-team effects, and all four final policies to `results/sketch5-vgcbench-bc`.

## M-B battle-policy and preview post-training pilot

The pilot uses the pinned benchmark simulator for both training and evaluation. Use the Python overlay and single-thread settings above. All commands below run in the project checkout on Sketch5. The roster manifest records the upstream paths and hashes. The previous twelve teams are a separate MC461 development schedule.

```bash
.cache/node-runtime/node_modules/node/bin/node scripts/lock_mb_pilot.cjs
.venv/bin/python scripts/vgc_posttrain.py baseline --jobs 10
for s in 2 3 4 5; do
  for arm in native flat structured; do
    .venv/bin/python scripts/vgc_posttrain.py train --arm "$arm" --seed "$s"
  done
  for arm in ppo grpo; do
    .venv/bin/python scripts/vgc_posttrain.py preview --arm "$arm" --seed "$s"
  done
done
.venv/bin/python scripts/run_mb_matrix.py --jobs 16 bc bc-native \
  history-{2,3,4,5} native-{2,3,4,5} flat-{2,3,4,5} structured-{2,3,4,5} \
  native-preview-{2,3,4,5} preview-ppo-{2,3,4,5} preview-grpo-{2,3,4,5}
.venv/bin/python scripts/eval_mb_development.py
.venv/bin/python scripts/mb_reciprocal.py
.venv/bin/python scripts/probe_mb_posttraining.py
.venv/bin/python scripts/verify_mb_posttraining.py
.venv/bin/python scripts/report_mb_posttraining.py
.venv/bin/python scripts/plot_mb_posttraining.py
```

The training jobs are independent across arms and seeds. The evaluation commands read completed checkpoints under `runs/sketch5-mb-posttraining-pilot`. The `native-preview` condition combines a trained native battle policy with the released BC preview. The history control uses the earlier 65k M-B weights and samples legal battle actions, matching the deployment mode of the post-training pilot. See the [pilot results](../results/sketch5-mb-posttraining-pilot/README.md) for the lineup rules, primary comparison and roster-swap diagnostic.

## HP and opponent-survival reward ablation

Use the same pinned simulator, model, teams, Python overlay and single-thread settings on Sketch5. The [reward protocol](../results/sketch5-mb-reward-factorial/protocol.json) defines the two independent reward factors, optimizer settings and evaluation schedule. Each training command below is independent.

```bash
for algo in ppo awac; do
  for reward in none hp alive both; do
    for seed in 2 3 4 5; do
      .venv/bin/python scripts/mb_reward_factorial.py "$algo" "$reward" "$seed"
    done
  done
done
.venv/bin/python scripts/eval_mb_rewards.py --jobs 28
.venv/bin/python scripts/eval_mb_rewards.py --verify
.venv/bin/python scripts/report_mb_rewards.py
.venv/bin/python scripts/plot_mb_rewards.py
```

The evaluator also scores the released BC policy and tracks surviving opponents at the end of every battle. Each new policy and its training records are under `runs/sketch5-mb-reward-factorial/<algorithm>-<reward>-<seed>/`. The [report](../results/sketch5-mb-reward-factorial/README.md) links the game records, checkpoint hashes and roster matrix.

## Shared, per-team and top-2 MoE battle PPO

Run with the same pinned simulator and single-thread Python overlay on Sketch5. The [protocol](../results/sketch5-mb-roster-ppo/protocol.json) fixes budgets, rosters, preview behavior and the specialist assignment rule before scoring. The four specialist jobs collectively consume the same 65,536 decisions as each shared model.

```bash
node scripts/lock_mb_specialists.cjs
for seed in 2 3 4 5; do
  .venv/bin/python scripts/mb_roster_ppo.py shared "$seed"
  .venv/bin/python scripts/mb_roster_ppo.py moe "$seed"
  for roster in 0 1 2 3; do
    .venv/bin/python scripts/mb_roster_ppo.py specialist "$seed" --index "$roster"
  done
done
.venv/bin/python scripts/eval_mb_roster_ppo.py --jobs 20
.venv/bin/python scripts/eval_mb_roster_ppo.py --verify
.venv/bin/python scripts/report_mb_roster_ppo.py
.venv/bin/python scripts/plot_mb_roster_ppo.py
.venv/bin/python scripts/check_mb_expert_routing.py --jobs 20
```

Each trained checkpoint and its episode log are under `runs/sketch5-mb-roster-ppo/`. The evaluator records both ordered lineups in every game, the deployed specialist and the MoE expert counts.

The final command is a post hoc inference check on the held-out games. It keeps the gate and all four trained experts frozen, assigning each gate output to the next expert cyclically. The primary three-way comparison uses normal routing.

## PPO duration and expert-routing controls

Use the same frozen BC assets and roster-ppo environment on Sketch5. The archived 65k shared and original MoE weights from the preceding experiment are required for the exact-continuation check. The 65k score archives supply the original paired baseline. Every long command can be rerun after interruption and resumes at its last 16,384-decision optimizer checkpoint.

```bash
.venv/bin/python scripts/check_mb_long_state.py
for seed in 2 3 4 5; do
  for arm in shared moe moe8 moe1; do
    .venv/bin/python scripts/mb_roster_ppo.py "$arm" "$seed" --long
  done
done
.venv/bin/python scripts/eval_mb_long_ppo.py --jobs 20
.venv/bin/python scripts/eval_mb_long_ppo.py --verify
.venv/bin/python scripts/probe_mb_long_routing.py
.venv/bin/python scripts/report_mb_long_ppo.py
.venv/bin/python scripts/plot_mb_long_ppo.py
```

The [long-run protocol](../results/sketch5-mb-long-ppo/protocol.json) fixes architectural controls, paired evaluation and routing interventions. The report uses 262,144 total decisions for each final model.

## Learner roster diversity at 262k

The 16-team comparison reuses the four-team experiment's final held-out games as a paired reference. Use the same Sketch5 checkout, frozen BC weights, simulator, and environment variables. The roster lock checks the pinned bank. Each learner team has an independent battle stream. The trainer resumes every 16,384 decisions from a checkpoint of all 16 streams.

```bash
.cache/node-runtime/node_modules/node/bin/node scripts/lock_mb_diversity.cjs
for seed in 2 3 4 5; do
  for arm in shared moe; do
    .venv/bin/python scripts/mb_roster_ppo.py "$arm" "$seed" --diverse
  done
done
.venv/bin/python scripts/eval_mb_long_ppo.py --diverse --jobs 12
.venv/bin/python scripts/eval_mb_long_ppo.py --diverse --verify
.venv/bin/python scripts/report_mb_team_diversity.py
```

The [team diversity report](../results/sketch5-mb-team-diversity/README.md) lists the exact assignments and defines the paired routing comparisons.
