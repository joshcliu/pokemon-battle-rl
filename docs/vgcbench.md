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
