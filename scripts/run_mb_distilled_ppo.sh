#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/distill.lock"
flock -n 9 || exit 0
exec 8>"$run/planned-battle.lock"
flock 8
pids=()
for s in 2 3 4 5; do
    .venv/bin/python -u scripts/mb_joint_mechanics_ppo.py --seed "$s" --steps 16384 --distill >"$run/logs/distill-$s.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then exit "$status"; fi
.venv/bin/python -u scripts/mb_policy_iteration_eval.py distilled-ppo-16384 --workers 4
.venv/bin/python -u scripts/mb_policy_iteration_eval.py distilled-ppo-16384 --split custom-dev --workers 4
