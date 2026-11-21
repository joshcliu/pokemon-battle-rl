#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/context.lock"
flock -n 9 || exit 0
# Reuse four workers after the preview/battle factorial finishes.
exec 8>"$run/combined.lock"
flock 8
pids=()
for seed in 2 3 4 5; do
    (
        for attempt in 1 2 3; do
            if .venv/bin/python -u scripts/mb_context_ppo.py --seed "$seed"; then exit 0; fi
        done
        exit 1
    ) >> "$run/logs/context-$seed.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then exit 1; fi
.venv/bin/python -u scripts/mb_policy_iteration_eval.py context-ppo --workers 4
.venv/bin/python -u scripts/mb_policy_iteration_eval.py context-ppo --split custom-dev --workers 4
