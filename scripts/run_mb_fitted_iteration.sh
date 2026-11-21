#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/fitted.lock"
flock -n 9 || exit 0
# The collector is separately resumable. Take over its lock after it exits,
# then retry any missing shards before fitting a complete run.
for attempt in 1 2 3; do
    if flock "$run/branch.lock" .venv/bin/python -u scripts/mb_rollout_branches.py --blocks 40 --workers 12; then
        break
    fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
done
.venv/bin/python -u scripts/mb_counterfactual.py --blocks 40
.venv/bin/python -u scripts/mb_policy_iteration_eval.py mcq-40-2 mcq-40-8 --workers 8
.venv/bin/python -u scripts/mb_policy_iteration_eval.py mcq-40-2 mcq-40-8 --split custom-dev --workers 8
