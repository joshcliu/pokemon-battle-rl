#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/mechanics.lock"
flock -n 9 || exit 0
# Reuse the control evaluator's CPU allocation when it finishes.
exec 8>"$run/baseline.lock"
flock 8
for attempt in 1 2 3; do
    if .venv/bin/python -u scripts/mb_mechanics_es.py train --jobs 8; then break; fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
done
.venv/bin/python -u scripts/mb_policy_iteration_eval.py es-initial es-trained --workers 8
.venv/bin/python -u scripts/mb_policy_iteration_eval.py es-initial es-trained --split custom-dev --workers 8
