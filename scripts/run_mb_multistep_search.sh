#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/multistep.lock"
flock -n 9 || exit 0
# Reuse the eight workers when the one-turn validation and Sun Room scores finish.
exec 8>"$run/midgame.lock"
flock 8
(
    .venv/bin/python -u scripts/mb_multistep_value.py train --jobs 4 &&
    .venv/bin/python -u scripts/mb_policy_iteration_eval.py multistep-es --workers 4 &&
    .venv/bin/python -u scripts/mb_policy_iteration_eval.py multistep-es --workers 4 --split custom-dev
) > "$run/logs/multistep-fit.log" 2>&1 &
learn_pid=$!
(
    .venv/bin/python -u scripts/mb_public_preview_search.py validation --battle belief-es --jobs 4 &&
    .venv/bin/python -u scripts/mb_public_preview_search.py custom-dev --battle belief-es --jobs 4 &&
    .venv/bin/python -u scripts/mb_policy_transfer.py belief-es search-belief-es --jobs 4
) > "$run/logs/belief-transfer.log" 2>&1 &
eval_pid=$!
status=0
wait "$learn_pid" || status=1
wait "$eval_pid" || status=1
exit "$status"
