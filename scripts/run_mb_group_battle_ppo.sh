#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-policy-iteration
exec 9>"$run/group.lock"
flock -n 9 || exit 0
for limit in 16 64; do
    pids=()
    for seed in 2 3 4 5; do
        (
            for attempt in 1 2 3; do
                if .venv/bin/python -u scripts/mb_group_battle_ppo.py --seed "$seed" --updates "$limit"; then exit 0; fi
            done
            exit 1
        ) >> "$run/logs/group-$seed.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    if [[ "$status" != 0 ]]; then exit 1; fi
    .venv/bin/python -u scripts/mb_policy_iteration_eval.py "group-$limit-map" "group-$limit-native" --workers 6
done
.venv/bin/python -u scripts/mb_policy_iteration_eval.py group-64-map group-64-native --split custom-dev --workers 6
