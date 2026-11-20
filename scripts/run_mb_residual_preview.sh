#!/usr/bin/env bash
# Run from the configured project checkout on Sketch5.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.cache/vgcbench-external/packages:$PWD/.cache/vgcbench-external/source"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$PWD/.cache/matplotlib"
run=runs/sketch5-mb-preview-ppo
mkdir -p "$run/logs"
exec 9>"$run/driver.lock"
flock -n 9 || { echo "A preview training driver is already running."; exit 1; }
.venv/bin/python -u scripts/mb_residual_preview.py check
run_seed() {
    local seed="$1" stage attempt
    for stage in train evaluate; do
        for attempt in 1 2 3; do
            if .venv/bin/python -u scripts/mb_residual_preview.py "$stage" --seed "$seed" --jobs "${PREVIEW_WORKERS:-6}"; then
                break
            fi
            if [[ "$attempt" == 3 ]]; then return 1; fi
            echo "Retrying $stage for seed $seed from the last complete batch."
        done
    done
}
pids=()
for seed in 2 3 4 5; do
    run_seed "$seed" >> "$run/logs/seed-$seed.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then exit "$status"; fi
.venv/bin/python -u scripts/report_mb_residual_preview.py
