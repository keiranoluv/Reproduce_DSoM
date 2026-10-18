#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

R3_LOG="runs/student/FB15K-237/table3_r3_scheduler_max/full/train.log"
MAX_WAIT=43200
WAITED=0

echo "[Slot A] Waiting for R3..."

while true; do
    if [[ -f "$R3_LOG" ]] &&
       grep -q 'Test result on best Valid MRR model:' "$R3_LOG"; then
        echo "[Slot A] R3 final test result found."
        break
    fi

    if (( WAITED >= MAX_WAIT )); then
        echo "[Slot A] Timeout waiting for R3." >&2
        exit 1
    fi

    sleep 60
    WAITED=$((WAITED + 60))
done

echo "[Slot A] Starting R5"
bash scripts/train_table3_r5_batch1000.sh --full

echo "[Slot A] R5 completed. Starting R7"
bash scripts/train_table3_r7_continuous_ce.sh --full

echo "[Slot A] R7 completed. Queue finished."
