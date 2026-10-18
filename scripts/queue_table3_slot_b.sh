#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[Slot B] Starting R4"
bash scripts/train_table3_r4_input_norm.sh --full

echo "[Slot B] R4 completed. Starting R6"
bash scripts/train_table3_r6_temp2.sh --full

echo "[Slot B] R6 completed. Queue finished."
