#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_teacher_fb15k237.sh [--smoke|--full] [--no-wandb]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../src/learn.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/src/learn.py" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Cannot find src/learn.py; put this file under scripts/ or repo root" >&2
    exit 1
fi

MODE=full
USE_WANDB=1
for arg in "$@"; do
    case "$arg" in
        --smoke) MODE=smoke ;;
        --full) MODE=full ;;
        --no-wandb) USE_WANDB=0 ;;
        *) echo "Usage: $0 [--smoke|--full] [--no-wandb]" >&2; exit 2 ;;
    esac
done
if [[ "$MODE" == smoke ]]; then
    MAX_EPOCHS=1
    VALID=1
else
    MAX_EPOCHS=200
    VALID=3
fi

PYTHON="${REPO_ROOT}/dsom/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing Python environment: $PYTHON" >&2; exit 1; }
DATA_DIR="${REPO_ROOT}/data/FB15K-237"
for f in train.pickle valid.pickle test.pickle to_skip.pickle img_vec.pickle dscp_vec.pickle; do
    [[ -f "${DATA_DIR}/${f}" ]] || { echo "Missing ${DATA_DIR}/${f}" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="teacher_fb15k237_${MODE}_${STAMP}"
RUN_DIR="${REPO_ROOT}/runs/teacher/FB15K-237/${MODE}/${STAMP}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/train.log"
mkdir -p "$CKPT_DIR"

export DSOM_WANDB="$USE_WANDB"
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="$RUN_NAME"
export WANDB_DIR="$RUN_DIR"
export DSOM_RUN_DIR="$RUN_DIR"

# Check W&B dependencies and logging integration
if [[ "$USE_WANDB" == 1 ]]; then
    if ! "$PYTHON" -c 'import wandb'; then
        echo "Error: wandb is not installed." >&2
        echo "Install using: ${PYTHON} -m pip install wandb" >&2
        exit 1
    fi

    if ! grep -q 'DSOM_WANDB' "${REPO_ROOT}/src/learn.py"; then
        echo "Error: src/learn.py does not support DSOM_WANDB logging." >&2
        exit 1
    fi
fi

{
    echo "run_name=$RUN_NAME"
    echo "mode=$MODE"
    echo "max_epochs=$MAX_EPOCHS"
    echo "valid_interval=$VALID"
    echo "wandb=$USE_WANDB"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "timestamp=$STAMP"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
} > "${RUN_DIR}/run_metadata.txt"

cd "${REPO_ROOT}/src"  # datasets.py uses ../data
printf 'Run: %s\nOutput: %s\nW&B: %s\n' "$RUN_NAME" "$RUN_DIR" "$USE_WANDB"

"$PYTHON" -u learn.py \
    --dataset FB15K-237 \
    --model ComplExMDR \
    --modality_split True \
    --fusion_img True \
    --fusion_dscp True \
    --img_info "${DATA_DIR}/img_vec.pickle" \
    --dscp_info "${DATA_DIR}/dscp_vec.pickle" \
    --modal_missing 0.0 \
    --rank 2000 \
    --batch_size 1000 \
    --max_epochs "$MAX_EPOCHS" \
    --valid "$VALID" \
    --optimizer Adagrad \
    --learning_rate 0.1 \
    --regularizer N3 \
    --reg 0.01 \
    --init 0.001 \
    --early_stopping 10 \
    --scale 20 \
    --alpha 1 \
    --note "$RUN_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    2>&1 | tee "$LOG_FILE"
