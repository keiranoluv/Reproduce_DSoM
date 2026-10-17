#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/train_teacher_fb15k237.sh --smoke
# bash scripts/train_teacher_fb15k237.sh --full
# bash scripts/train_teacher_fb15k237.sh --full --scheduler-max
# bash scripts/train_teacher_fb15k237.sh --full --scheduler-min
# bash scripts/train_teacher_fb15k237.sh --smoke --scheduler-max --no-wandb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/../src/learn.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/src/learn.py" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Cannot find src/learn.py" >&2
    exit 1
fi

# ============================================================
# Arguments
# ============================================================

MODE=full
USE_WANDB=1
SCHEDULER_MODE=min

for arg in "$@"; do
    case "$arg" in
        --smoke)
            MODE=smoke
            ;;
        --full)
            MODE=full
            ;;
        --no-wandb)
            USE_WANDB=0
            ;;
        --scheduler-min)
            SCHEDULER_MODE=min
            ;;
        --scheduler-max)
            SCHEDULER_MODE=max
            ;;
        *)
            echo "Usage: $0 [--smoke|--full] [--scheduler-min|--scheduler-max] [--no-wandb]" >&2
            exit 2
            ;;
    esac
done

if [[ "$MODE" == "smoke" ]]; then
    MAX_EPOCHS=1
    VALID=1
else
    MAX_EPOCHS=200
    VALID=3
fi

# ============================================================
# Environment and dataset
# ============================================================

PYTHON="${REPO_ROOT}/dsom/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON" >&2
    exit 1
fi

DATA_DIR="${REPO_ROOT}/data/FB15K-237"

for f in \
    train.pickle \
    valid.pickle \
    test.pickle \
    to_skip.pickle \
    img_vec.pickle \
    dscp_vec.pickle
do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        echo "Missing ${DATA_DIR}/${f}" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Verify scheduler support in learn.py
if ! grep -q -- '--scheduler_mode' "${REPO_ROOT}/src/learn.py"; then
    echo "Error: src/learn.py does not support --scheduler_mode" >&2
    echo "Add the scheduler_mode argument to learn.py first." >&2
    exit 1
fi

# ============================================================
# Experiment directories
# ============================================================

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="teacher_fb15k237_${MODE}_scheduler-${SCHEDULER_MODE}_${STAMP}"

RUN_DIR="${REPO_ROOT}/runs/teacher/FB15K-237/${MODE}/scheduler-${SCHEDULER_MODE}/${STAMP}"

CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "$CKPT_DIR"

# ============================================================
# Weights & Biases
# ============================================================

export DSOM_WANDB="$USE_WANDB"
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="$RUN_NAME"
export WANDB_DIR="$RUN_DIR"
export DSOM_RUN_DIR="$RUN_DIR"

if [[ "$USE_WANDB" == 1 ]]; then
    if ! "$PYTHON" -c 'import wandb'; then
        echo "Error: wandb is not installed." >&2
        exit 1
    fi

    if ! grep -q 'DSOM_WANDB' "${REPO_ROOT}/src/learn.py"; then
        echo "Error: src/learn.py does not support DSOM_WANDB logging." >&2
        exit 1
    fi
fi

# ============================================================
# Experiment metadata
# ============================================================

{
    echo "run_name=$RUN_NAME"
    echo "dataset=FB15K-237"
    echo "model=ComplExMDR"
    echo "mode=$MODE"
    echo "scheduler_mode=$SCHEDULER_MODE"
    echo "max_epochs=$MAX_EPOCHS"
    echo "valid_interval=$VALID"
    echo "early_stopping=10"
    echo "seed=2022"
    echo "rank=2000"
    echo "batch_size=1000"
    echo "optimizer=Adagrad"
    echo "learning_rate=0.1"
    echo "regularizer=N3"
    echo "reg=0.01"
    echo "wandb=$USE_WANDB"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "timestamp=$STAMP"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
} > "${RUN_DIR}/run_metadata.txt"

# ============================================================
# Training
# ============================================================

cd "${REPO_ROOT}/src"

echo "============================================================"
echo "DSoM Teacher Training"
echo "============================================================"
echo "Run name       : $RUN_NAME"
echo "Dataset        : FB15K-237"
echo "Mode           : $MODE"
echo "Scheduler mode : $SCHEDULER_MODE"
echo "Max epochs     : $MAX_EPOCHS"
echo "Validation     : every $VALID epoch(s)"
echo "Early stopping : 10"
echo "W&B            : $USE_WANDB"
echo "GPU            : $CUDA_VISIBLE_DEVICES"
echo "Output         : $RUN_DIR"
echo "============================================================"

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
    --scheduler_mode "$SCHEDULER_MODE" \
    --scale 20 \
    --alpha 1 \
    --note "$RUN_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    2>&1 | tee "$LOG_FILE"
