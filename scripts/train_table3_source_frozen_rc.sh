#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---smoke}"

TEACHER="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

case "$MODE" in
    --smoke)
        NAME="smoke"
        MAX_EPOCHS=1
        TRAIN_EXAMPLES=1000
        EVAL_QUERIES=256
        VALID_INTERVAL=1
        ;;
    --full)
        NAME="full"
        MAX_EPOCHS=200
        TRAIN_EXAMPLES=0
        EVAL_QUERIES=0
        VALID_INTERVAL=3
        ;;
    *)
        echo "Usage: bash $0 [--smoke|--full]"
        exit 1
        ;;
esac

if [[ ! -f "$TEACHER" ]]; then
    echo "Missing Teacher checkpoint: $TEACHER"
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="table3_source_frozen_rc_${NAME}_${STAMP}"
OUT="$ROOT/runs/student/FB15K-237/table3_source_frozen_rc/$NAME/$STAMP"

if [[ -e "$OUT" ]]; then
    echo "Run directory already exists: $OUT"
    exit 1
fi

mkdir -p "$OUT/checkpoints"

export DSOM_WANDB="${DSOM_WANDB:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_NAME="$RUN_NAME"
export WANDB_DIR="$OUT"
export DSOM_RUN_DIR="$OUT"
export PYTHONUNBUFFERED=1

{
    echo "run_name=$RUN_NAME"
    echo "mode=$MODE"
    echo "experiment=source_frozen_rc"
    echo "policy_optimizer=source_frozen"
    echo "batch_size=1000"
    echo "max_epochs=$MAX_EPOCHS"
    echo "train_examples=$TRAIN_EXAMPLES"
    echo "eval_queries=$EVAL_QUERIES"
    echo "teacher=$TEACHER"
    echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
    echo "wandb=$DSOM_WANDB"
} > "$OUT/run_metadata.txt"

echo "Experiment: Source-frozen RC"
echo "Mode: $MODE"
echo "Output: $OUT"
echo "Log: $OUT/train.log"

CMD=(
    "$ROOT/dsom/bin/python" -u kd_learn.py
    --dataset FB15K-237
    --teacher_checkpoint "$TEACHER"
    --kd_loss_type kdrl
    --teacher_strategy rc
    --kd_ce_weight 1
    --kd_kl_weight 2
    --kd_temp 4
    --dkd_alpha 1
    --dkd_beta 1
    --rank 2000
    --batch_size 1000
    --max_epochs "$MAX_EPOCHS"
    --valid "$VALID_INTERVAL"
    --max_train_examples "$TRAIN_EXAMPLES"
    --eval_queries "$EVAL_QUERIES"
    --optimizer Adagrad
    --learning_rate 0.1
    --policy_optimizer source_frozen
    --policy_temp 1
    --rl_hs 1024
    --rl_reward 1
    --rl_penalize 10
    --rl_weight 1
    --rc_reward_mode binary
    --regularizer N3
    --reg 0.01
    --init 0.001
    --scheduler_mode min
    --scale 20
    --alpha 1
    --early_stopping 10
    --note "$RUN_NAME"
    --ckpt_dir "$OUT/checkpoints"
)

if ! (
    cd "$ROOT/src"
    "${CMD[@]}"
) > "$OUT/train.log" 2>&1; then
    echo "FAILED: $OUT/train.log"
    tail -n 80 "$OUT/train.log"
    exit 1
fi

echo "COMPLETED: $MODE"
echo "LOG_PATH=$OUT/train.log"
tail -n 25 "$OUT/train.log"
