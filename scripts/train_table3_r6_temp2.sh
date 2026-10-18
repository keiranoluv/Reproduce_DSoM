#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:---smoke}"

TEACHER="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

case "$MODE" in
    --smoke)
        NAME="smoke"
        MAX_EPOCHS=1
        TRAIN_EXAMPLES=2048
        EVAL_QUERIES=1024
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

OUT="$ROOT/runs/student/FB15K-237/table3_r6_temp2/$NAME"

if [[ ! -f "$TEACHER" ]]; then
    echo "Teacher checkpoint not found"
    exit 1
fi

if [[ -e "$OUT/train.log" ]]; then
    echo "Existing run detected: $OUT"
    echo "Refusing to overwrite."
    exit 1
fi

mkdir -p "$OUT/checkpoints"

cd "$ROOT/src"

if [[ "$MODE" == "--full" ]]; then
    export DSOM_WANDB=1
    export WANDB_PROJECT="DSoM-Reproduction"
    export WANDB_NAME="table3_r6_temp2_full"
else
    export DSOM_WANDB=0
fi

export PYTHONUNBUFFERED=1

echo "Mode: $MODE"
echo "Output: $OUT"
echo "Teacher: $TEACHER"

"$ROOT/dsom/bin/python" -u kd_learn.py \
    --dataset FB15K-237 \
    --teacher_checkpoint "$TEACHER" \
    --kd_loss_type kdrl \
    --teacher_strategy rc \
    --kd_ce_weight 1 \
    --kd_kl_weight 2 \
    --kd_temp 4 \
    --dkd_alpha 1 \
    --dkd_beta 1 \
    --rank 2000 \
    --batch_size 1024 \
    --max_epochs "$MAX_EPOCHS" \
    --valid "$VALID_INTERVAL" \
    --max_train_examples "$TRAIN_EXAMPLES" \
    --eval_queries "$EVAL_QUERIES" \
    --optimizer Adagrad \
    --learning_rate 0.1 \
    --policy_optimizer same \
    --policy_temp 2 \
    --rl_hs 1024 \
    --rl_reward 1 \
    --rl_penalize 10 \
    --rl_weight 1 \
    --regularizer N3 \
    --reg 0.01 \
    --init 0.001 \
    --scheduler_mode min \
    --scale 20 \
    --alpha 1 \
    --early_stopping 10 \
    --note "table3_r6_temp2_$NAME" \
    --ckpt_dir "$OUT/checkpoints" \
    > "$OUT/train.log" 2>&1

echo "R6_TEMP2 completed: $MODE"
