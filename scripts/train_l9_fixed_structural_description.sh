#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TEACHER="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

OUT="$ROOT/runs/student/FB15K-237/l9_fixed_structural_description"

if [[ ! -f "$TEACHER" ]]; then
    echo "Teacher checkpoint not found: $TEACHER" >&2
    exit 1
fi

if [[ -e "$OUT/train.log" ]]; then
    echo "Existing L9 log found; refusing to overwrite." >&2
    exit 1
fi

mkdir -p "$OUT/checkpoints"

cd "$ROOT/src"

export DSOM_WANDB=0
export PYTHONUNBUFFERED=1

"$ROOT/dsom/bin/python" -m py_compile \
    kd.py kd_learn.py kd_optimizers.py

echo "Starting L9: Fixed Structural+Description"

"$ROOT/dsom/bin/python" -u kd_learn.py \
    --dataset FB15K-237 \
    --teacher_checkpoint "$TEACHER" \
    --kd_loss_type fixed_ndkd \
    --teacher_strategy structural_description \
    --kd_ce_weight 1 \
    --kd_kl_weight 2 \
    --kd_temp 4 \
    --dkd_alpha 1 \
    --dkd_beta 1 \
    --rank 2000 \
    --batch_size 16 \
    --max_epochs 5 \
    --valid 1 \
    --max_train_examples 0 \
    --eval_queries 1024 \
    --optimizer Adagrad \
    --learning_rate 0.1 \
    --regularizer N3 \
    --reg 0.01 \
    --init 0.001 \
    --scheduler_mode min \
    --scale 20 \
    --alpha 1 \
    --early_stopping 10 \
    --note l9_fixed_structural_description \
    --ckpt_dir "$OUT/checkpoints" \
    > "$OUT/train.log" 2>&1

echo "Experiment L9 training completed"
echo "Log: $OUT/train.log"
