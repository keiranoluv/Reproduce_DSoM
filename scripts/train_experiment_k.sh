#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

export DSOM_WANDB=0

TEACHER="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

OUT="$ROOT/runs/student/FB15K-237/experiment_k"

test -f "$TEACHER" || {
    echo "Teacher checkpoint not found" >&2
    exit 1
}

"$ROOT/dsom/bin/python" -m py_compile \
    kd.py kd_learn.py kd_optimizers.py

COMMON=(
    --dataset FB15K-237
    --teacher_checkpoint "$TEACHER"
    --kd_ce_weight 1
    --kd_kl_weight 2
    --kd_temp 4
    --dkd_alpha 1
    --dkd_beta 1
    --rank 2000
    --batch_size 16
    --max_epochs 5
    --valid 1
    --max_train_examples 0
    --eval_queries 1024
    --optimizer Adagrad
    --learning_rate 0.1
    --regularizer N3
    --reg 0.01
    --init 0.001
    --scheduler_mode min
    --scale 20
    --alpha 1
    --early_stopping 10
)

run() {
    local name="$1"
    shift

    local DIR="$OUT/$name"

    if [[ -e "$DIR/train.log" ]]; then
        echo "Existing log: $DIR/train.log" >&2
        exit 1
    fi

    mkdir -p "$DIR/checkpoints"

    echo "Starting $name"

    "$ROOT/dsom/bin/python" -u kd_learn.py \
        "${COMMON[@]}" \
        --note "$name" \
        --ckpt_dir "$DIR/checkpoints" \
        "$@" \
        > "$DIR/train.log" 2>&1

    echo "Completed $name"
}

run k1_description \
    --kd_loss_type fixed_ndkd \
    --teacher_strategy description

run k2_learned_rc \
    --kd_loss_type kdrl \
    --teacher_strategy rc \
    --policy_optimizer Adam \
    --policy_lr 0.00001 \
    --policy_temp 1 \
    --rl_hs 1024 \
    --rl_reward 1 \
    --rl_penalize 10 \
    --rl_weight 1

echo "Experiment K pilot completed"
