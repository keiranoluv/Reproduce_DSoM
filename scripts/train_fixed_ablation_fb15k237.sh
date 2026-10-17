#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

export DSOM_WANDB=0

TEACHER_CKPT="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

OUT="$ROOT/runs/student/FB15K-237/fixed_ablation/medium"

test -f "$TEACHER_CKPT" || {
    echo "Teacher checkpoint missing: $TEACHER_CKPT" >&2
    exit 1
}

"$ROOT/dsom/bin/python" -m py_compile \
    kd.py kd_learn.py kd_optimizers.py

# Common configuration for fair comparison.
COMMON_ARGS=(
    --dataset FB15K-237
    --teacher_checkpoint "$TEACHER_CKPT"
    --kd_ce_weight 1
    --kd_kl_weight 2
    --kd_temp 4
    --dkd_alpha 1
    --dkd_beta 1
    --rank 2000
    --batch_size 16
    --max_epochs 3
    --valid 1
    --max_train_examples 20000
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

run_experiment() {
    local name="$1"
    shift

    local dir="$OUT/$name"
    mkdir -p "$dir/checkpoints"

    # Prevent accidental reruns from silently mixing results.
    if [[ -f "$dir/train.log" ]]; then
        echo "Existing log found: $dir/train.log" >&2
        echo "Move or rename that run directory before rerunning." >&2
        exit 1
    fi

    echo "======================================="
    echo "Starting $name"
    echo "======================================="

    "$ROOT/dsom/bin/python" -u kd_learn.py \
        "${COMMON_ARGS[@]}" \
        --note "$name" \
        --ckpt_dir "$dir/checkpoints" \
        "$@" \
        2>&1 | tee "$dir/train.log"

    echo "Finished $name"
}

run_experiment j0_structural \
    --kd_loss_type fixed_ndkd \
    --teacher_strategy structural

run_experiment j1_description \
    --kd_loss_type fixed_ndkd \
    --teacher_strategy description

run_experiment j2_average \
    --kd_loss_type fixed_ndkd \
    --teacher_strategy average

run_experiment j3_learned_rc \
    --kd_loss_type kdrl \
    --teacher_strategy rc \
    --rl_hs 1024 \
    --rl_reward 1 \
    --rl_penalize 10 \
    --rl_weight 1 \
    --policy_optimizer Adam \
    --policy_lr 0.00001 \
    --policy_temp 1

echo "======================================="
echo "All Experiment J runs completed"
echo "======================================="
