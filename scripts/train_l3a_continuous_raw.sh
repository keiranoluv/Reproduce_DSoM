#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/student/FB15K-237/l3a_continuous_raw"

TEACHER="$ROOT/runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth"

test -f "$TEACHER" || {
    echo "Teacher checkpoint not found" >&2
    exit 1
}

mkdir -p "$OUT/checkpoints"

cd "$ROOT/src"

export DSOM_WANDB=0
export PYTHONUNBUFFERED=1

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
  --batch_size 16 \
  --max_epochs 1 \
  --valid 1 \
  --max_train_examples 2048 \
  --eval_queries 1024 \
  --optimizer Adagrad \
  --learning_rate 0.1 \
  --policy_optimizer Adam \
  --policy_lr 0.00001 \
  --decay1 0.9 \
  --decay2 0.999 \
  --policy_temp 1 \
  --rc_reward_mode continuous_ce \
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
  --note l3a_continuous_raw \
  --ckpt_dir "$OUT/checkpoints" \
  > "$OUT/train.log" 2>&1

echo "Experiment L2 completed"
