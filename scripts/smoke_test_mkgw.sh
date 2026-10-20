#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
EXP_ID="T1_MKGW_Exp01"
TEACHER_NAME="${EXP_ID}_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022"
STUDENT_NAME="${EXP_ID}_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"
export DSOM_WANDB=0
cd "$ROOT"
for file in data/MKG-W/img_vec.pickle data/MKG-W/dscp_vec.pickle; do
    [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 1; }
done
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
TEACHER_DIR="$ROOT/runs/teacher/MKG-W/smoke/$TEACHER_NAME/$RUN_ID"
STUDENT_DIR="$ROOT/runs/student/MKG-W/smoke/$STUDENT_NAME/$RUN_ID"
mkdir -p "$TEACHER_DIR/checkpoints" "$STUDENT_DIR/checkpoints"
GIT_COMMIT="$(git rev-parse HEAD)"

teacher_args=(
    --dataset MKG-W --model ComplExMDR
    --modality_split True --fusion_img True --fusion_dscp True
    --img_info "$ROOT/data/MKG-W/img_vec.pickle"
    --dscp_info "$ROOT/data/MKG-W/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --max_epochs 1 --valid 1
    --max_train_examples 1000 --eval_queries 64
    --ckpt_dir "$TEACHER_DIR/checkpoints"
    --note "${TEACHER_NAME}_Smoke"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)

echo "=== MKG-W TEACHER SMOKE ==="
printf '%q ' "$PYTHON" learn.py "${teacher_args[@]}" > "$TEACHER_DIR/command.sh"
(
    cd "$ROOT/src"
    "$PYTHON" learn.py "${teacher_args[@]}"
) 2>&1 | tee "$TEACHER_DIR/teacher.log"
mapfile -t teachers < <(find "$TEACHER_DIR/checkpoints" -type f -name 'm-*.pth' -print)
[[ ${#teachers[@]} == 1 ]] || { echo "Expected exactly one Teacher checkpoint" >&2; exit 1; }
TEACHER_CKPT="${teachers[0]}"

student_args=(
    --dataset MKG-W --model ComplExMDR
    --teacher_checkpoint "$TEACHER_CKPT"
    --img_info "$ROOT/data/MKG-W/img_vec.pickle"
    --dscp_info "$ROOT/data/MKG-W/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --kd_loss_type kdrl
    --teacher_strategy rc --policy_optimizer source_frozen
    --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
    --rl_hs 1024 --rl_reward 1 --rl_penalize 10
    --policy_temp 1 --rc_reward_mode binary
    --max_epochs 1 --valid 1
    --max_train_examples 1000 --eval_queries 64
    --ckpt_dir "$STUDENT_DIR/checkpoints"
    --note "${STUDENT_NAME}_Smoke"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)

echo "=== MKG-W STUDENT SMOKE ==="
printf '%s\n' "$TEACHER_CKPT" > "$STUDENT_DIR/teacher_checkpoint.txt"
printf '%q ' "$PYTHON" kd_learn.py "${student_args[@]}" > "$STUDENT_DIR/command.sh"
(
    cd "$ROOT/src"
    "$PYTHON" kd_learn.py "${student_args[@]}"
) 2>&1 | tee "$STUDENT_DIR/student.log"
grep -q 'SOURCE_FROZEN_PARAMS: PASS' "$STUDENT_DIR/student.log"

"$PYTHON" - "$TEACHER_DIR/checkpoints" "$STUDENT_DIR/checkpoints" <<'PY'
import json
import sys
from pathlib import Path
for label, p in (("Teacher", sys.argv[1]), ("Student", sys.argv[2])):
    reports = list(Path(p).rglob('result.json'))
    assert len(reports) == 1, (label, reports)
    r = json.loads(reports[0].read_text())
    assert r['dataset'] == 'MKG-W'
    assert r['best_epoch'] > 0
    assert r['max_train_examples'] == 1000
    assert r['eval_queries'] == 64
    print(f'{label} smoke PASS; best_epoch={r["best_epoch"]}')
PY
echo 'MKG-W Exp01 SMOKE PIPELINE: PASS'
