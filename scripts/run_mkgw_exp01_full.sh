#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
EXP_ID="T1_MKGW_Exp01"
TEACHER_NAME="${EXP_ID}_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022"
STUDENT_NAME="${EXP_ID}_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"
cd "$ROOT"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo 'ERROR: Commit tracked code changes before full training' >&2
    exit 1
fi
for file in \
    src/learn.py src/kd_learn.py src/process_datasets.py \
    scripts/mkgw_alignment.py scripts/prepare_mkgw_dataset.py \
    scripts/build_mkgw_text_features.py scripts/build_mkgw_image_features.py \
    scripts/smoke_test_mkgw.sh scripts/run_mkgw_exp01_full.sh; do
    git ls-files --error-unmatch "$file" >/dev/null 2>&1 || {
        echo "ERROR: Not committed: $file" >&2
        exit 1
    }
done
for file in data/MKG-W/img_vec.pickle data/MKG-W/dscp_vec.pickle; do
    [[ -f "$file" ]] || { echo "ERROR: Missing $file" >&2; exit 1; }
done
"$PYTHON" -c 'import wandb; print("W&B version", wandb.__version__)'
GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
TEACHER_RUN="$ROOT/runs/teacher/MKG-W/full/$TEACHER_NAME/$RUN_ID"
STUDENT_RUN="$ROOT/runs/student/MKG-W/full/$STUDENT_NAME/$RUN_ID"
mkdir -p "$TEACHER_RUN/checkpoints" "$TEACHER_RUN/wandb"
mkdir -p "$STUDENT_RUN/checkpoints" "$STUDENT_RUN/wandb"
printf '%s\n' "$GIT_COMMIT" > "$TEACHER_RUN/git_commit.txt"
printf '%s\n' "$GIT_COMMIT" > "$STUDENT_RUN/git_commit.txt"
export DSOM_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${RUN_ID}"
echo "=== MKG-W Exp01: commit=$GIT_COMMIT group=$WANDB_RUN_GROUP ==="

teacher_args=(
    --dataset MKG-W --model ComplExMDR
    --modality_split True --fusion_img True --fusion_dscp True
    --img_info "$ROOT/data/MKG-W/img_vec.pickle"
    --dscp_info "$ROOT/data/MKG-W/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --max_epochs 200 --valid 3 --early_stopping 10
    --ckpt_dir "$TEACHER_RUN/checkpoints" --note "$TEACHER_NAME"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)
export WANDB_NAME="$TEACHER_NAME"
export WANDB_DIR="$TEACHER_RUN/wandb"
export DSOM_RUN_DIR="$TEACHER_RUN"
printf '%q ' "$PYTHON" learn.py "${teacher_args[@]}" > "$TEACHER_RUN/command.sh"
echo "=== MKG-W TEACHER FULL ==="
(
    cd "$ROOT/src"
    "$PYTHON" learn.py "${teacher_args[@]}"
) 2>&1 | tee "$TEACHER_RUN/teacher.log"
mapfile -t teacher_ckpts < <(find "$TEACHER_RUN/checkpoints" -type f -name 'm-*.pth' -print)
[[ ${#teacher_ckpts[@]} == 1 ]] || { echo 'ERROR: Exactly one Teacher checkpoint required' >&2; exit 1; }
TEACHER_CKPT="${teacher_ckpts[0]}"

"$PYTHON" - "$TEACHER_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path
reports = list(Path(sys.argv[1]).rglob('result.json'))
assert len(reports) == 1
r = json.loads(reports[0].read_text())
assert r['dataset'] == 'MKG-W'
assert r['max_train_examples'] == 0
assert r['eval_queries'] == 0
assert r['git_commit'] == sys.argv[2]
assert r['best_epoch'] > 0
print('TEACHER FULL: PASS, best epoch:', r['best_epoch'])
PY
printf '%s\n' "$TEACHER_CKPT" > "$STUDENT_RUN/teacher_checkpoint.txt"

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
    --max_epochs 200 --valid 3 --early_stopping 10
    --ckpt_dir "$STUDENT_RUN/checkpoints" --note "$STUDENT_NAME"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)
export WANDB_NAME="$STUDENT_NAME"
export WANDB_DIR="$STUDENT_RUN/wandb"
export DSOM_RUN_DIR="$STUDENT_RUN"
printf '%q ' "$PYTHON" kd_learn.py "${student_args[@]}" > "$STUDENT_RUN/command.sh"
echo "=== MKG-W STUDENT FULL ==="
(
    cd "$ROOT/src"
    "$PYTHON" kd_learn.py "${student_args[@]}"
) 2>&1 | tee "$STUDENT_RUN/student.log"
grep -q 'SOURCE_FROZEN_PARAMS: PASS' "$STUDENT_RUN/student.log"

"$PYTHON" - "$STUDENT_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path
folder = Path(sys.argv[1]); reports = list(folder.rglob('result.json'))
models = list(folder.rglob('student-*.pth'))
assert len(reports) == len(models) == 1
r = json.loads(reports[0].read_text())
assert r['dataset'] == 'MKG-W'
assert r['max_train_examples'] == 0
assert r['eval_queries'] == -1
assert r['git_commit'] == sys.argv[2]
assert r['policy_optimizer'] == 'source_frozen'
assert r['best_epoch'] > 0
print('STUDENT FULL: PASS')
print('Best epoch:', r['best_epoch'], 'Total epochs:', r['epoch'])
print('MRR:', r['final_result']['MRR'])
print('Hits@1/3/10:', r['final_result']['hits@[1,3,10]'])
print('Student checkpoint:', models[0])
PY
echo 'MKG-W Exp01 FULL PIPELINE: PASS'
