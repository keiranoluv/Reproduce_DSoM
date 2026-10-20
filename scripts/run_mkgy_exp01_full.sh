#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
EXP_ID="T1_MKGY_Exp01"
TEACHER_NAME="${EXP_ID}_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022"
STUDENT_NAME="${EXP_ID}_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"

cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: Commit tracked changes before training." >&2
    exit 1
fi
for source in src/learn.py src/kd_learn.py scripts/run_mkgy_exp01_full.sh scripts/smoke_test_mkgy_student.sh; do
    git ls-files --error-unmatch "$source" >/dev/null 2>&1 || {
        echo "ERROR: File is not committed: $source" >&2
        exit 1
    }
done
for feature in data/MKG-Y/img_vec.pickle data/MKG-Y/dscp_vec.pickle; do
    [[ -f "$feature" ]] || { echo "Missing: $feature" >&2; exit 1; }
done
GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
TEACHER_RUN="$ROOT/runs/teacher/MKG-Y/full/$TEACHER_NAME/$RUN_ID"
STUDENT_RUN="$ROOT/runs/student/MKG-Y/full/$STUDENT_NAME/$RUN_ID"
mkdir -p "$TEACHER_RUN/checkpoints" "$TEACHER_RUN/wandb"
mkdir -p "$STUDENT_RUN/checkpoints" "$STUDENT_RUN/wandb"
printf '%s\n' "$GIT_COMMIT" > "$TEACHER_RUN/git_commit.txt"
printf '%s\n' "$GIT_COMMIT" > "$STUDENT_RUN/git_commit.txt"

export DSOM_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${RUN_ID}"

echo "Git commit: $GIT_COMMIT"
echo "W&B group: $WANDB_RUN_GROUP"

teacher_args=(
    --dataset MKG-Y --model ComplExMDR
    --modality_split True --fusion_img True --fusion_dscp True
    --img_info "$ROOT/data/MKG-Y/img_vec.pickle"
    --dscp_info "$ROOT/data/MKG-Y/dscp_vec.pickle"
    --rank 2000 --batch_size 1000
    --optimizer Adagrad --learning_rate 0.1
    --regularizer N3 --reg 0.01 --scheduler_mode min
    --max_epochs 200 --valid 3 --early_stopping 10
    --ckpt_dir "$TEACHER_RUN/checkpoints"
    --note "$TEACHER_NAME"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)

export WANDB_NAME="$TEACHER_NAME"
export WANDB_DIR="$TEACHER_RUN/wandb"
export DSOM_RUN_DIR="$TEACHER_RUN"
printf '%q ' "$PYTHON" learn.py "${teacher_args[@]}" > "$TEACHER_RUN/command.sh"
echo "Starting Teacher: $TEACHER_RUN"
(
    cd "$ROOT/src"
    "$PYTHON" learn.py "${teacher_args[@]}"
) 2>&1 | tee "$TEACHER_RUN/teacher.log"

mapfile -t checkpoints < <(
    find "$TEACHER_RUN/checkpoints" -type f -name 'm-*.pth' -print
)
if [[ ${#checkpoints[@]} -ne 1 ]]; then
    echo "ERROR: Expected exactly one Teacher checkpoint." >&2
    exit 1
fi
TEACHER_CKPT="${checkpoints[0]}"

"$PYTHON" - "$TEACHER_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

files = list(Path(sys.argv[1]).rglob("result.json"))
assert len(files) == 1
result = json.loads(files[0].read_text())
assert result["dataset"] == "MKG-Y"
assert result["best_epoch"] > 0
assert result["git_commit"] == sys.argv[2]
print("TEACHER FULL: PASS", result["best_epoch"])
PY
printf '%s\n' "$TEACHER_CKPT" > "$STUDENT_RUN/teacher_checkpoint.txt"

student_args=(
    --dataset MKG-Y --model ComplExMDR
    --teacher_checkpoint "$TEACHER_CKPT"
    --img_info "$ROOT/data/MKG-Y/img_vec.pickle"
    --dscp_info "$ROOT/data/MKG-Y/dscp_vec.pickle"
    --rank 2000 --batch_size 1000
    --optimizer Adagrad --learning_rate 0.1
    --regularizer N3 --reg 0.01 --scheduler_mode min
    --kd_loss_type kdrl --teacher_strategy rc
    --policy_optimizer source_frozen
    --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
    --rl_hs 1024 --rl_reward 1 --rl_penalize 10
    --policy_temp 1 --rc_reward_mode binary
    --max_epochs 200 --valid 3 --early_stopping 10
    --ckpt_dir "$STUDENT_RUN/checkpoints"
    --note "$STUDENT_NAME"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)

export WANDB_NAME="$STUDENT_NAME"
export WANDB_DIR="$STUDENT_RUN/wandb"
export DSOM_RUN_DIR="$STUDENT_RUN"
printf '%q ' "$PYTHON" kd_learn.py "${student_args[@]}" > "$STUDENT_RUN/command.sh"
echo "Starting Student: $STUDENT_RUN"
(
    cd "$ROOT/src"
    "$PYTHON" kd_learn.py "${student_args[@]}"
) 2>&1 | tee "$STUDENT_RUN/student.log"

"$PYTHON" - "$STUDENT_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
files = list(folder.rglob("result.json"))
models = list(folder.rglob("student-*.pth"))
assert len(files) == 1 and len(models) == 1
result = json.loads(files[0].read_text())
assert result["dataset"] == "MKG-Y"
assert result["best_epoch"] > 0
assert result["policy_optimizer"] == "source_frozen"
assert result["git_commit"] == sys.argv[2]
test = result["final_result"]
print("STUDENT FULL: PASS")
print("Best epoch:", result["best_epoch"])
print("MRR:", test["MRR"])
print("Hits@1/3/10:", test["hits@[1,3,10]"])
print("Student checkpoint:", models[0])
PY

echo "MKG-Y Exp01 FULL PIPELINE: PASS"
