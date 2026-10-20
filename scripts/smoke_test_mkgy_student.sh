#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
EXP_ID="T1_MKGY_Exp01"
STUDENT_NAME="${EXP_ID}_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"

cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: Commit tracked changes before running." >&2
    exit 1
fi
GIT_COMMIT="$(git rev-parse HEAD)"

TEACHER_CKPT="${1:-}"
if [[ -z "$TEACHER_CKPT" ]]; then
    TEACHER_DIR="$ROOT/runs/teacher/MKG-Y/smoke/checkpoints/0"
    mapfile -t matches < <(
        find "$TEACHER_DIR" -maxdepth 1 -type f -name 'm-*.pth' -print
    )
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "ERROR: Expected exactly one Teacher smoke checkpoint." >&2
        exit 1
    fi
    TEACHER_CKPT="${matches[0]}"
fi
TEACHER_CKPT="$(realpath "$TEACHER_CKPT")"
[[ -f "$TEACHER_CKPT" ]] || { echo "Missing: $TEACHER_CKPT" >&2; exit 1; }

BASE="$ROOT/runs/student/MKG-Y/smoke/$STUDENT_NAME"
mkdir -p "$BASE"
RUN_DIR="$(mktemp -d "$BASE/run_XXXXXX")"
mkdir -p "$RUN_DIR/checkpoints"
printf '%s\n' "$GIT_COMMIT" > "$RUN_DIR/git_commit.txt"
printf '%s\n' "$TEACHER_CKPT" > "$RUN_DIR/teacher_checkpoint.txt"

export DSOM_WANDB=0
args=(
    --dataset MKG-Y
    --model ComplExMDR
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
    --max_epochs 1 --valid 1
    --max_train_examples 1000 --eval_queries 64
    --ckpt_dir "$RUN_DIR/checkpoints"
    --note "${STUDENT_NAME}_Smoke"
    --git_commit "$GIT_COMMIT"
    --experiment_id "$EXP_ID"
)

printf '%q ' "$PYTHON" kd_learn.py "${args[@]}" > "$RUN_DIR/command.sh"

(
    cd "$ROOT/src"
    "$PYTHON" kd_learn.py "${args[@]}"
) 2>&1 | tee "$RUN_DIR/student.log"

grep -q 'SOURCE_FROZEN_PARAMS: PASS' "$RUN_DIR/student.log"
"$PYTHON" - "$RUN_DIR/checkpoints" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
results = list(folder.rglob("result.json"))
models = list(folder.rglob("student-*.pth"))
assert len(results) == 1 and len(models) == 1
result = json.loads(results[0].read_text())
assert result["dataset"] == "MKG-Y"
assert result["best_epoch"] > 0
assert result["policy_optimizer"] == "source_frozen"
assert result["git_commit"] == sys.argv[2]
print("STUDENT SMOKE: PASS")
print("Checkpoint:", models[0])
PY
