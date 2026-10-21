#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
DATASET="MMKB-DB15K"
FEATURES="$ROOT/data/$DATASET/h5_pilot"
BASE="$ROOT/runs/db15k_h5_pilot/smoke"

cd "$ROOT"

test -f "$FEATURES/img_vec.pickle"
test -f "$FEATURES/dscp_vec.pickle"
test -f "$ROOT/data/$DATASET/train.pickle"
test -f "$ROOT/data/$DATASET/valid.pickle"
test -f "$ROOT/data/$DATASET/test.pickle"

mkdir -p "$BASE"
RUN="$(mktemp -d "$BASE/run_XXXXXX")"

mkdir -p \
  "$RUN/teacher/checkpoints" \
  "$RUN/student/checkpoints"

export DSOM_WANDB=0

COMMON=(
  --dataset "$DATASET"
  --model ComplExMDR
  --img_info "$FEATURES/img_vec.pickle"
  --dscp_info "$FEATURES/dscp_vec.pickle"
  --rank 2000
  --batch_size 1000
  --optimizer Adagrad
  --learning_rate 0.1
  --regularizer N3
  --reg 0.01
  --scheduler_mode min
  --max_epochs 1
  --valid 1
  --max_train_examples 1000
  --eval_queries 64
  --experiment_id T1_DB15K_H5_Pilot
)

echo "=== TEACHER SMOKE ==="

(
  cd "$ROOT/src"

  "$PYTHON" learn.py \
    "${COMMON[@]}" \
    --modality_split True \
    --fusion_img True \
    --fusion_dscp True \
    --ckpt_dir "$RUN/teacher/checkpoints" \
    --note DB15K_H5_Teacher_Smoke

) 2>&1 | tee "$RUN/teacher.log"

mapfile -t TEACHER_FILES < <(
  find "$RUN/teacher/checkpoints" \
    -type f -name 'm-*.pth'
)

if [[ ${#TEACHER_FILES[@]} -ne 1 ]]; then
  echo "Expected exactly one Teacher checkpoint" >&2
  exit 1
fi

TEACHER_CKPT="${TEACHER_FILES[0]}"

echo "Teacher checkpoint: $TEACHER_CKPT"
echo "=== STUDENT SMOKE ==="

(
  cd "$ROOT/src"

  "$PYTHON" kd_learn.py \
    "${COMMON[@]}" \
    --teacher_checkpoint "$TEACHER_CKPT" \
    --kd_loss_type kdrl \
    --teacher_strategy rc \
    --policy_optimizer source_frozen \
    --kd_temp 4 \
    --kd_ce_weight 1 \
    --kd_kl_weight 2 \
    --rl_hs 1024 \
    --rl_reward 1 \
    --rl_penalize 10 \
    --policy_temp 1 \
    --rc_reward_mode binary \
    --ckpt_dir "$RUN/student/checkpoints" \
    --note DB15K_H5_Student_Smoke

) 2>&1 | tee "$RUN/student.log"

"$PYTHON" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])

for role in ("teacher", "student"):
    results = list(
        (run / role / "checkpoints").rglob("result.json")
    )
    assert len(results) == 1, (role, results)

    result = json.loads(results[0].read_text())

    assert result["dataset"] == "MMKB-DB15K"
    assert result["best_epoch"] == 1
    assert result["max_train_examples"] == 1000
    assert result["eval_queries"] == 64

    if role == "student":
        assert result["policy_optimizer"] == "source_frozen"
        checkpoints = list(
            (run / role / "checkpoints").rglob("student-*.pth")
        )
        assert len(checkpoints) == 1
    else:
        checkpoints = list(
            (run / role / "checkpoints").rglob("m-*.pth")
        )
        assert len(checkpoints) == 1

    print(f"{role.upper()} SMOKE: PASS")
    print("Checkpoint:", checkpoints[0])

print("DB15K H5 PILOT SMOKE: PASS")
print("Run directory:", run)
PY
