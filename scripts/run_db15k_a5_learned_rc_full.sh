#!/usr/bin/env bash
set -euo pipefail

# DB15K A5: learned RC (separate Adam LR 1e-5) + NDKD.
# Student Adagrad LR 0.1; reuse the frozen Teacher from Exp01.
# This is an experimental learned-policy ablation, not released-code behavior.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/dsom/bin/python"
DATASET="MMKB-DB15K"
EXP_ID="T1_DB15K_A5_LearnedRC_Adam1e-5"
NAME="${EXP_ID}_Student_Adagrad0p1_NDKD_KL2_T4_Seed2022"

[[ -x "$PYTHON" ]] || { echo "ERROR: Missing Python: $PYTHON" >&2; exit 1; }
for f in \
  "data/$DATASET/train.pickle" "data/$DATASET/valid.pickle" \
  "data/$DATASET/test.pickle" "data/$DATASET/to_skip.pickle" \
  "data/$DATASET/img_vec.pickle" "data/$DATASET/dscp_vec.pickle"; do
  [[ -s "$f" ]] || { echo "ERROR: Missing or empty: $f" >&2; exit 1; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: Commit tracked changes before the full experiment." >&2
  exit 1
fi
git ls-files --error-unmatch scripts/run_db15k_a5_learned_rc_full.sh >/dev/null || {
  echo "ERROR: Commit scripts/run_db15k_a5_learned_rc_full.sh first." >&2
  exit 1
}

TEACHER_CKPT_DIR="$ROOT/runs/teacher/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022/20261021_151535_2887785/checkpoints"
mapfile -t teacher_files < <(find "$TEACHER_CKPT_DIR" -type f -name 'm-*.pth' -print | sort)
[[ ${#teacher_files[@]} -eq 1 ]] || {
  echo "ERROR: Expected one teacher checkpoint; found ${#teacher_files[@]}." >&2
  exit 1
}
TEACHER_CKPT="${teacher_files[0]}"

# Ensure that A5 uses precisely the same features as source-frozen Exp01.
BASELINE="runs/student/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022/20261021_151535_2887785"
[[ -s "$BASELINE/feature_sha256.txt" ]] || {
  echo "ERROR: Exp01 feature_sha256.txt is missing: $BASELINE" >&2
  exit 1
}
sha256sum --check "$BASELINE/feature_sha256.txt"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
RUN="$ROOT/runs/student/$DATASET/learned_rc_adam_full/$RUN_ID"
mkdir -p "$RUN/checkpoints" "$RUN/wandb"
GIT_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$GIT_COMMIT" > "$RUN/git_commit.txt"
printf '%s\n' "$TEACHER_CKPT" > "$RUN/teacher_checkpoint.txt"
sha256sum "data/$DATASET/img_vec.pickle" "data/$DATASET/dscp_vec.pickle" > "$RUN/feature_sha256.txt"

export DSOM_WANDB=1
export DSOM_RUN_DIR="$RUN"
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${RUN_ID}"
export WANDB_NAME="$NAME"
export WANDB_DIR="$RUN/wandb"

ARGS=(
  --dataset "$DATASET" --model ComplExMDR
  --teacher_checkpoint "$TEACHER_CKPT"
  --img_info "$ROOT/data/$DATASET/img_vec.pickle"
  --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
  --rank 2000 --batch_size 1000
  --optimizer Adagrad --learning_rate 0.1
  --regularizer N3 --reg 0.01 --scheduler_mode min
  --kd_loss_type kdrl --teacher_strategy rc
  --policy_optimizer Adam --policy_lr 0.00001
  --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
  --dkd_alpha 1 --dkd_beta 1
  --rl_hs 1024 --rl_reward 1 --rl_penalize 10
  --rl_weight 1 --policy_temp 1 --rc_reward_mode binary
  --max_epochs 200 --valid 3 --early_stopping 10
  --max_train_examples 0 --eval_queries -1
  --ckpt_dir "$RUN/checkpoints" --note "$NAME"
  --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
)

printf '%q ' "$PYTHON" kd_learn.py "${ARGS[@]}" > "$RUN/command.sh"
printf '\n' >> "$RUN/command.sh"
echo "=== DB15K A5 LEARNED RC (Adam 1e-5) ==="
echo "Run: $RUN"
echo "Teacher: $TEACHER_CKPT"
echo "Git commit: $GIT_COMMIT"
echo "WARNING: DB15K validation contains known overlap with train/test."
echo "WARNING: This is a learned RC ablation, not the released-code optimizer."

(
  cd "$ROOT/src"
  "$PYTHON" -u kd_learn.py "${ARGS[@]}"
) 2>&1 | tee "$RUN/train.log"

"$PYTHON" - "$RUN" "$GIT_COMMIT" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected_commit = sys.argv[2]
reports = list((run / "checkpoints").rglob("result.json"))
checkpoints = list((run / "checkpoints").rglob("student-*.pth"))
assert len(reports) == 1, reports
assert len(checkpoints) == 1, checkpoints
r = json.loads(reports[0].read_text())
assert r["dataset"] == "MMKB-DB15K"
assert r["kd_loss_type"] == "kdrl"
assert r["teacher_strategy"] == "rc"
assert r["policy_optimizer"] == "Adam"
assert abs(r["policy_lr"] - 1e-5) < 1e-12
assert r["max_train_examples"] == 0
assert r["eval_queries"] == -1
assert r["git_commit"] == expected_commit
assert r["best_epoch"] > 0

updates = [int(v) for v in re.findall(
    r"Batches with updated policy weights:\s*(\d+)",
    (run / "train.log").read_text(errors="replace"),
)]
assert len(updates) == r["epoch"], (len(updates), r["epoch"])
assert any(x > 0 for x in updates), "RC weights never updated"

m = r["final_result"]
print("\n=== DB15K A5 LEARNED RC: PASS ===")
print("Best epoch:", r["best_epoch"], "Total epochs:", r["epoch"])
print("Best validation MRR:", r["best_mrr"])
print("Best-checkpoint test MR:", m["MR"])
print("Best-checkpoint test MRR:", m["MRR"])
print("Best-checkpoint test Hits@1/3/10:", m["hits@[1,3,10]"])
print("Policy updated batches by epoch (first 10):", updates[:10])
print("Epochs with at least one policy update:", sum(x > 0 for x in updates))
print("Student checkpoint:", checkpoints[0])
print("Result:", reports[0])
PY
