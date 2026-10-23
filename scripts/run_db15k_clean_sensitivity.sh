#!/usr/bin/env bash
# Isolated teacher -> source-frozen RC student, matching original Exp01 settings.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/dsom/bin/python"
MODE="${1:---help}"
[[ "$MODE" == --smoke || "$MODE" == --full ]] || {
    echo "Usage: $0 --smoke|--full" >&2
    exit 2
}
cd "$ROOT"
"$PY" scripts/prepare_db15k_clean_split.py --verify-only

DATASET=MMKB-DB15K-CleanV1
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
RUN="$ROOT/runs/clean_sensitivity/$DATASET/${MODE#--}/$RUN_ID"
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir -p "$RUN/teacher/checkpoints" "$RUN/student/checkpoints"
printf '%s\n' "$RUN" > "$RUN/run_path.txt"
cp "data/$DATASET/split_manifest.json" "$RUN/split_manifest.json"
git rev-parse HEAD > "$RUN/git_commit.txt"
git diff --binary HEAD > "$RUN/tracked_changes.patch"
# Snapshot new/uncommitted scripts and source too; HEAD alone does not describe this run.
tar --exclude='__pycache__' -czf "$RUN/source_snapshot.tar.gz" src scripts ruff.toml \
    2> "$RUN/source_snapshot.stderr"
sha256sum "$RUN/source_snapshot.tar.gz" > "$RUN/source_snapshot.sha256"
GIT_COMMIT="$(git rev-parse HEAD)"
export DSOM_WANDB="${DSOM_WANDB:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="DB15K_CleanV1_${RUN_ID}"

COMMON=(
    --dataset "$DATASET" --model ComplExMDR
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --git_commit "$GIT_COMMIT"
    --experiment_id DB15K_CleanV1_Sensitivity
)
if [[ "$MODE" == --smoke ]]; then
    COMMON+=(--max_epochs 1 --valid 1 --early_stopping 10 --max_train_examples 1000 --eval_queries 64)
else
    COMMON+=(--max_epochs 200 --valid 3 --early_stopping 10)
fi

echo "RUN_DIRECTORY=$RUN"
TEACHER_ARGS=(
    "${COMMON[@]}" --modality_split True --fusion_img True --fusion_dscp True
    --ckpt_dir "$RUN/teacher/checkpoints" --note "DB15K_CleanV1_Teacher_${MODE#--}"
)
export DSOM_RUN_DIR="$RUN/teacher"
export WANDB_DIR="$RUN/teacher/wandb"
export WANDB_NAME="DB15K_CleanV1_Teacher_${MODE#--}_${RUN_ID}"
printf '%q ' "$PY" learn.py "${TEACHER_ARGS[@]}" > "$RUN/teacher/command.sh"
printf '\n' >> "$RUN/teacher/command.sh"
(cd "$ROOT/src"; "$PY" -u learn.py "${TEACHER_ARGS[@]}") 2>&1 | tee "$RUN/teacher/train.log"

mapfile -t TEACHERS < <(find "$RUN/teacher/checkpoints" -type f -name 'm-*.pth' -print)
[[ ${#TEACHERS[@]} -eq 1 ]] || { echo 'Missing or ambiguous new teacher' >&2; exit 1; }
STUDENT_ARGS=(
    "${COMMON[@]}" --teacher_checkpoint "${TEACHERS[0]}"
    --kd_loss_type kdrl --teacher_strategy rc --policy_optimizer source_frozen
    --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
    --dkd_alpha 1 --dkd_beta 1 --rl_hs 1024 --rl_reward 1 --rl_penalize 10
    --policy_temp 1 --rc_reward_mode binary
    --ckpt_dir "$RUN/student/checkpoints" --note "DB15K_CleanV1_Student_${MODE#--}"
)
export DSOM_RUN_DIR="$RUN/student"
export WANDB_DIR="$RUN/student/wandb"
export WANDB_NAME="DB15K_CleanV1_Student_${MODE#--}_${RUN_ID}"
printf '%q ' "$PY" kd_learn.py "${STUDENT_ARGS[@]}" > "$RUN/student/command.sh"
printf '\n' >> "$RUN/student/command.sh"
(cd "$ROOT/src"; "$PY" -u kd_learn.py "${STUDENT_ARGS[@]}") 2>&1 | tee "$RUN/student/train.log"

"$PY" - "$RUN" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

run, mode = Path(sys.argv[1]), sys.argv[2]
results = {}
for stage in ("teacher", "student"):
    paths = list((run / stage / "checkpoints").rglob("result.json"))
    assert len(paths) == 1, paths
    result = json.loads(paths[0].read_text())
    assert result["dataset"] == "MMKB-DB15K-CleanV1"
    assert result["best_epoch"] > 0
    assert result["max_train_examples"] == (1000 if mode == "--smoke" else 0)
    assert result["eval_queries"] == (64 if mode == "--smoke" else (0 if stage == "teacher" else -1))
    results[stage] = result
    results[stage + "_result_path"] = str(paths[0])
assert results["student"]["policy_optimizer"] == "source_frozen"
assert Path(results["student"]["teacher_checkpoint"]).is_relative_to(run / "teacher")
assert "SOURCE_FROZEN_PARAMS: PASS" in (run / "student/train.log").read_text()
baseline_dir = run.parents[3] / "student/MMKB-DB15K/full"
baseline_paths = list(baseline_dir.rglob("result.json"))
assert len(baseline_paths) == 1, baseline_paths
baseline = json.loads(baseline_paths[0].read_text())
summary = {
    "status": "PASS", "mode": mode, "run": str(run),
    "teacher_result": results["teacher_result_path"],
    "student_result": results["student_result_path"],
    "teacher_best_epoch": results["teacher"]["best_epoch"],
    "teacher_best_valid_mrr": results["teacher"]["best_mrr"],
    "teacher_metrics_are_oracle_style": True,
    "student_best_epoch": results["student"]["best_epoch"],
    "student_best_valid_mrr": results["student"]["best_mrr"],
    "student_test": results["student"]["final_result"],
    "baseline_result": str(baseline_paths[0]),
    "baseline_student_test": baseline["final_result"],
    "test_mrr_difference": results["student"]["final_result"]["MRR"] - baseline["final_result"]["MRR"],
    "interpretation": "Smoke metrics are not comparable" if mode == "--smoke" else "Sensitivity analysis changes validation, training size, test-reversal exclusion, and evaluation filters; not a causal leakage effect estimate",
}
(run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
"$PY" scripts/prepare_db15k_clean_split.py --verify-only
echo "CLEAN SENSITIVITY PIPELINE: PASS | $RUN"
