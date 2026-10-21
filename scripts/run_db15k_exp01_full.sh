#!/usr/bin/env bash
set -euo pipefail

# Table 1, DB15K Exp01: historical MMKB images + BERT/ViT features.
# Mirrors run_mkgw_exp01_full.sh. This is a source_frozen RC run,
# NOT a learned-policy REINFORCE replication.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
DATASET="MMKB-DB15K"
EXP_ID="T1_DB15K_Exp01_BERT_ViT"
TEACHER_NAME="${EXP_ID}_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022"
STUDENT_NAME="${EXP_ID}_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "ERROR: Python not executable: $PYTHON" >&2; exit 1; }
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo 'ERROR: Commit tracked code changes before full training.' >&2
    exit 1
fi
for source in \
    src/learn.py src/kd_learn.py src/models.py src/kd.py \
    scripts/build_db15k_bert_features.py \
    scripts/build_db15k_vit_features.py \
    scripts/download_db15k_mmkb_images.py \
    scripts/smoke_test_db15k_bert_vit.sh \
    scripts/run_db15k_exp01_full.sh; do
    git ls-files --error-unmatch "$source" >/dev/null 2>&1 || {
        echo "ERROR: Not tracked by Git: $source (commit the script first)" >&2
        exit 1
    }
done
for file in \
    "data/$DATASET/ent_id" \
    "data/$DATASET/train.pickle" \
    "data/$DATASET/valid.pickle" \
    "data/$DATASET/test.pickle" \
    "data/$DATASET/dscp_vec.pickle" \
    "data/$DATASET/img_vec.pickle"; do
    [[ -s "$file" ]] || { echo "ERROR: Missing or empty: $file" >&2; exit 1; }
done

"$PYTHON" - "$ROOT/data/$DATASET" <<'PY'
import json
import pickle
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
for name, expected in [('dscp_vec.pickle', (12842, 768)), ('img_vec.pickle', (12842, 1000))]:
    with (root / name).open('rb') as handle:
        arr = np.asarray(pickle.load(handle))
    assert arr.shape == expected, (name, arr.shape, expected)
    assert np.isfinite(arr).all(), f'Non-finite values in {name}'
    print(f'FEATURES PASS: {name} {arr.shape} {arr.dtype}')
print('Full run uses data/MMKB-DB15K/ features (not h5_pilot)')
PY
"$PYTHON" -c 'import wandb; print("W&B version:", wandb.__version__)'

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
TEACHER_RUN="$ROOT/runs/teacher/$DATASET/full/$TEACHER_NAME/$RUN_ID"
STUDENT_RUN="$ROOT/runs/student/$DATASET/full/$STUDENT_NAME/$RUN_ID"
mkdir -p "$TEACHER_RUN/checkpoints" "$TEACHER_RUN/wandb"
mkdir -p "$STUDENT_RUN/checkpoints" "$STUDENT_RUN/wandb"
printf '%s\n' "$GIT_COMMIT" > "$TEACHER_RUN/git_commit.txt"
printf '%s\n' "$GIT_COMMIT" > "$STUDENT_RUN/git_commit.txt"

# Provenance for the feature files actually used in training.
sha256sum "data/$DATASET/dscp_vec.pickle" "data/$DATASET/img_vec.pickle" \
    > "$TEACHER_RUN/feature_sha256.txt"
cp "$TEACHER_RUN/feature_sha256.txt" "$STUDENT_RUN/feature_sha256.txt"
for manifest in data/$DATASET/*manifest*.json; do
    [[ -f "$manifest" ]] || continue
    cp "$manifest" "$TEACHER_RUN/$(basename "$manifest")"
done

export DSOM_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${RUN_ID}"
echo "=== DB15K Exp01 source_frozen | commit=$GIT_COMMIT | group=$WANDB_RUN_GROUP ==="
echo 'WARNING: DB15K validation has known overlap with train/test; the original split is unchanged.'
echo 'WARNING: Raw images were recovered from historic MMKB URLs and are not verified as DSoM-original images.'

teacher_args=(
    --dataset "$DATASET" --model ComplExMDR
    --modality_split True --fusion_img True --fusion_dscp True
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
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
printf '\n' >> "$TEACHER_RUN/command.sh"
echo "=== DB15K TEACHER FULL: $TEACHER_RUN ==="
(
    cd "$ROOT/src"
    "$PYTHON" learn.py "${teacher_args[@]}"
) 2>&1 | tee "$TEACHER_RUN/teacher.log"

mapfile -t teacher_ckpts < <(find "$TEACHER_RUN/checkpoints" -type f -name 'm-*.pth' -print)
[[ ${#teacher_ckpts[@]} -eq 1 ]] || {
    echo "ERROR: Expected exactly one Teacher checkpoint; found ${#teacher_ckpts[@]}" >&2
    exit 1
}
TEACHER_CKPT="${teacher_ckpts[0]}"
"$PYTHON" - "$TEACHER_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json, sys
from pathlib import Path
reports = list(Path(sys.argv[1]).rglob('result.json'))
assert len(reports) == 1, reports
r = json.loads(reports[0].read_text())
assert r['dataset'] == 'MMKB-DB15K'
assert r['max_train_examples'] == 0
assert r['eval_queries'] == 0
assert r['git_commit'] == sys.argv[2]
assert r['best_epoch'] > 0
print('TEACHER FULL: PASS | best epoch:', r['best_epoch'])
PY
printf '%s\n' "$TEACHER_CKPT" > "$STUDENT_RUN/teacher_checkpoint.txt"

student_args=(
    --dataset "$DATASET" --model ComplExMDR
    --teacher_checkpoint "$TEACHER_CKPT"
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
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
printf '\n' >> "$STUDENT_RUN/command.sh"
echo "=== DB15K STUDENT FULL: $STUDENT_RUN ==="
(
    cd "$ROOT/src"
    "$PYTHON" kd_learn.py "${student_args[@]}"
) 2>&1 | tee "$STUDENT_RUN/student.log"
grep -q 'SOURCE_FROZEN_PARAMS: PASS' "$STUDENT_RUN/student.log" || {
    echo 'ERROR: Source-frozen verification marker not found in Student log.' >&2
    exit 1
}
"$PYTHON" - "$STUDENT_RUN/checkpoints" "$GIT_COMMIT" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
reports = list(folder.rglob('result.json'))
models = list(folder.rglob('student-*.pth'))
assert len(reports) == len(models) == 1, (reports, models)
r = json.loads(reports[0].read_text())
assert r['dataset'] == 'MMKB-DB15K'
assert r['max_train_examples'] == 0
assert r['eval_queries'] == -1
assert r['git_commit'] == sys.argv[2]
assert r['policy_optimizer'] == 'source_frozen'
assert r['best_epoch'] > 0
print('STUDENT FULL: PASS')
print('Best epoch:', r['best_epoch'], 'Total epochs:', r['epoch'])
print('Best-checkpoint test MRR:', r['final_result']['MRR'])
print('Best-checkpoint test Hits@1/3/10:', r['final_result']['hits@[1,3,10]'])
print('Student checkpoint:', models[0])
PY
echo 'DB15K Exp01 BERT ViT FULL PIPELINE: PASS'
