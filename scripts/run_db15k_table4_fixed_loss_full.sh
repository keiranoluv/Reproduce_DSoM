#!/usr/bin/env bash
set -euo pipefail

# A7: fixed structural_description + ordinary KL KD
# A8: fixed structural_description + target/non-target DKD
# A3 existing fixed structural_description + neighbor-aware NDKD is the reference.
# Usage: bash scripts/run_db15k_table4_fixed_loss_full.sh --smoke
#        bash scripts/run_db15k_table4_fixed_loss_full.sh --full
# Optional: --parallel 1|2 (default 1 for smoke, 2 for full)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
DATASET=MMKB-DB15K
EXP_ID=T4_DB15K_A3_FixedTeacher_LossAblation
TEACHER_RUN="$ROOT/runs/teacher/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022/20261021_151535_2887785"
A3_RESULT="$ROOT/runs/student/$DATASET/fixed_teacher_ndkd_full/20261021_163151_2915928/structural_description/checkpoints/0/result.json"
MODE="${1:---smoke}"
[[ "$MODE" == --smoke || "$MODE" == --full ]] || {
  echo "Usage: $0 --smoke|--full [--parallel 1|2]" >&2; exit 2;
}
shift || true
PARALLEL=1
[[ "$MODE" == --full ]] && PARALLEL=2
if [[ "${1:-}" == --parallel ]]; then
  PARALLEL="${2:-}"
  shift 2
fi
[[ $# == 0 && ( "$PARALLEL" == 1 || "$PARALLEL" == 2 ) ]] || {
  echo "Usage: $0 --smoke|--full [--parallel 1|2]" >&2; exit 2;
}

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$A3_RESULT" ]] || { echo "Missing existing A3 reference: $A3_RESULT" >&2; exit 1; }
for path in src/kd_learn.py src/kd_optimizers.py src/kd.py \
    scripts/apply_db15k_table4_fixed_losses.py scripts/run_db15k_table4_fixed_loss_full.sh; do
  [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
"$PYTHON" scripts/apply_db15k_table4_fixed_losses.py --repo "$ROOT" --check
"$PYTHON" -m py_compile src/kd_learn.py src/kd_optimizers.py
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Commit modified tracked source files before running, for provenance." >&2
  exit 1
fi
for path in src/kd_learn.py src/kd_optimizers.py src/kd.py \
    scripts/apply_db15k_table4_fixed_losses.py scripts/run_db15k_table4_fixed_loss_full.sh; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Untracked script/source; commit first: $path" >&2; exit 1;
  }
done
mapfile -t TEACHERS < <(find "$TEACHER_RUN/checkpoints" -type f -name 'm-*.pth' -print | sort)
[[ ${#TEACHERS[@]} -eq 1 ]] || {
  echo "Expected one teacher checkpoint; found ${#TEACHERS[@]}" >&2; exit 1;
}
TEACHER="${TEACHERS[0]}"
for path in "$ROOT/data/$DATASET/img_vec.pickle" "$ROOT/data/$DATASET/dscp_vec.pickle"; do
  [[ -s "$path" ]] || { echo "Missing feature: $path" >&2; exit 1; }
done
if [[ -f "$TEACHER_RUN/feature_sha256.txt" ]]; then
  sha256sum -c "$TEACHER_RUN/feature_sha256.txt"
else
  echo "WARNING: Teacher feature_sha256.txt missing." >&2
fi
"$PYTHON" -c 'import wandb; print("W&B version:", wandb.__version__)'

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
BASE="$ROOT/runs/student/$DATASET/table4_fixed_loss_${MODE#--}/$RUN_ID"
mkdir -p "$BASE"
printf '%s\n' "$GIT_COMMIT" > "$BASE/git_commit.txt"
printf '%s\n' "$TEACHER" > "$BASE/teacher_checkpoint.txt"
sha256sum "$TEACHER" "$ROOT/data/$DATASET/img_vec.pickle" \
  "$ROOT/data/$DATASET/dscp_vec.pickle" > "$BASE/input_sha256.txt"

export DSOM_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${MODE#--}_${RUN_ID}"
echo "=== TABLE4 DB15K $MODE | parallel=$PARALLEL ==="
echo "Git commit: $GIT_COMMIT"
echo "Teacher: $TEACHER"
echo "A3 baseline: $A3_RESULT"
echo "Run directory: $BASE"
echo "WARNING: DB15K validation leakage; interpret best-epoch selection carefully."

run_one() {
  local LOSS="$1"
  local LABEL
  [[ "$LOSS" == fixed_kd ]] && LABEL=A7_KL || LABEL=A8_DKD
  local RUN="$BASE/$LABEL" NOTE="${EXP_ID}_${LABEL}_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"
  mkdir -p "$RUN/checkpoints" "$RUN/wandb"
  local MAX_EPOCHS=200 VALID=3 EARLY_STOPPING=10 MAX_EXAMPLES=0 EVAL_QUERIES=-1
  if [[ "$MODE" == --smoke ]]; then
    MAX_EPOCHS=1; VALID=1; EARLY_STOPPING=2; MAX_EXAMPLES=3000; EVAL_QUERIES=64
  fi
  local -a ARGS=(
    --dataset "$DATASET" --model ComplExMDR
    --teacher_checkpoint "$TEACHER"
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --kd_loss_type "$LOSS"
    --teacher_strategy structural_description
    --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
    --dkd_alpha 1 --dkd_beta 1
    --max_epochs "$MAX_EPOCHS" --valid "$VALID" --early_stopping "$EARLY_STOPPING"
    --max_train_examples "$MAX_EXAMPLES" --eval_queries "$EVAL_QUERIES"
    --ckpt_dir "$RUN/checkpoints" --note "$NOTE"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
  )
  printf '%q ' "$PYTHON" kd_learn.py "${ARGS[@]}" > "$RUN/command.sh"
  printf '\n' >> "$RUN/command.sh"
  printf '%s\n' "$GIT_COMMIT" > "$RUN/git_commit.txt"
  printf '%s\n' "$TEACHER" > "$RUN/teacher_checkpoint.txt"
  export WANDB_NAME="$NOTE"
  export WANDB_DIR="$RUN/wandb"
  echo "START $LABEL ($LOSS)"
  if ! (cd "$ROOT/src" && "$PYTHON" -u kd_learn.py "${ARGS[@]}"); then
    echo "FAILED student training for $LABEL" >&2
    return 1
  fi
  "$PYTHON" - "$RUN/checkpoints" "$LOSS" "$TEACHER" "$GIT_COMMIT" "$MODE" <<'PY'
import json, sys
from pathlib import Path
folder, loss, teacher, sha, mode = Path(sys.argv[1]), *sys.argv[2:]
reports = list(folder.rglob('result.json'))
assert len(reports) == 1, reports
r = json.loads(reports[0].read_text())
assert r['kd_loss_type'] == loss, r['kd_loss_type']
assert r['teacher_strategy'] == 'structural_description'
assert r['teacher_checkpoint'] == teacher
assert r['git_commit'] == sha
assert r['dataset'] == 'MMKB-DB15K'
assert r['best_epoch'] > 0
if mode == '--full':
    assert r['max_train_examples'] == 0 and r['eval_queries'] == -1
    assert r['rank'] == 2000 and r['batch_size'] == 1000
    assert r['kd_temp'] == 4 and r['kd_kl_weight'] == 2
else:
    assert r['max_train_examples'] == 3000 and r['eval_queries'] == 64
checkpoints = list(folder.rglob('student-*.pth'))
assert len(checkpoints) == 1, checkpoints
t = r['final_result']
assert 0 <= t['MRR'] <= 1 and len(t['hits@[1,3,10]']) == 3
print('FULL PASS' if mode == '--full' else 'SMOKE PASS', loss,
      'best epoch', r['best_epoch'], 'test MRR', format(t['MRR'], '.6f'))
PY
}

if [[ "$PARALLEL" == 1 ]]; then
  for LOSS in fixed_kd fixed_dkd; do
    LABEL=A7_KL; [[ "$LOSS" == fixed_dkd ]] && LABEL=A8_DKD
    run_one "$LOSS" > "$BASE/$LABEL.log" 2>&1 || {
      echo "FAILED $LABEL; inspect $BASE/$LABEL.log" >&2
      tail -n 40 "$BASE/$LABEL.log" >&2 || true
      exit 1
    }
    echo "PASS $LABEL | log: $BASE/$LABEL.log"
  done
else
  run_one fixed_kd > "$BASE/A7_KL.log" 2>&1 &
  PID_KD=$!
  run_one fixed_dkd > "$BASE/A8_DKD.log" 2>&1 &
  PID_DKD=$!
  STATUS_KD=0; STATUS_DKD=0
  wait "$PID_KD" || STATUS_KD=$?
  wait "$PID_DKD" || STATUS_DKD=$?
  if [[ $STATUS_KD -ne 0 || $STATUS_DKD -ne 0 ]]; then
    echo "One or more jobs failed: KD=$STATUS_KD, DKD=$STATUS_DKD" >&2
    tail -n 35 "$BASE/A7_KL.log" >&2 || true
    tail -n 35 "$BASE/A8_DKD.log" >&2 || true
    exit 1
  fi
  echo "PASS A7 & A8 in parallel"
fi

"$PYTHON" - "$BASE" "$A3_RESULT" <<'PY'
import json, sys
from pathlib import Path
base, a3 = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
candidates = [
    ('A7 KL', next((base / 'A7_KL/checkpoints').rglob('result.json'))),
    ('A8 DKD', next((base / 'A8_DKD/checkpoints').rglob('result.json'))),
]
if 'table4_fixed_loss_full' in str(base):
    candidates.insert(0, ('A3 NDKD', a3))
for label, file in candidates:
    obj = json.loads(file.read_text())
    t = obj['final_result']
    rows.append({'experiment': label, 'loss': obj['kd_loss_type'],
                 'best_epoch': obj['best_epoch'], 'valid_mrr': obj['best_mrr'],
                 'mr': t['MR'], 'mrr': t['MRR'],
                 'h1': t['hits@[1,3,10]'][0], 'h3': t['hits@[1,3,10]'][1],
                 'h10': t['hits@[1,3,10]'][2], 'result_file': str(file)})
print('\n=== DB15K FIXED STRUCTURAL+DESCRIPTION LOSS ABLATION ===')
print(f"{'name':10s} {'loss':12s} {'epoch':>5s} {'valid MRR':>10s} {'test MRR':>10s} {'H@1':>10s} {'H@3':>10s} {'H@10':>10s}")
for r in rows:
    print(f"{r['experiment']:10s} {r['loss']:12s} {r['best_epoch']:5d} {r['valid_mrr']:10.6f} {r['mrr']:10.6f} {r['h1']:10.6f} {r['h3']:10.6f} {r['h10']:10.6f}")
(base / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Summary:', base / 'summary.json')
PY

echo "TABLE4 FIXED LOSS ${MODE#--}: PASS"
