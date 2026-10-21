#!/usr/bin/env bash
set -euo pipefail

# DB15K fixed-teacher NDKD ablation. Reuses existing Exp01 Teacher.
# No teacher retraining; each Student is trained from scratch.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/dsom/bin/python"
DATASET="MMKB-DB15K"
EXP_ID="T1_DB15K_FixedTeacher_NDKD_Ablation"
TEACHER_RUN="$ROOT/runs/teacher/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022/20261021_151535_2887785"
BASELINE_STUDENT="$ROOT/runs/student/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022/20261021_151535_2887785/checkpoints/0/result.json"

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "Missing executable: $PYTHON" >&2; exit 1; }
[[ -f "$BASELINE_STUDENT" ]] || { echo "Missing source_frozen baseline: $BASELINE_STUDENT" >&2; exit 1; }
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Commit tracked changes before running the ablation." >&2
  exit 1
fi
for source in src/kd_learn.py src/kd.py src/kd_optimizers.py scripts/run_db15k_fixed_teacher_full.sh; do
  git ls-files --error-unmatch "$source" >/dev/null 2>&1 || {
    echo "File not tracked (commit first): $source" >&2; exit 1;
  }
done

mapfile -t TEACHERS < <(find "$TEACHER_RUN/checkpoints" -type f -name 'm-*.pth' -print | sort)
[[ ${#TEACHERS[@]} -eq 1 ]] || {
  echo "Expected exactly one teacher checkpoint; found ${#TEACHERS[@]}" >&2; exit 1;
}
TEACHER="${TEACHERS[0]}"
for path in "$ROOT/data/$DATASET/img_vec.pickle" "$ROOT/data/$DATASET/dscp_vec.pickle"; do
  [[ -s "$path" ]] || { echo "Missing feature: $path" >&2; exit 1; }
done
if [[ -f "$TEACHER_RUN/feature_sha256.txt" ]]; then
  sha256sum -c "$TEACHER_RUN/feature_sha256.txt"
else
  echo "WARNING: Teacher's feature_sha256.txt not found; cannot verify same feature bytes." >&2
fi
"$PYTHON" -c 'import wandb; print("W&B version:", wandb.__version__)'

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
BASE="$ROOT/runs/student/$DATASET/fixed_teacher_ndkd_full/$RUN_ID"
mkdir -p "$BASE"
printf '%s\n' "$GIT_COMMIT" > "$BASE/git_commit.txt"
printf '%s\n' "$TEACHER" > "$BASE/teacher_checkpoint.txt"
sha256sum "$TEACHER" "$ROOT/data/$DATASET/img_vec.pickle" "$ROOT/data/$DATASET/dscp_vec.pickle" > "$BASE/input_sha256.txt"

export DSOM_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${RUN_ID}"

echo "=== DB15K Fixed NDKD FULL ==="
echo "Commit: $GIT_COMMIT"
echo "Teacher: $TEACHER"
echo "Baseline: $BASELINE_STUDENT"
echo "W&B group: $WANDB_RUN_GROUP"
echo "Results: $BASE"
echo "WARNING: Original DB15K validation overlaps train/test and has duplicate triples."
echo "WARNING: Seeds match, but minibatch RNG trajectories may differ from RC because it constructs a policy MLP."

for STRATEGY in structural description structural_description average; do
  NOTE="${EXP_ID}_${STRATEGY}_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022"
  RUN="$BASE/$STRATEGY"
  mkdir -p "$RUN/checkpoints" "$RUN/wandb"
  printf '%s\n' "$GIT_COMMIT" > "$RUN/git_commit.txt"
  printf '%s\n' "$TEACHER" > "$RUN/teacher_checkpoint.txt"
  export DSOM_RUN_DIR="$RUN"
  export WANDB_NAME="$NOTE"
  export WANDB_DIR="$RUN/wandb"

  ARGS=(
    --dataset "$DATASET" --model ComplExMDR
    --teacher_checkpoint "$TEACHER"
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
    --rank 2000 --batch_size 1000 --optimizer Adagrad
    --learning_rate 0.1 --regularizer N3 --reg 0.01
    --scheduler_mode min --kd_loss_type fixed_ndkd
    --teacher_strategy "$STRATEGY"
    --kd_temp 4 --kd_ce_weight 1 --kd_kl_weight 2
    --dkd_alpha 1 --dkd_beta 1
    --max_epochs 200 --valid 3 --early_stopping 10
    --max_train_examples 0 --eval_queries -1
    --ckpt_dir "$RUN/checkpoints" --note "$NOTE"
    --git_commit "$GIT_COMMIT" --experiment_id "$EXP_ID"
  )
  printf '%q ' "$PYTHON" kd_learn.py "${ARGS[@]}" > "$RUN/command.sh"
  printf '\n' >> "$RUN/command.sh"
  echo "=== FULL: $STRATEGY ==="
  (cd "$ROOT/src" && "$PYTHON" -u kd_learn.py "${ARGS[@]}") 2>&1 | tee "$RUN/student.log"

  "$PYTHON" - "$RUN/checkpoints" "$STRATEGY" "$GIT_COMMIT" "$TEACHER" <<'PY'
import json, sys
from pathlib import Path
folder, strategy, sha, teacher = Path(sys.argv[1]), *sys.argv[2:]
reports = list(folder.rglob('result.json'))
checkpoints = list(folder.rglob('student-*.pth'))
assert len(reports) == len(checkpoints) == 1, (reports, checkpoints)
r = json.loads(reports[0].read_text())
assert r['dataset'] == 'MMKB-DB15K'
assert r['kd_loss_type'] == 'fixed_ndkd'
assert r['teacher_strategy'] == strategy
assert r['max_train_examples'] == 0
assert r['eval_queries'] == -1
assert r['git_commit'] == sha
assert r['teacher_checkpoint'] == teacher
assert r['best_epoch'] > 0
assert 0 <= r['final_result']['MRR'] <= 1
assert len(r['final_result']['hits@[1,3,10]']) == 3
print(f"FULL PASS: {strategy} | best epoch {r['best_epoch']} | test MRR {r['final_result']['MRR']:.6f}")
PY
done

"$PYTHON" - "$BASE" "$BASELINE_STUDENT" <<'PY'
import json, sys
from pathlib import Path
base, baseline_file = Path(sys.argv[1]), Path(sys.argv[2])
records=[]
for name, p in [('source_frozen_rc', baseline_file)] + [
    (s, next((base / s / 'checkpoints').rglob('result.json')))
    for s in ['structural', 'description', 'structural_description', 'average']
]:
    r=json.loads(p.read_text())
    t=r['final_result']
    records.append(dict(strategy=name, best_epoch=r['best_epoch'], valid_mrr=r['best_mrr'],
                        mr=t['MR'], mrr=t['MRR'], h1=t['hits@[1,3,10]'][0],
                        h3=t['hits@[1,3,10]'][1], h10=t['hits@[1,3,10]'][2],
                        result_file=str(p)))
print('\n=== DB15K FULL ABLATION: BEST-CHECKPOINT TEST ===')
print(f"{'strategy':25s} {'best':>5s} {'valid MRR':>10s} {'test MRR':>10s} {'H@1':>10s} {'H@3':>10s} {'H@10':>10s}")
for r in records:
    print(f"{r['strategy']:25s} {r['best_epoch']:5d} {r['valid_mrr']:10.6f} {r['mrr']:10.6f} {r['h1']:10.6f} {r['h3']:10.6f} {r['h10']:10.6f}")
(base / 'summary.json').write_text(json.dumps(records,indent=2)+'\n')
print('Summary:', base / 'summary.json')
PY

echo "DB15K FIXED TEACHER NDKD FULL PIPELINE: PASS"
