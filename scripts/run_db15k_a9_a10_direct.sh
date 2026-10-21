#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_db15k_a9_a10_direct.sh --prepare
# bash scripts/run_db15k_a9_a10_direct.sh --smoke
# bash scripts/run_db15k_a9_a10_direct.sh --full

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/dsom/bin/python"
DATASET=MMKB-DB15K
TEACHER_DIR="$ROOT/runs/teacher/$DATASET/full/T1_DB15K_Exp01_BERT_ViT_Teacher_SchMin_BS1000_R2000_LR0p1_N3_Seed2022/20261021_151535_2887785/checkpoints"
A5_RESULT="$ROOT/runs/student/$DATASET/learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json"
EXP_ID=T4_DB15K_A9_A10_LearnedRC_LossComparison
MODE="${1:---help}"

cd "$ROOT"
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }

# ============================================================
# Step 1: Apply RC loss patch
# ============================================================

if [[ "$MODE" == --prepare ]]; then
  "$PY" - "$ROOT" <<'PY'
import ast
import datetime
import sys
from pathlib import Path

root = Path(sys.argv[1])
learn = root / "src/kd_learn.py"
optim = root / "src/kd_optimizers.py"

a = learn.read_text()
b = optim.read_text()

arg_key = '"--rc_distill_loss"'
loss_key = 'rc_distill_loss = getattr(self.args, "rc_distill_loss", "ndkd")'

if (arg_key in a) != (loss_key in b):
    raise RuntimeError("Partial patch detected; inspect sources")

if arg_key not in a:
    parse_anchor = "args = parser.parse_args()"
    assert a.count(parse_anchor) == 1

    add = '''parser.add_argument(
    "--rc_distill_loss",
    choices=["ndkd", "kd", "dkd"],
    default="ndkd",
    help="RC loss objective; ndkd preserves A5/A6",
)

'''
    a2 = a.replace(parse_anchor, add + parse_anchor, 1)

    reward_line = "                    l = l_ce + l_reg + l_kd + l_reward"
    assert b.count(reward_line) == 1

    pos = b.index(reward_line)
    old_start = "                    l_kd = self.kd_weight * multi_dkd_loss("
    begin = b.rfind(old_start, 0, pos)

    assert begin >= 0
    original = b[begin:pos]
    assert "golden_truths," in original
    assert original.count("multi_dkd_loss(") == 1
    assert "l = " not in original
    assert "elif self.kd_loss" not in original

    default_block = "".join(
        "    " + line
        for line in original.splitlines(keepends=True)
    )

    replacement = (
        '                    rc_distill_loss = getattr(self.args, "rc_distill_loss", "ndkd")\n'
        '                    if rc_distill_loss == "ndkd":\n'
        + default_block
        + '                    elif rc_distill_loss == "kd":\n'
        + '                        l_kd = self.kd_weight * kd_loss(\n'
        + '                            preds_student, preds_selected, self.temperature\n'
        + '                        )\n'
        + '                    elif rc_distill_loss == "dkd":\n'
        + '                        l_kd = self.kd_weight * dkd_loss(\n'
        + '                            preds_student, preds_selected, truth,\n'
        + '                            self.args.dkd_alpha, self.args.dkd_beta, self.temperature,\n'
        + '                        )\n'
        + '                    else:\n'
        + '                        raise ValueError(f"Unexpected RC loss: {rc_distill_loss}")\n'
    )

    b2 = b[:begin] + replacement + b[pos:]

    ast.parse(a2)
    ast.parse(b2)

    backup = (
        root / "runs" / "patch_backups"
        / ("a9_a10_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    backup.mkdir(parents=True, exist_ok=False)
    (backup / "kd_learn.py").write_text(a)
    (backup / "kd_optimizers.py").write_text(b)

    try:
        learn.write_text(a2)
        optim.write_text(b2)
    except Exception:
        learn.write_text(a)
        optim.write_text(b)
        raise

    print("PATCHED:", learn, optim)
    print("Backup:", backup)
else:
    print("ALREADY PATCHED")

for p in (learn, optim):
    ast.parse(p.read_text(), filename=str(p))

assert arg_key in learn.read_text()
assert loss_key in optim.read_text()
print("A9/A10 PATCH CHECK: PASS")
PY
  exit 0
fi

# ============================================================
# Step 2: Preflight
# ============================================================

[[ "$MODE" == --smoke || "$MODE" == --full ]] || {
  echo "Usage: $0 --prepare|--smoke|--full" >&2
  exit 2
}

"$PY" - "$ROOT" <<'PY'
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
a = (root / "src/kd_learn.py").read_text()
b = (root / "src/kd_optimizers.py").read_text()

assert '"--rc_distill_loss"' in a, "Run --prepare first"
assert 'rc_distill_loss = getattr(self.args, "rc_distill_loss", "ndkd")' in b
assert 'elif rc_distill_loss == "kd":' in b
assert 'elif rc_distill_loss == "dkd":' in b

for f in ("kd_learn.py", "kd_optimizers.py"):
    ast.parse((root / "src" / f).read_text(), filename=f)

print("PATCH CHECK: PASS")
PY

if [[ "$MODE" == --full ]]; then
  git ls-files --error-unmatch \
    src/kd_learn.py \
    src/kd_optimizers.py \
    scripts/run_db15k_a9_a10_direct.sh >/dev/null

  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Commit tracked changes before full training" >&2
    exit 1
  fi
fi

mapfile -t TEACHERS < <(
  find "$TEACHER_DIR" -type f -name 'm-*.pth' -print | sort
)

[[ "${#TEACHERS[@]}" -eq 1 ]] || {
  echo "Teacher checkpoint missing or ambiguous" >&2
  exit 1
}

TEACHER="${TEACHERS[0]}"

for f in \
  "data/$DATASET/img_vec.pickle" \
  "data/$DATASET/dscp_vec.pickle"; do
  [[ -s "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

if [[ -f "$(dirname "$TEACHER_DIR")/feature_sha256.txt" ]]; then
  sha256sum -c "$(dirname "$TEACHER_DIR")/feature_sha256.txt"
fi

if [[ "$MODE" == --full ]]; then
  "$PY" - "$A5_RESULT" "$TEACHER" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
assert p.exists(), f"Missing A5 baseline: {p}"

r = json.loads(p.read_text())
assert r["kd_loss_type"] == "kdrl"
assert r["teacher_strategy"] == "rc"
assert r["policy_optimizer"] == "Adam"
assert abs(r["policy_lr"] - 1e-5) < 1e-12
assert r["rc_reward_mode"] == "binary"
assert r["teacher_checkpoint"] == sys.argv[2]
assert r["max_train_examples"] == 0
assert r["eval_queries"] == -1

print("A5 baseline verified:", r["final_result"]["MRR"])
PY

  "$PY" -c 'import wandb; print("W&B:", wandb.__version__)'
fi

# ============================================================
# Step 3: Setup experiment directories
# ============================================================

COMMIT="$(git rev-parse HEAD)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
BASE="$ROOT/runs/student/$DATASET/table4_learned_rc_a9_a10/${MODE#--}/$RUN_ID"

mkdir -p "$BASE"
printf '%s\n' "$COMMIT" > "$BASE/git_commit.txt"

sha256sum \
  "$TEACHER" \
  "data/$DATASET/img_vec.pickle" \
  "data/$DATASET/dscp_vec.pickle" \
  > "$BASE/input_sha256.txt"

export WANDB_PROJECT="${WANDB_PROJECT:-DSoM-Reproduction}"
export WANDB_RUN_GROUP="${EXP_ID}_${MODE#--}_${RUN_ID}"

echo "Mode: $MODE"
echo "Results: $BASE"
echo "Teacher: $TEACHER"

# ============================================================
# Step 4: Training function
# ============================================================

run_one() {
  local label="$1"
  local loss="$2"
  local run="$BASE/$1"
  local note="${EXP_ID}_${label}_${MODE#--}_Seed2022"

  mkdir -p "$run/checkpoints" "$run/wandb"

  local epochs=200 valid=3 early=10
  local max_train=0 eval_queries=-1

  if [[ "$MODE" == --smoke ]]; then
    epochs=1
    valid=1
    early=2
    max_train=3000
    eval_queries=64
  fi

  local -a args=(
    --dataset "$DATASET"
    --model ComplExMDR
    --teacher_checkpoint "$TEACHER"
    --img_info "$ROOT/data/$DATASET/img_vec.pickle"
    --dscp_info "$ROOT/data/$DATASET/dscp_vec.pickle"
    --rank 2000
    --batch_size 1000
    --optimizer Adagrad
    --learning_rate 0.1
    --regularizer N3
    --reg 0.01
    --scheduler_mode min
    --kd_loss_type kdrl
    --teacher_strategy rc
    --rc_distill_loss "$loss"
    --policy_optimizer Adam
    --policy_lr 0.00001
    --kd_temp 4
    --kd_ce_weight 1
    --kd_kl_weight 2
    --dkd_alpha 1
    --dkd_beta 1
    --rl_hs 1024
    --rl_reward 1
    --rl_penalize 10
    --rl_weight 1
    --policy_temp 1
    --rc_reward_mode binary
    --max_epochs "$epochs"
    --valid "$valid"
    --early_stopping "$early"
    --max_train_examples "$max_train"
    --eval_queries "$eval_queries"
    --ckpt_dir "$run/checkpoints"
    --note "$note"
    --git_commit "$COMMIT"
    --experiment_id "$EXP_ID"
  )

  printf '%q ' "$PY" kd_learn.py "${args[@]}" > "$run/command.sh"
  printf '\n' >> "$run/command.sh"

  echo "START $label ($loss)"

  if ! (
    export DSOM_WANDB=0

    if [[ "$MODE" == --full ]]; then
      export DSOM_WANDB=1
      export WANDB_NAME="$note"
      export WANDB_DIR="$run/wandb"
    fi

    cd "$ROOT/src"
    "$PY" -u kd_learn.py "${args[@]}"
  ) > "$run/train.log" 2>&1; then
    echo "FAILED $label. See $run/train.log" >&2
    tail -n 25 "$run/train.log" >&2
    return 1
  fi

  "$PY" - "$run" "$loss" "$MODE" <<'PY'
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
loss = sys.argv[2]
mode = sys.argv[3]

p = list((run / "checkpoints").rglob("result.json"))
assert len(p) == 1, p

r = json.loads(p[0].read_text())
t = r["final_result"]

assert r["kd_loss_type"] == "kdrl"
assert r["teacher_strategy"] == "rc"
assert r["rc_distill_loss"] == loss
assert r["policy_optimizer"] == "Adam"
assert r["rc_reward_mode"] == "binary"
assert abs(r["policy_lr"] - 1e-5) < 1e-12
assert r["best_epoch"] > 0

if mode == "--full":
    assert r["max_train_examples"] == 0
    assert r["eval_queries"] == -1
else:
    assert r["max_train_examples"] == 3000
    assert r["eval_queries"] == 64

log = (run / "train.log").read_text(errors="replace")
updates = [
    int(x)
    for x in re.findall(
        r"Batches with updated policy weights:\s*(\d+)",
        log,
    )
]
assert updates and max(updates) > 0, "RC policy never updated"

print(
    f"PASS {run.name}: "
    f"best_epoch={r['best_epoch']}, "
    f"MRR={t['MRR']:.6f}, "
    f"epochs_with_policy_updates={sum(x > 0 for x in updates)}"
)
PY
}

# ============================================================
# Step 5: Parallel A9 + A10
# ============================================================

run_one A9_KL kd &
pid9=$!

run_one A10_DKD dkd &
pid10=$!

s9=0
s10=0

wait "$pid9" || s9=$?
wait "$pid10" || s10=$?

if [[ "$s9" -ne 0 || "$s10" -ne 0 ]]; then
  echo "FAIL A9=$s9 A10=$s10" >&2
  exit 1
fi

# ============================================================
# Step 6: Summarize best-checkpoint metrics
# ============================================================

"$PY" - "$BASE" "$A5_RESULT" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
a5 = Path(sys.argv[2])
mode = sys.argv[3]

files = [
    ("A9_KL", next((base / "A9_KL/checkpoints").rglob("result.json"))),
    ("A10_DKD", next((base / "A10_DKD/checkpoints").rglob("result.json"))),
]

if mode == "--full":
    files.insert(0, ("A5_NDKD", a5))

rows = []

print("\n=== DB15K LEARNED RC: BEST-CHECKPOINT TEST ===")
print(
    f'{"Method":<12} {"Epoch":>6} {"Valid MRR":>10} '
    f'{"Test MRR":>10} {"H@1":>9} {"H@3":>9} {"H@10":>9}'
)

for name, path in files:
    r = json.loads(path.read_text())
    t = r["final_result"]
    h = t["hits@[1,3,10]"]

    rows.append({
        "method": name,
        "epoch": r["best_epoch"],
        "valid_mrr": r["best_mrr"],
        "test": t,
        "result_file": str(path),
    })

    print(
        f'{name:<12} {r["best_epoch"]:6d} '
        f'{r["best_mrr"]:10.6f} {t["MRR"]:10.6f} '
        f'{h[0]:9.6f} {h[1]:9.6f} {h[2]:9.6f}'
    )

(base / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("Summary:", base / "summary.json")
PY

echo "A9/A10 PIPELINE: PASS"
