#!/usr/bin/env bash
set -euo pipefail

# DSoM / FB15K-237: full multimodal teacher pre-training
# Run from anywhere: bash scripts/train_teacher_fb15k237.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../src/learn.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/src/learn.py" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Error: cannot find DSoM/src/learn.py (place script in scripts/ or repo root)." >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/dsom/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Error: Python environment not found: ${PYTHON}" >&2; exit 1; }

DATA_DIR="${REPO_ROOT}/data/FB15K-237"
for f in train.pickle valid.pickle test.pickle to_skip.pickle img_vec.pickle dscp_vec.pickle; do
    [[ -f "${DATA_DIR}/${f}" ]] || { echo "Error: missing ${DATA_DIR}/${f}" >&2; exit 1; }
done

RUN_NAME="teacher_fb15k237_full"
CKPT_DIR="${REPO_ROOT}/ckpt/${RUN_NAME}"
LOG_DIR="${REPO_ROOT}/logs/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${REPO_ROOT}/src"  # required by datasets.py's relative data path

echo "Repository: ${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "Log: ${LOG_FILE}"

"${PYTHON}" -u learn.py \
    --dataset FB15K-237 \
    --model ComplExMDR \
    --modality_split True \
    --fusion_img True \
    --fusion_dscp True \
    --img_info "${DATA_DIR}/img_vec.pickle" \
    --dscp_info "${DATA_DIR}/dscp_vec.pickle" \
    --modal_missing 0.0 \
    --rank 2000 \
    --batch_size 1000 \
    --max_epochs 200 \
    --valid 3 \
    --optimizer Adagrad \
    --learning_rate 0.1 \
    --regularizer N3 \
    --reg 0.01 \
    --init 0.001 \
    --early_stopping 10 \
    --scale 20 \
    --alpha 1 \
    --note "${RUN_NAME}" \
    --ckpt_dir "${CKPT_DIR}" \
    2>&1 | tee "${LOG_FILE}"