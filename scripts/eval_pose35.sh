#!/bin/bash
# EquiSSL SO(3) Rotation-Robustness Evaluation
#
# Paper-default angle: theta_max = 90 degrees (--max_angle 90.0 below).
# The script name (eval_pose35) is a historical artefact of the original
# +/-35 degree sweep and is retained for backward compatibility with
# checkpoint metadata.
#
# Usage:
#   bash scripts/eval_pose35.sh <checkpoint> [gpu_id] [split]
#   bash scripts/eval_pose35.sh outputs/<run>/best_model.pth 0 val
#   bash scripts/eval_pose35.sh outputs/<run>/best_model.pth 0 test

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

CHECKPOINT="${1:?Usage: $0 <checkpoint> [gpu_id] [split]}"
GPU="${2:-0}"
SPLIT="${3:-val}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${GPU}

OUT_DIR=$(dirname "${CHECKPOINT}")
LOG_FILE="${OUT_DIR}/pose35_${SPLIT}.log"

echo "============================================"
echo "EquiSSL SO(3) Rotation-Robustness Evaluation"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  GPU: ${GPU}"
echo "  Split: ${SPLIT}"
echo "  theta_max: 90.0 deg (paper default)"
echo "  Start: $(date)"
echo "============================================"

python tools/eval_pose35.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/pretrain.yaml \
    --max_angle 90.0 \
    --num_rotations 10 \
    --num_repeats 3 \
    --batch_size 4 \
    --num_workers 8 \
    --split ${SPLIT} \
    2>&1 | tee "${LOG_FILE}"

echo "Finished at $(date)"
