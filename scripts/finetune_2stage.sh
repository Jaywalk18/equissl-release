#!/bin/bash
# EquiSSL Two-Stage Finetuning Pipeline
# Automatically runs: stage1 (freeze encoder) -> stage2 (full finetune) -> Pose35 eval
#
# Usage:
#   bash scripts/finetune_2stage.sh <pretrained_checkpoint> <output_name> [gpu_id]
#   bash scripts/finetune_2stage.sh outputs/pretrain_v3/checkpoint_epoch99.pth v7_focal 0
#
# Extra args after gpu_id are passed to finetune_seg.py (e.g. --loss focal --gamma 2.0)

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

PRETRAINED="${1:?Usage: $0 <pretrained_checkpoint> <output_name> [gpu_id] [extra_args...]}"
NAME="${2:?Usage: $0 <pretrained_checkpoint> <output_name> [gpu_id] [extra_args...]}"
GPU="${3:-0}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS="$@"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${GPU}
export TMPDIR=/mnt/ssd/tmp

STAGE1_DIR="outputs/finetune_${NAME}_s1"
STAGE2_DIR="outputs/finetune_${NAME}_s2"

mkdir -p "${STAGE1_DIR}" "${STAGE2_DIR}"

echo "============================================"
echo "EquiSSL Two-Stage Finetuning: ${NAME}"
echo "  Pretrained: ${PRETRAINED}"
echo "  GPU: ${GPU}"
echo "  Stage1 output: ${STAGE1_DIR}"
echo "  Stage2 output: ${STAGE2_DIR}"
echo "  Extra args: ${EXTRA_ARGS}"
echo "  Start: $(date)"
echo "============================================"

# --- Stage 1: Freeze encoder, train decoder ---
echo ""
echo "[Stage 1] Freeze encoder, 50 epochs, lr=5e-4"
python tools/finetune_seg.py \
    --pretrained "${PRETRAINED}" \
    --config configs/pretrain.yaml \
    --output_dir "${STAGE1_DIR}" \
    --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \
    --freeze --load_decoder \
    ${EXTRA_ARGS} \
    2>&1 | tee "${STAGE1_DIR}/stage1.log"

echo ""
echo "[Stage 1] Done. Best model: ${STAGE1_DIR}/best_model.pth"

# --- Stage 2: Full finetune from stage1 best ---
echo ""
echo "[Stage 2] Full finetune, 350 epochs, lr=1e-4"
python tools/finetune_seg.py \
    --pretrained "${STAGE1_DIR}/best_model.pth" \
    --config configs/pretrain.yaml \
    --output_dir "${STAGE2_DIR}" \
    --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --load_decoder \
    ${EXTRA_ARGS} \
    2>&1 | tee "${STAGE2_DIR}/stage2.log"

echo ""
echo "[Stage 2] Done. Best model: ${STAGE2_DIR}/best_model.pth"

# --- Pose35 Evaluation ---
echo ""
echo "[Pose35] Evaluating rotation robustness..."

for SPLIT in val test; do
    echo "  Evaluating ${SPLIT}..."
    python tools/eval_pose35.py \
        --checkpoint "${STAGE2_DIR}/best_model.pth" \
        --config configs/pretrain.yaml \
        --max_angle 35.0 --num_rotations 10 --num_repeats 3 \
        --batch_size 4 --num_workers 8 --split ${SPLIT} \
        2>&1 | tee "${STAGE2_DIR}/pose35_${SPLIT}.log"
done

echo ""
echo "============================================"
echo "Pipeline complete: ${NAME}"
echo "  Stage1: ${STAGE1_DIR}/stage1.log"
echo "  Stage2: ${STAGE2_DIR}/stage2.log"
echo "  Pose35: ${STAGE2_DIR}/pose35_val.log, pose35_test.log"
echo "  Finished: $(date)"
echo "============================================"
