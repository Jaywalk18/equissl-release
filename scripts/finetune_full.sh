#!/bin/bash
# EquiSSL Single-Stage Full Finetuning
#
# Usage:
#   bash scripts/finetune_full.sh <pretrained_checkpoint> <output_name> [gpu_id] [extra_args...]
#   bash scripts/finetune_full.sh outputs/pretrain_v3/checkpoint_epoch99.pth v7_full 0

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

OUTPUT_DIR="outputs/finetune_${NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "EquiSSL Full Finetuning: ${NAME}"
echo "  Pretrained: ${PRETRAINED}"
echo "  GPU: ${GPU}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Extra args: ${EXTRA_ARGS}"
echo "  Start: $(date)"
echo "============================================"

python tools/finetune_seg.py \
    --pretrained "${PRETRAINED}" \
    --config configs/pretrain.yaml \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 400 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --load_decoder \
    ${EXTRA_ARGS} \
    2>&1 | tee "${OUTPUT_DIR}/finetune.log"

echo ""
echo "[Pose35] Evaluating rotation robustness..."
for SPLIT in val test; do
    python tools/eval_pose35.py \
        --checkpoint "${OUTPUT_DIR}/best_model.pth" \
        --config configs/pretrain.yaml \
        --max_angle 35.0 --num_rotations 10 --num_repeats 3 \
        --batch_size 4 --num_workers 8 --split ${SPLIT} \
        2>&1 | tee "${OUTPUT_DIR}/pose35_${SPLIT}.log"
done

echo ""
echo "============================================"
echo "Done: ${NAME} at $(date)"
echo "============================================"
