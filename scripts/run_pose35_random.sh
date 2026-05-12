#!/bin/bash
# Pose35 on v8_random — quick robustness eval at ±35°.
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2
export TMPDIR=/mnt/ssd/tmp

CKPT="outputs/finetune_v8_random_s/best_model.pth"
CONFIG="configs/pretrain_v8_large.yaml"
OUTDIR="outputs/finetune_v8_random_s"

echo "============================================"
echo "Pose35 — v8_random"
echo "  GPU: 2"
echo "  Start: $(date)"
echo "============================================"

for SPLIT in val test; do
    echo "--- Pose35 ${SPLIT} ---"
    python tools/eval_pose35.py \
        --checkpoint "$CKPT" \
        --config "$CONFIG" \
        --max_angle 35.0 \
        --num_rotations 10 \
        --num_repeats 3 \
        --batch_size 4 \
        --num_workers 8 \
        --split ${SPLIT} \
        2>&1 | tee "${OUTDIR}/pose35_${SPLIT}.log"
done

echo ""
echo "============================================"
echo "=== v8_random Pose35 Summary ==="
echo "============================================"
grep -E "Base mIoU|SO.3. mIoU|Drop" "${OUTDIR}/pose35_val.log" | tail -5
echo "---"
grep -E "Base mIoU|SO.3. mIoU|Drop" "${OUTDIR}/pose35_test.log" | tail -5
echo "============================================"
echo "Done: $(date)"
