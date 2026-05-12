#!/bin/bash
# Phase 2 only: SphereUFormer rotation degradation curve
# (training already done, see scripts/run_sphereuformer_baseline.sh for Phase 1)
# GPU 0 only
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/mnt/ssd/tmp
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"

CKPT="${PROJECT_ROOT}/outputs/sphereuformer_baseline/best_model.pth"
CURVE_DIR="${PROJECT_ROOT}/outputs/rotation_curve_sphereuformer"
mkdir -p "$CURVE_DIR"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found"
    exit 1
fi

echo "============================================"
echo "SphereUFormer Rotation Curve (Phase 2 only)"
echo "  Checkpoint: $CKPT"
echo "  Output: $CURVE_DIR"
echo "  GPU: 0"
echo "  Start: $(date)"
echo "============================================"

for ANGLE in 0 10 20 35 45 60 90; do
    for SPLIT in val test; do
        echo ""
        echo "--- Angle=${ANGLE}° Split=${SPLIT} ---"
        python tools/eval_pose35_sphereuformer.py \
            --checkpoint "$CKPT" \
            --max_angle ${ANGLE} \
            --num_rotations 10 \
            --num_repeats 3 \
            --batch_size 4 \
            --num_workers 8 \
            --split ${SPLIT} \
            2>&1 | tee "${CURVE_DIR}/angle${ANGLE}_${SPLIT}.log"
    done
done

echo ""
echo "============================================"
echo "=== SphereUFormer Rotation Curve Summary ==="
echo "============================================"
printf "%-8s %-14s %-14s\n" "Angle" "Val mIoU" "Test mIoU"
printf "%-8s %-14s %-14s\n" "-----" "--------" "---------"
for ANGLE in 0 10 20 35 45 60 90; do
    VAL=$(grep -oP "SO\(3\) mIoU:\s+\K[0-9.]+" "${CURVE_DIR}/angle${ANGLE}_val.log" 2>/dev/null | tail -1 || echo "N/A")
    TEST=$(grep -oP "SO\(3\) mIoU:\s+\K[0-9.]+" "${CURVE_DIR}/angle${ANGLE}_test.log" 2>/dev/null | tail -1 || echo "N/A")
    printf "%-8s %-14s %-14s\n" "${ANGLE}°" "$VAL" "$TEST"
done
echo "============================================"
echo "All done at $(date)"
