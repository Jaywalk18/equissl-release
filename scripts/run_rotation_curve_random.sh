#!/bin/bash
# Rotation curve on v8_random (verify rotation invariance is architecture-driven).
#
# If v8_random gives nearly identical rotation curve to v8_f (bugged),
# it proves rotation invariance comes from Gauge-Equivariant RPE, not SSL.
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/mnt/ssd/tmp

CKPT="outputs/finetune_v8_random_s/best_model.pth"
CONFIG="configs/pretrain_v8_large.yaml"
OUTDIR="outputs/rotation_curve"
mkdir -p "$OUTDIR"

echo "============================================"
echo "Rotation Curve — v8_random (no SSL)"
echo "  Checkpoint: $CKPT"
echo "  Angles: 0 10 20 35 45 60 90"
echo "  GPU: 1"
echo "  Start: $(date)"
echo "============================================"

for ANGLE in 0 10 20 35 45 60 90; do
    for SPLIT in val test; do
        echo ""
        echo "--- Angle=${ANGLE}° Split=${SPLIT} ---"
        python tools/eval_pose35.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --max_angle ${ANGLE} \
            --num_rotations 10 \
            --num_repeats 3 \
            --batch_size 4 \
            --num_workers 8 \
            --split ${SPLIT} \
            2>&1 | tee "${OUTDIR}/v8_random_angle${ANGLE}_${SPLIT}.log"
    done
done

echo ""
echo "============================================"
echo "=== v8_random Rotation Curve Summary ==="
echo "============================================"
printf "%-8s %-12s %-12s\n" "Angle" "Val mIoU" "Test mIoU"
printf "%-8s %-12s %-12s\n" "-----" "--------" "---------"
for ANGLE in 0 10 20 35 45 60 90; do
    VAL=$(grep -oP "mIoU.*?(\d+\.\d+)" "${OUTDIR}/v8_random_angle${ANGLE}_val.log" 2>/dev/null | tail -1 | grep -oP "\d+\.\d+" || echo "N/A")
    TEST=$(grep -oP "mIoU.*?(\d+\.\d+)" "${OUTDIR}/v8_random_angle${ANGLE}_test.log" 2>/dev/null | tail -1 | grep -oP "\d+\.\d+" || echo "N/A")
    printf "%-8s %-12s %-12s\n" "${ANGLE}°" "$VAL" "$TEST"
done

echo ""
echo "Compare to v8_f (bugged, should be similar):"
for ANGLE in 0 10 20 35 45 60 90; do
    VAL=$(grep -oP "mIoU.*?(\d+\.\d+)" "${OUTDIR}/angle${ANGLE}_val.log" 2>/dev/null | tail -1 | grep -oP "\d+\.\d+" || echo "N/A")
    printf "  v8_f @%s°: %s\n" "$ANGLE" "$VAL"
done
echo "============================================"
echo "All done at $(date)"
