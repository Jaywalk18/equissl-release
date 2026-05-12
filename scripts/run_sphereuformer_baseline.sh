#!/bin/bash
# Train SphereUFormer baseline + rotation degradation curve
# GPU 0 only
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/mnt/ssd/tmp
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"

UFORMER_SRC="${SPHERE_UFORMER_SRC}"
# NOTE: SphereUFormer's loader joins {root_dir}/stanford2d3d/{split_path}.
# A symlink ${STANFORD2D3D_PATH}/stanford2d3d -> extracted bridges the layout.
DATA_ROOT="${STANFORD2D3D_PATH}"
LOG_DIR="${PROJECT_ROOT}/outputs/sphereuformer_baseline"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Phase 1: Train SphereUFormer (segmentation)"
echo "  Data: $DATA_ROOT"
echo "  Log:  $LOG_DIR"
echo "  GPU:  0"
echo "  Start: $(date)"
echo "============================================"

cd "$UFORMER_SRC"
python train.py \
    --task segmentation \
    --dataset_name stanford2d3d \
    --dataset_root_dir "$DATA_ROOT" \
    --log_dir "$LOG_DIR" \
    --num_epochs 400 \
    --train_batch_size 16 \
    --val_batch_size 10 \
    --learning_rate 1e-4 \
    --num_workers 8 \
    --save_frequency 50 \
    --img_rank 7 \
    --img_width 512 \
    --num_scales 4 \
    --scale_depth 2 \
    --win_size_coef 2 \
    --scale_factor 2 \
    --d_head_coef 2 \
    --use_checkpoint 1 \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "============================================"
echo "Phase 1 done at $(date)"
echo "============================================"

# Phase 2: Rotation degradation curve using best model
cd "$PROJECT_ROOT"
CKPT="$LOG_DIR/best_model.pth"

if [ ! -f "$CKPT" ]; then
    echo "WARNING: best_model.pth not found, using last saved model"
    CKPT="$LOG_DIR/models/model.pth"
fi

echo ""
echo "============================================"
echo "Phase 2: Rotation Curve (SphereUFormer)"
echo "  Checkpoint: $CKPT"
echo "  Start: $(date)"
echo "============================================"

CURVE_DIR="${PROJECT_ROOT}/outputs/rotation_curve_sphereuformer"
mkdir -p "$CURVE_DIR"

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
printf "%-8s %-12s %-12s\n" "Angle" "Val mIoU" "Test mIoU"
printf "%-8s %-12s %-12s\n" "-----" "--------" "---------"
for ANGLE in 0 10 20 35 45 60 90; do
    VAL=$(grep -oP "SO\(3\) mIoU.*?(\d+\.\d+)" "${CURVE_DIR}/angle${ANGLE}_val.log" 2>/dev/null | tail -1 | grep -oP "\d+\.\d+" | tail -1 || echo "N/A")
    TEST=$(grep -oP "SO\(3\) mIoU.*?(\d+\.\d+)" "${CURVE_DIR}/angle${ANGLE}_test.log" 2>/dev/null | tail -1 | grep -oP "\d+\.\d+" | tail -1 || echo "N/A")
    printf "%-8s %-12s %-12s\n" "${ANGLE}°" "$VAL" "$TEST"
done
echo "============================================"
echo "All done at $(date)"
