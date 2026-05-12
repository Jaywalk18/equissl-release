#!/bin/bash
# RPE Ablation: train v8_random with 3 RPE modes, then evaluate rotation curves.
#
# Existing baseline: v8_random with --rpe_mode=standard (default "config" was
# actually standard RPE because finetune_seg.py didn't pass equivariant flags).
# That model is already at outputs/finetune_v8_random_s/ with val 65.99, 0.7% rot drop.
#
# This script runs the two missing conditions:
#   GPU 1: --rpe_mode=none     (no RPE at all)
#   GPU 2: --rpe_mode=equivariant (GE-RPE C6, the paper's proposed method)
#
# After training, each model gets a rotation curve evaluation.
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export TMPDIR=/mnt/ssd/tmp

train_and_eval() {
    local GPU=$1 RPE_MODE=$2 OUT=$3
    export CUDA_VISIBLE_DEVICES=$GPU
    mkdir -p "$OUT"

    echo "============================================"
    echo "RPE Ablation: $RPE_MODE on GPU $GPU"
    echo "  Output: $OUT"
    echo "  Start: $(date)"
    echo "============================================"

    # Train
    python tools/finetune_seg.py \
        --pretrained /mnt/ssd/tmp/empty_ckpt.pth \
        --config configs/pretrain_v8_large.yaml \
        --output_dir "$OUT" \
        --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
        --loss ce_dice --dice_weight 0.5 \
        --rpe_mode "$RPE_MODE" \
        2>&1 | tee "${OUT}/stage.log"

    echo "[${RPE_MODE}] Training done: $(date)"

    # Rotation curve (val only for speed)
    echo "[${RPE_MODE}] Starting rotation curve..."
    for ANGLE in 0 35 90; do
        echo "--- ${RPE_MODE} angle=${ANGLE}° val ---"
        python tools/eval_pose35.py \
            --checkpoint "${OUT}/best_model.pth" \
            --config configs/pretrain_v8_large.yaml \
            --max_angle ${ANGLE} \
            --num_rotations 10 \
            --num_repeats 3 \
            --batch_size 4 \
            --num_workers 8 \
            --split val \
            2>&1 | tee "${OUT}/rot_angle${ANGLE}_val.log"
    done

    echo ""
    echo "============================================"
    echo "=== ${RPE_MODE} Results ==="
    echo "============================================"
    grep -E "Best val mIoU|Test mIoU" "${OUT}/stage.log" | tail -5
    echo "Rotation:"
    for ANGLE in 0 35 90; do
        grep -E "SO.3. mIoU:" "${OUT}/rot_angle${ANGLE}_val.log" 2>/dev/null | tail -1 | sed "s/^/  @${ANGLE}°: /"
    done
    echo "Done: $(date)"
}

# Launch both in parallel via background processes
train_and_eval 1 "none" "outputs/rpe_ablation_none" &
PID_NONE=$!

train_and_eval 2 "equivariant" "outputs/rpe_ablation_equivariant" &
PID_EQ=$!

echo "Launched: none (GPU 1, PID $PID_NONE), equivariant (GPU 2, PID $PID_EQ)"
echo "Waiting..."

wait $PID_NONE
echo "=== none finished ==="

wait $PID_EQ
echo "=== equivariant finished ==="

echo ""
echo "============================================"
echo "=== RPE Ablation Summary ==="
echo "============================================"
echo "Standard RPE (existing v8_random):"
echo "  val mIoU: 65.99, rotation 0->90° drop: 0.7%"
echo ""
echo "No RPE:"
grep -E "Best val mIoU" "outputs/rpe_ablation_none/stage.log" 2>/dev/null | tail -1
for a in 0 35 90; do grep -E "SO.3. mIoU:" "outputs/rpe_ablation_none/rot_angle${a}_val.log" 2>/dev/null | tail -1 | sed "s/^/  @${a}°: /"; done
echo ""
echo "GE-RPE C6 (equivariant):"
grep -E "Best val mIoU" "outputs/rpe_ablation_equivariant/stage.log" 2>/dev/null | tail -1
for a in 0 35 90; do grep -E "SO.3. mIoU:" "outputs/rpe_ablation_equivariant/rot_angle${a}_val.log" 2>/dev/null | tail -1 | sed "s/^/  @${a}°: /"; done
echo ""
echo "All done at $(date)"
