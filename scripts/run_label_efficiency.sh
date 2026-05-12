#!/bin/bash
# Label Efficiency experiments — 10%/25%/50% label fractions
# GPU 0 only, runs 3 experiments sequentially
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/mnt/ssd/tmp

PRETRAINED="outputs/pretrain_v3/checkpoint_epoch99.pth"
CONFIG="configs/pretrain.yaml"

echo "============================================"
echo "Label Efficiency Experiments (Exp F config)"
echo "  GPU: 0"
echo "  Fractions: 10%, 25%, 50%"
echo "  Start: $(date)"
echo "============================================"

for FRAC in 0.1 0.25 0.5; do
    FRAC_NAME=$(echo $FRAC | sed 's/0\.//')
    S1="outputs/label_eff_${FRAC_NAME}pct_s1"
    S2="outputs/label_eff_${FRAC_NAME}pct_s2"
    mkdir -p "$S1" "$S2"

    echo ""
    echo "========== Label Fraction: ${FRAC} (${FRAC_NAME}%) =========="
    echo "  Start: $(date)"

    # Stage1: focal, freeze encoder, 50ep
    echo "[Stage1] Focal, freeze, 50ep, label_fraction=${FRAC}"
    python tools/finetune_seg.py \
        --pretrained "${PRETRAINED}" --config "${CONFIG}" \
        --output_dir "${S1}" --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \
        --freeze --load_decoder \
        --loss focal --focal_gamma 2.0 \
        --label_fraction ${FRAC} \
        2>&1 | tee "${S1}/stage1.log"

    # Stage2: CE+Dice, full finetune, 300ep
    echo "[Stage2] CE+Dice(0.5), full, 300ep, label_fraction=${FRAC}"
    python tools/finetune_seg.py \
        --pretrained "${S1}/best_model.pth" --config "${CONFIG}" \
        --output_dir "${S2}" --epochs 300 --lr 1e-4 --batch_size 8 --num_workers 8 \
        --load_decoder \
        --loss ce_dice --dice_weight 0.5 \
        --label_fraction ${FRAC} \
        2>&1 | tee "${S2}/stage2.log"

    echo "--- ${FRAC_NAME}% Results ---"
    grep "Best val mIoU\|Test mIoU" "${S2}/stage2.log"
    echo "  Done: $(date)"
done

echo ""
echo "============================================"
echo "=== Label Efficiency Summary ==="
echo "============================================"
for FRAC in 0.1 0.25 0.5; do
    FRAC_NAME=$(echo $FRAC | sed 's/0\.//')
    echo "--- ${FRAC_NAME}% labels ---"
    grep "Best val mIoU\|Test mIoU" "outputs/label_eff_${FRAC_NAME}pct_s2/stage2.log" 2>/dev/null || echo "  (no results)"
done
echo ""
echo "100% labels (Exp F baseline): val 66.01, test 29.77"
echo "============================================"
echo "All done at $(date)"
