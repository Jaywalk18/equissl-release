#!/bin/bash
# v8 seg at 1% labels with FULLY RANDOM init (SSL label-efficiency ablation).
#
# Purpose: head-to-head random vs SSL at 1% labels (10 train images).
#   - Encoder: RANDOM (empty checkpoint)
#   - Decoder: random init
#   - 350 epochs, lr 1e-4, ce_dice 0.5 (matches v8_direct recipe)
#   - label_fraction 0.01 -> 10 train samples
#
# Companion to run_direct_seg_1pct.sh (same config but SSL encoder).
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export TMPDIR=/mnt/ssd/tmp
export CUDA_VISIBLE_DEVICES=1

OUT="outputs/finetune_v8_random_1pct_s"
mkdir -p "$OUT"

echo "============================================"
echo "v8 Seg @1% Labels — RANDOM init"
echo "  GPU: 1"
echo "  Init:   empty checkpoint (random enc + random dec)"
echo "  Output: $OUT"
echo "  Labels: 1% (~10 train samples)"
echo "  350 epochs, lr 1e-4, ce_dice 0.5"
echo "  Start:  $(date)"
echo "============================================"

python tools/finetune_seg.py \
    --pretrained /mnt/ssd/tmp/empty_ckpt.pth \
    --config configs/pretrain_v8_large.yaml \
    --output_dir "$OUT" \
    --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --loss ce_dice --dice_weight 0.5 \
    --label_fraction 0.01 \
    2>&1 | tee "${OUT}/stage.log"

echo ""
echo "============================================"
echo "Done: $(date)"
echo "============================================"
grep -E "Best val mIoU|Test mIoU" "${OUT}/stage.log" | tail -5
