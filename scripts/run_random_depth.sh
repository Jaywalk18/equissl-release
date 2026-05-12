#!/bin/bash
# v8 depth with FULLY RANDOM init (paper SSL ablation for depth).
#
# Purpose: quantify SSL contribution to depth estimation.
#   - Encoder: RANDOM (empty checkpoint -> strict=False all missing)
#   - Decoder: random init
#   - 200 epochs, lr 1e-4, batch_size 8
#
# Companion to outputs/depth_v8_s2 (SSL encoder, same config).
# Direct head-to-head comparison to decide whether SSL helps depth.
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export TMPDIR=/mnt/ssd/tmp
export CUDA_VISIBLE_DEVICES=0

OUT="outputs/depth_v8_random_s2"
mkdir -p "$OUT"

echo "============================================"
echo "v8 Depth Random Init Baseline"
echo "  GPU: 0"
echo "  Init:   empty checkpoint (random enc + random dec)"
echo "  Output: $OUT"
echo "  200 epochs, lr 1e-4, batch_size 8"
echo "  Start:  $(date)"
echo "============================================"

python tools/finetune_depth.py \
    --pretrained /mnt/ssd/tmp/empty_ckpt.pth \
    --config configs/pretrain_v8_large.yaml \
    --output_dir "$OUT" \
    --epochs 200 --lr 1e-4 --batch_size 8 --num_workers 8 \
    2>&1 | tee "${OUT}/stage.log"

echo ""
echo "============================================"
echo "Done: $(date)"
echo "============================================"
grep -E "Best val delta1|Test results|delta1:|rmse:|abs_rel:" "${OUT}/stage.log" | tail -20
