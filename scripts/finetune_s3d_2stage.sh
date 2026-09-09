#!/bin/bash
# S3D (NYU40) two-stage finetune — tuned for SSL-init:
#   Stage 1: freeze encoder, 10 ep, lr=5e-4 (let decoder adapt to SSL features)
#   Stage 2: unfreeze, 40 ep, lr=3e-5, layer_decay=0.65 (gentle update)
#
# Usage: bash scripts/finetune_s3d_2stage.sh <pretrained> <name> <gpu> <rpe_mode> [n_gauges] [target] [target_key]
set -e
cd "$(dirname "$0")/.."

PRETRAINED="$1"; NAME="$2"; GPU="$3"; RPE="$4"; NG="${5:-4}"; TARGET="${6:-0}"; TKEY="${7:-Table2/SSL/$RPE}"

export PYTHONPATH="$(pwd):$(pwd)/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

S1="outputs/${NAME}_s1"; S2="outputs/${NAME}_s2"
mkdir -p "$S1" "$S2"

COMMON=(--config configs/pretrain_v8_large.yaml
        --data_dir ${STRUCTURED3D_PATH}_new/Structured3D --dataset s3d
        --loss ce --seed 42 --label_fraction 1.0 --batch_size 8
        --rpe_mode "$RPE")
[ "$RPE" = "equivariant" ] && COMMON+=(--n_gauges "$NG")

echo "=== $NAME Stage 1: freeze encoder, 10 ep, lr 5e-4 ==="
python tools/finetune_seg.py \
  --pretrained "$PRETRAINED" --output_dir "$S1" \
  --epochs 10 --lr 5e-4 --freeze --load_decoder \
  "${COMMON[@]}" 2>&1 | tee "$S1/stage1.log"

echo "=== $NAME Stage 2: unfreeze, 40 ep, lr 3e-5, layer_decay 0.65 ==="
python tools/finetune_seg.py \
  --pretrained "$S1/best_model.pth" --output_dir "$S2" \
  --epochs 40 --lr 3e-5 --load_decoder --layer_decay 0.65 \
  "${COMMON[@]}" 2>&1 | tee "$S2/stage2.log"

echo "=== $NAME done ==="

if [ "$(echo "$TARGET > 0" | bc 2>/dev/null)" = "1" ]; then
    python .ops/track_experiment.py \
      --name "${NAME}_s2" --target_key "$TKEY" --target "$TARGET" \
      --notes "2-stage freeze10+layerdecay0.65"
fi

