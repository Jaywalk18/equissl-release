#!/bin/bash
# Stanford2D3D two-stage finetune tuned for SSL-init:
#   Stage 1: freeze encoder, 30 ep, lr=5e-4 (let decoder adapt)
#   Stage 2: unfreeze, 200 ep, lr=3e-5 + layer_decay 0.65 (gentle)
#
# Usage: bash scripts/finetune_s2d3d_2stage.sh <pretrained> <name> <gpu> <rpe_mode> [n_gauges] [area_weighted]
set -e
cd "$(dirname "$0")/.."

PRETRAINED="$1"; NAME="$2"; GPU="$3"; RPE="$4"; NG="${5:-4}"; AREA="${6:-1}"; TARGET="${7:-0}"; TKEY="${8:-Table1/SSL/$RPE}"

export PYTHONPATH="$(pwd):$(pwd)/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

S1="outputs/${NAME}_s1"; S2="outputs/${NAME}_s2"
mkdir -p "$S1" "$S2"

COMMON=(--config configs/pretrain_v8_large.yaml
        --data_dir /mnt/ssd/Stanford2D3D_extracted
        --loss ce_dice --seed 42 --label_fraction 1.0 --batch_size 8
        --rpe_mode "$RPE")
[ "$RPE" = "equivariant" ] && COMMON+=(--n_gauges "$NG")
[ "$AREA" = "0" ] && COMMON+=(--no_area_weight)

echo "[$(date)] === $NAME Stage 1: freeze encoder, 30 ep, lr 5e-4 ===" | tee "$S1/stage1.log"
python tools/finetune_seg.py \
  --pretrained "$PRETRAINED" --output_dir "$S1" \
  --epochs 30 --lr 5e-4 --freeze --load_decoder \
  "${COMMON[@]}" 2>&1 | tee -a "$S1/stage1.log"

echo "[$(date)] === $NAME Stage 2: unfreeze, 200 ep, lr 3e-5, layer_decay 0.65 ===" | tee "$S2/stage2.log"
python tools/finetune_seg.py \
  --pretrained "$S1/best_model.pth" --output_dir "$S2" \
  --epochs 200 --lr 3e-5 --load_decoder --layer_decay 0.65 \
  "${COMMON[@]}" 2>&1 | tee -a "$S2/stage2.log"

echo "[$(date)] === Pose35 rotation eval ==="
python tools/eval_pose35.py \
  --checkpoint "$S2/best_model.pth" \
  --config configs/pretrain_v8_large.yaml \
  --max_angle 35.0 --num_rotations 10 --num_repeats 3 \
  --batch_size 4 --split val 2>&1 | tee "$S2/pose35_val.log"
python tools/eval_pose35.py \
  --checkpoint "$S2/best_model.pth" \
  --config configs/pretrain_v8_large.yaml \
  --max_angle 90.0 --num_rotations 10 --num_repeats 3 \
  --batch_size 4 --split val 2>&1 | tee "$S2/pose90_val.log"

echo "[$(date)] === $NAME done ==="

if [ "$(echo "$TARGET > 0" | bc 2>/dev/null)" = "1" ]; then
    python .ops/track_experiment.py \
      --name "${NAME}_s2" --target_key "$TKEY" --target "$TARGET" \
      --notes "2-stage freeze30+layerdecay0.65 + Pose35"
fi

