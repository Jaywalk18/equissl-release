#!/bin/bash
# S3D fine-tune: 4 RPE variants on Structured3D (native NYU40 40-class)
# Matches SphereUFormer / PanoFormer evaluation protocol for direct SOTA comparison
# Usage: bash scripts/run_s3d_finetune.sh <gpu_id> <rpe_mode> [n_gauges]
set -e
GPU=${1:?usage: $0 <gpu_id> <rpe_mode> [n_gauges]}
RPE=${2:?usage: $0 <gpu_id> <rpe_mode> [n_gauges]}
NGAUGES=${3:-6}

export PYTHONPATH="${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH}"
CKPT="/mnt/ssd/tmp/empty_ckpt.pth"
CFG="configs/pretrain_v8_large.yaml"
DATADIR="${STRUCTURED3D_PATH}_new/Structured3D"

EXTRA=""
if [ "$RPE" = "equivariant" ]; then
    EXTRA="--n_gauges $NGAUGES"
    LABEL="c${NGAUGES}"
else
    LABEL="$RPE"
fi

TAG="s3d_nyu40_${LABEL}"
echo "========================================"
echo "GPU=$GPU  RPE=$RPE  DATASET=s3d  OUT=outputs/$TAG"
echo "40-class NYU40 native, full 18k train, 50 epochs, CE loss, yaw+flip aug"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
    --pretrained "$CKPT" \
    --output_dir "outputs/$TAG" \
    --config "$CFG" \
    --dataset s3d \
    --data_dir "$DATADIR" \
    --epochs 50 --lr 1e-4 --loss ce --seed 42 \
    --label_fraction 1.0 \
    --batch_size 8 \
    --rpe_mode "$RPE" $EXTRA

echo "Done: $TAG"
