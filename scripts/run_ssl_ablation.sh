#!/bin/bash
# SSL pretrain + RPE variant finetune on {Stanford2D3D, Structured3D}
# Matches paper's SSL ablation: EquiSSL (v8 iBOT+MAE) init vs random init
# Usage: bash scripts/run_ssl_ablation.sh <gpu_id> <dataset> <rpe_mode> [n_gauges]
#   dataset: s2d3d or s3d
#   rpe_mode: none / standard / equivariant
set -e
GPU=${1:?usage: $0 <gpu_id> <dataset> <rpe_mode> [n_gauges]}
DATASET=${2:?usage: $0 <gpu_id> <dataset> <rpe_mode> [n_gauges]}
RPE=${3:?usage: $0 <gpu_id> <dataset> <rpe_mode> [n_gauges]}
NGAUGES=${4:-6}

export PYTHONPATH="${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH}"
SSL_CKPT="outputs/pretrain_v8/checkpoint_epoch99.pth"
CFG="configs/pretrain_v8_large.yaml"

if [ "$DATASET" = "s2d3d" ]; then
    DATADIR="/mnt/ssd/Stanford2D3D_extracted"
    [ ! -d "$DATADIR" ] && DATADIR="${STANFORD2D3D_PATH}"
    EPOCHS=350
    LABEL_FRAC=1.0
    LOSS="ce_dice"
    DSARG=""
elif [ "$DATASET" = "s3d" ]; then
    DATADIR="${STRUCTURED3D_PATH}_new/Structured3D"
    EPOCHS=50
    LABEL_FRAC=1.0
    LOSS="ce"
    DSARG="--dataset s3d"
else
    echo "unknown dataset: $DATASET"; exit 1
fi

EXTRA=""
if [ "$RPE" = "equivariant" ]; then
    EXTRA="--n_gauges $NGAUGES"
    LABEL="c${NGAUGES}"
else
    LABEL="$RPE"
fi

TAG="ssl_${DATASET}_${LABEL}"
echo "========================================"
echo "GPU=$GPU  DATASET=$DATASET  RPE=$RPE  INIT=SSL"
echo "OUT=outputs/$TAG  EPOCHS=$EPOCHS  LOSS=$LOSS"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
    --pretrained "$SSL_CKPT" \
    --output_dir "outputs/$TAG" \
    --config "$CFG" \
    --data_dir "$DATADIR" \
    $DSARG \
    --epochs $EPOCHS --lr 1e-4 --loss $LOSS --seed 42 \
    --label_fraction $LABEL_FRAC \
    --batch_size 8 \
    --rpe_mode "$RPE" $EXTRA

echo "Done: $TAG"
