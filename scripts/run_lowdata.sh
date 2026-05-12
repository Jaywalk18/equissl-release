#!/bin/bash
# Low-data ablation: 3 fractions × 4 RPE variants
# Usage: bash scripts/run_lowdata.sh <gpu_id> <rpe_mode> [n_gauges]
# Examples:
#   bash scripts/run_lowdata.sh 0 none
#   bash scripts/run_lowdata.sh 1 standard
#   bash scripts/run_lowdata.sh 2 equivariant 4

set -e
GPU=${1:?usage: $0 <gpu_id> <rpe_mode> [n_gauges]}
RPE=${2:?usage: $0 <gpu_id> <rpe_mode> [n_gauges]}
NGAUGES=${3:-6}

export PYTHONPATH="${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH}"
CKPT="/mnt/ssd/tmp/empty_ckpt.pth"
CFG="configs/pretrain_v8_large.yaml"
# Use SSD copy if available
DATADIR="/mnt/ssd/Stanford2D3D_extracted"
[ ! -d "$DATADIR" ] && DATADIR="${STANFORD2D3D_PATH}"

EXTRA=""
if [ "$RPE" = "equivariant" ]; then
    EXTRA="--n_gauges $NGAUGES"
    LABEL="c${NGAUGES}"
else
    LABEL="$RPE"
fi

for FRAC in 0.01 0.05 0.10; do
    TAG="label_eff_${LABEL}_${FRAC}"
    echo "========================================"
    echo "GPU=$GPU  RPE=$RPE  FRAC=$FRAC  OUT=outputs/$TAG"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
        --pretrained "$CKPT" \
        --output_dir "outputs/$TAG" \
        --config "$CFG" \
        --epochs 350 --lr 1e-4 --loss ce_dice --seed 42 \
        --label_fraction "$FRAC" \
        --data_dir "$DATADIR" \
        --rpe_mode "$RPE" $EXTRA
    echo "Done: $TAG"
done
echo "All fractions done for $LABEL on GPU $GPU"
