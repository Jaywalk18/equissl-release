#!/bin/bash
# Launch the v10 contrastive SSL rescue pretrain on all visible GPUs.

set -euo pipefail

cd ${EQUISSL_ROOT}
export PYTHONPATH=${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH:-}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export TMPDIR=/mnt/ssd/tmp
mkdir -p "$TMPDIR" .ops outputs/pretrain_v10_contrastive

CONFIG=${CONFIG:-configs/pretrain_v10_contrastive.yaml}
OUT=${OUT:-outputs/pretrain_v10_contrastive}
LOG=${LOG:-.ops/pretrain_v10_contrastive.log}
NPROC=${NPROC:-3}
PORT=${PORT:-29610}

RESUME_ARGS=()
if [ -f "$OUT/checkpoint_latest.pth" ]; then
  RESUME_ARGS=(--resume "$OUT/checkpoint_latest.pth")
fi

echo "[$(date '+%F %T')] Starting SSL rescue pretrain" | tee -a "$LOG"
echo "CONFIG=$CONFIG OUT=$OUT NPROC=$NPROC PORT=$PORT" | tee -a "$LOG"
if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
  echo "Resuming from $OUT/checkpoint_latest.pth" | tee -a "$LOG"
fi

CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$PORT" \
  tools/pretrain.py \
  --config "$CONFIG" \
  --output_dir "$OUT" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] SSL rescue pretrain finished" | tee -a "$LOG"
