#!/usr/bin/env bash
# Retry-with-backoff finetune launcher — survives transient SIGKILL during encoder build.
# Usage: bash scripts/finetune_v9_watchdog.sh <SEED> <GPU>
# Logs: .ops/v9_finetune_s<SEED>.log (training log) + .ops/watchdog_s<SEED>.log (retry events)

set -u
cd ${EQUISSL_ROOT}
SEED=${1:?seed required}
GPU=${2:?gpu required}
OUT=/mnt/ssd/phase2_staging/v9_c6area_s${SEED}
LOG=.ops/v9_finetune_s${SEED}.log
WLOG=.ops/watchdog_s${SEED}.log
MAX_RETRIES=20
SUCCESS_MARKER="Loaded encoder"   # if we see this, build survived

mkdir -p "$OUT"
echo "[$(date)] watchdog s${SEED} gpu${GPU} starting" >> "$WLOG"

for attempt in $(seq 1 $MAX_RETRIES); do
  echo "[$(date)] attempt $attempt/$MAX_RETRIES" >> "$WLOG"
  : > "$LOG"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC} \
    CUDA_VISIBLE_DEVICES=$GPU \
    nice -n 19 ionice -c 3 \
    python3 -u tools/finetune_seg.py \
      --pretrained outputs/pretrain_v9/checkpoint_epoch99.pth \
      --config configs/pretrain_v9_repaired.yaml \
      --output_dir "$OUT" \
      --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 0 \
      --load_decoder --loss ce_dice --dice_weight 0.5 \
      --rpe_mode equivariant --n_gauges 6 --seed $SEED \
      >> "$LOG" 2>&1
  EXIT=$?

  if grep -q "$SUCCESS_MARKER" "$LOG"; then
    if [ $EXIT -eq 0 ]; then
      echo "[$(date)] attempt $attempt: FINETUNE COMPLETE" >> "$WLOG"
      exit 0
    fi
    # Build OK but training crashed. Check if it's a transient SIGKILL (DataLoader worker, etc.)
    if grep -qE "killed by signal|signal: Killed|exitcode -9|exit_code -9" "$LOG"; then
      echo "[$(date)] attempt $attempt: build OK, transient SIGKILL during training. Backing off 60s and retrying." >> "$WLOG"
      sleep 60
    else
      echo "[$(date)] attempt $attempt: build OK but unexpected crash (exit=$EXIT). Stopping." >> "$WLOG"
      exit 1
    fi
  else
    echo "[$(date)] attempt $attempt: died during build (exit=$EXIT, likely SIGKILL). Backing off 90s." >> "$WLOG"
    sleep 90
  fi
done
echo "[$(date)] EXHAUSTED $MAX_RETRIES retries. Giving up." >> "$WLOG"
exit 2
