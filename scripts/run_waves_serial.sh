#!/bin/bash
# Phase-2 wave-serial runner: 1 experiment = 1 config × 3 seeds in parallel
# across GPUs 0/1/2. Next wave starts after current wave's 3 seeds all done.
#
# This matches the "每个实验都用三张卡 然后实验串行" brief: each experiment
# occupies all 3 GPUs for 3 seeds, no cross-experiment parallelism.
#
# Advantages vs 3-parallel-queue design:
#   - No job-collision concerns (each wave is a clean 3-seed cohort)
#   - Uniform GPU treatment per experiment
#   - Clearer progress tracking ("wave N/16 done")
#   - Mid-wave harvest can check 3-seed convergence before next wave
#
# Usage:
#   bash scripts/run_waves_serial.sh                 # run all 16 waves
#   bash scripts/run_waves_serial.sh seg_rand_c4     # run one specific wave
#   WAVE_FROM=5 bash scripts/run_waves_serial.sh     # start from wave 5
#
# Logs: .ops/waves.log + .ops/wave_<name>_seed<n>.log

set -o pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

mkdir -p .ops
MAIN_LOG=".ops/waves.log"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export TMPDIR=/mnt/ssd/tmp

# 16 waves in paper-priority order. Each entry: "wave_name:kind:init:rpe"
# kind=seg|depth, init=rand|ibot, rpe=none|standard|c4|c6noarea.
WAVES=(
  # Priority 1: Random 1%-seg (3 anchors already measured for seed 42)
  "seg_rand_none:seg:rand:none"
  "seg_rand_standard:seg:rand:standard"
  "seg_rand_c4:seg:rand:c4"
  "seg_rand_c6noarea:seg:rand:c6noarea"
  # Priority 2: iBOT 1%-seg (SSL low-label keystone)
  "seg_ibot_none:seg:ibot:none"
  "seg_ibot_standard:seg:ibot:standard"
  "seg_ibot_c4:seg:ibot:c4"
  "seg_ibot_c6noarea:seg:ibot:c6noarea"
  # Priority 3: Random depth (lighter, ~3h/seed)
  "depth_rand_none:depth:rand:none"
  "depth_rand_standard:depth:rand:standard"
  "depth_rand_c4:depth:rand:c4"
  "depth_rand_c6noarea:depth:rand:c6noarea"
  # Priority 4: iBOT depth
  "depth_ibot_none:depth:ibot:none"
  "depth_ibot_standard:depth:ibot:standard"
  "depth_ibot_c4:depth:ibot:c4"
  "depth_ibot_c6noarea:depth:ibot:c6noarea"
)

rpe_flags() {
  case "$1" in
    none)     echo "--rpe_mode none" ;;
    standard) echo "--rpe_mode standard" ;;
    c4)       echo "--rpe_mode equivariant --n_gauges 4" ;;
    c6noarea) echo "--rpe_mode equivariant --n_gauges 6 --no_area_weight" ;;
  esac
}

init_ckpt() {
  case "$1" in
    rand) echo "/mnt/ssd/tmp/empty_ckpt.pth" ;;
    ibot) echo "outputs/pretrain_v8/checkpoint_epoch99.pth" ;;
  esac
}

run_seed_seg() {
  local INIT=$1 RPE=$2 SEED=$3 GPU=$4
  local OUT="/mnt/ssd/phase2_staging/seg_${INIT}_${RPE}_s${SEED}"
  local RPE_F; RPE_F=$(rpe_flags "$RPE")
  local CKPT; CKPT=$(init_ckpt "$INIT")

  if [ -f "${OUT}/results.pth" ]; then
    echo "[$(date)] SKIP seg ${INIT} ${RPE} s${SEED} (results.pth exists)" >> "$MAIN_LOG"
    return 0
  fi

  if [ "$INIT" = "rand" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
      --pretrained "$CKPT" --config configs/pretrain_v8_large.yaml \
      --output_dir "$OUT" \
      --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
      --loss ce_dice --dice_weight 0.5 \
      --label_fraction 0.01 --seed $SEED $RPE_F \
      > ".ops/wave_seg_${INIT}_${RPE}_seed${SEED}.log" 2>&1
  else
    local S1="${OUT}_stage1" S2="${OUT}_stage2"
    CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
      --pretrained "$CKPT" --config configs/pretrain_v8_large.yaml \
      --output_dir "$S1" \
      --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \
      --freeze --load_decoder \
      --loss ce_dice --dice_weight 0.5 \
      --label_fraction 0.01 --seed $SEED $RPE_F \
      > ".ops/wave_seg_${INIT}_${RPE}_seed${SEED}_s1.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_seg.py \
      --pretrained "${S1}/best_model.pth" --config configs/pretrain_v8_large.yaml \
      --output_dir "$S2" \
      --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
      --load_decoder \
      --loss ce_dice --dice_weight 0.5 \
      --label_fraction 0.01 --seed $SEED $RPE_F \
      > ".ops/wave_seg_${INIT}_${RPE}_seed${SEED}_s2.log" 2>&1
  fi
}

run_seed_depth() {
  local INIT=$1 RPE=$2 SEED=$3 GPU=$4
  local S1="/mnt/ssd/phase2_staging/depth_${INIT}_${RPE}_s${SEED}_stage1"
  local S2="/mnt/ssd/phase2_staging/depth_${INIT}_${RPE}_s${SEED}_stage2"
  local RPE_F; RPE_F=$(rpe_flags "$RPE")
  local CKPT; CKPT=$(init_ckpt "$INIT")

  if [ -f "${S2}/results.pth" ]; then
    echo "[$(date)] SKIP depth ${INIT} ${RPE} s${SEED}" >> "$MAIN_LOG"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_depth.py \
    --pretrained "$CKPT" --config configs/pretrain_v8_large.yaml \
    --output_dir "$S1" \
    --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \
    --freeze --load_decoder \
    --seed $SEED $RPE_F \
    > ".ops/wave_depth_${INIT}_${RPE}_seed${SEED}_s1.log" 2>&1 \
  && CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_depth.py \
    --pretrained "${S1}/best_model.pth" --config configs/pretrain_v8_large.yaml \
    --output_dir "$S2" \
    --epochs 200 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --load_decoder \
    --seed $SEED $RPE_F \
    > ".ops/wave_depth_${INIT}_${RPE}_seed${SEED}_s2.log" 2>&1
}

run_wave() {
  local WAVE=$1
  IFS=':' read -r NAME KIND INIT RPE <<< "$WAVE"
  echo "[$(date)] ========== WAVE $NAME START (kind=$KIND init=$INIT rpe=$RPE) ==========" >> "$MAIN_LOG"

  # Fan out 3 seeds to 3 GPUs in parallel
  if [ "$KIND" = "seg" ]; then
    run_seed_seg "$INIT" "$RPE" 42 0 &
    PID0=$!
    run_seed_seg "$INIT" "$RPE" 123 1 &
    PID1=$!
    run_seed_seg "$INIT" "$RPE" 456 2 &
    PID2=$!
  else
    run_seed_depth "$INIT" "$RPE" 42 0 &
    PID0=$!
    run_seed_depth "$INIT" "$RPE" 123 1 &
    PID1=$!
    run_seed_depth "$INIT" "$RPE" 456 2 &
    PID2=$!
  fi

  wait $PID0; RC0=$?
  wait $PID1; RC1=$?
  wait $PID2; RC2=$?
  echo "[$(date)] WAVE $NAME DONE  rc0=$RC0 rc1=$RC1 rc2=$RC2" >> "$MAIN_LOG"

  # Per-wave PASS/FAIL gate + optional auto-promote (dry-run by default)
  python scripts/compare_vs_target.py --pass-only 2>/dev/null | grep "$KIND.*$INIT.*$RPE" | head -3 >> "$MAIN_LOG" || true
}

FILTER="${1:-}"
WAVE_FROM="${WAVE_FROM:-1}"

echo "[$(date)] ===== run_waves_serial.sh started =====" >> "$MAIN_LOG"
echo "  Filter: ${FILTER:-all}" >> "$MAIN_LOG"
echo "  Starting from wave: $WAVE_FROM" >> "$MAIN_LOG"

idx=0
for wave in "${WAVES[@]}"; do
  idx=$((idx + 1))
  [ "$idx" -lt "$WAVE_FROM" ] && continue
  if [ -n "$FILTER" ]; then
    [[ "$wave" == "$FILTER:"* ]] || continue
  fi
  run_wave "$wave"
done

echo "[$(date)] ===== all waves complete =====" >> "$MAIN_LOG"
