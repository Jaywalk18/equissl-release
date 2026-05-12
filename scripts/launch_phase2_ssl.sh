#!/bin/bash
# Phase-2 launcher for tab:ssl 48-run sweep (24 seg × 1%-label + 24 depth × 100%).
# Companion to:
#   - paper spec   : /mnt/ssd/siggraph-asia-2026-gerpe/data/ssl_tbd_rerun.md
#   - local execution plan : .ops/phase2_experiment_plan.md (gitignored)
#
# This script does NOT auto-launch; it prints the exact job list.
# Pipe into scripts/gpu_queue.sh or a cron/tmux runner to execute.
#
# Usage:
#   bash scripts/launch_phase2_ssl.sh              # print all 48 jobs
#   bash scripts/launch_phase2_ssl.sh seg          # print 24 seg jobs
#   bash scripts/launch_phase2_ssl.sh depth        # print 24 depth jobs
#   bash scripts/launch_phase2_ssl.sh seg rand     # filter by init
#   bash scripts/launch_phase2_ssl.sh depth ibot c6noarea 123
#
# Each printed line is a self-contained shell command. Seeds 42 anchors that
# already exist as symlinks in /mnt/ssd/phase2_staging/seg_* are still re-printed —
# harvest_phase2.py de-dups by reading results.pth (newest-wins).

set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

EXPS="${1:-all}"   # seg | depth | all
FILTER_INIT="${2:-}"   # rand | ibot | ""
FILTER_RPE="${3:-}"    # none | standard | c4 | c6noarea | ""
FILTER_SEED="${4:-}"   # 42 | 123 | 456 | ""

CFG="configs/pretrain_v8_large.yaml"
EMPTY_CKPT="/mnt/ssd/tmp/empty_ckpt.pth"
SSL_CKPT="outputs/pretrain_v8/checkpoint_epoch99.pth"
SEEDS=(42 123 456)
INITS=(rand ibot)
RPES=(none standard c4 c6noarea)

rpe_flags() {
  case "$1" in
    none)       echo "--rpe_mode none" ;;
    standard)   echo "--rpe_mode standard" ;;
    c4)         echo "--rpe_mode equivariant --n_gauges 4" ;;
    c6noarea)   echo "--rpe_mode equivariant --n_gauges 6 --no_area_weight" ;;
    *)          echo "ERROR_UNKNOWN_RPE_$1" ;;
  esac
}

init_ckpt() {
  case "$1" in
    rand) echo "$EMPTY_CKPT" ;;
    ibot) echo "$SSL_CKPT" ;;
  esac
}

emit_seg() {
  local INIT=$1 RPE=$2 SEED=$3
  local OUT_TAG="/mnt/ssd/phase2_staging/seg_${INIT}_${RPE}_s${SEED}"
  local RPE_F; RPE_F=$(rpe_flags "$RPE")
  local CKPT; CKPT=$(init_ckpt "$INIT")

  if [ "$INIT" = "rand" ]; then
    # single-stage 350ep, matches label_eff_* recipe used for seed-42 anchors
    echo "python tools/finetune_seg.py \\
  --pretrained $CKPT --config $CFG --output_dir $OUT_TAG \\
  --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \\
  --loss ce_dice --dice_weight 0.5 \\
  --label_fraction 0.01 --seed $SEED $RPE_F"
  else
    # two-stage: stage1 freeze 50ep + stage2 full 350ep
    local S1="${OUT_TAG}_stage1"
    local S2="${OUT_TAG}_stage2"
    echo "python tools/finetune_seg.py \\
  --pretrained $CKPT --config $CFG --output_dir $S1 \\
  --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \\
  --freeze --load_decoder \\
  --loss ce_dice --dice_weight 0.5 \\
  --label_fraction 0.01 --seed $SEED $RPE_F \\
  && python tools/finetune_seg.py \\
  --pretrained $S1/best_model.pth --config $CFG --output_dir $S2 \\
  --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \\
  --load_decoder \\
  --loss ce_dice --dice_weight 0.5 \\
  --label_fraction 0.01 --seed $SEED $RPE_F"
  fi
}

emit_depth() {
  local INIT=$1 RPE=$2 SEED=$3
  local OUT_TAG="/mnt/ssd/phase2_staging/depth_${INIT}_${RPE}_s${SEED}"
  local RPE_F; RPE_F=$(rpe_flags "$RPE")
  local CKPT; CKPT=$(init_ckpt "$INIT")
  local S1="${OUT_TAG}_stage1"
  local S2="${OUT_TAG}_stage2"
  # Both inits use two-stage depth per paper spec (50ep freeze + 200ep full)
  echo "python tools/finetune_depth.py \\
  --pretrained $CKPT --config $CFG --output_dir $S1 \\
  --epochs 50 --lr 5e-4 --batch_size 8 --num_workers 8 \\
  --freeze --load_decoder \\
  --seed $SEED $RPE_F \\
  && python tools/finetune_depth.py \\
  --pretrained $S1/best_model.pth --config $CFG --output_dir $S2 \\
  --epochs 200 --lr 1e-4 --batch_size 8 --num_workers 8 \\
  --load_decoder \\
  --seed $SEED $RPE_F"
}

emit_all() {
  local KIND=$1
  for INIT in "${INITS[@]}"; do
    [ -n "$FILTER_INIT" ] && [ "$FILTER_INIT" != "$INIT" ] && continue
    for RPE in "${RPES[@]}"; do
      [ -n "$FILTER_RPE" ] && [ "$FILTER_RPE" != "$RPE" ] && continue
      for SEED in "${SEEDS[@]}"; do
        [ -n "$FILTER_SEED" ] && [ "$FILTER_SEED" != "$SEED" ] && continue
        echo "# --- ${KIND} ${INIT} ${RPE} seed=${SEED} ---"
        if [ "$KIND" = "seg" ]; then emit_seg "$INIT" "$RPE" "$SEED"; else emit_depth "$INIT" "$RPE" "$SEED"; fi
        echo ""
      done
    done
  done
}

if [ "$EXPS" = "seg" ] || [ "$EXPS" = "all" ]; then emit_all seg; fi
if [ "$EXPS" = "depth" ] || [ "$EXPS" = "all" ]; then emit_all depth; fi
