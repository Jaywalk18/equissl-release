#!/usr/bin/env bash
# Auto-pipeline: wait for pretrain_v9 → forensic → if PASS, fire 3-seed finetune.
# Run this in tmux: tmux new -d -s autopipe 'bash scripts/auto_pipeline_v9.sh 2>&1 | tee .ops/autopipe_v9.log'

set -u
cd ${EQUISSL_ROOT}
export PYTHONPATH=${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}

CKPT=outputs/pretrain_v9/checkpoint_epoch99.pth
CFG=configs/pretrain_v9_repaired.yaml
FORENSIC_JSON=.ops/forensic_v9.json
STAGING=/mnt/ssd/phase2_staging

echo "[$(date)] auto_pipeline_v9 started — waiting for $CKPT"
until [ -f "$CKPT" ]; do
  sleep 60
done
echo "[$(date)] $CKPT exists. Waiting 30s for write to settle..."
sleep 30

echo "[$(date)] Running forensic check..."
CUDA_VISIBLE_DEVICES=0 python3 scripts/forensic_v9.py \
  --ckpt "$CKPT" --config "$CFG" --out_json "$FORENSIC_JSON" \
  --min_eff_rank 30 --max_avg_cos 0.85
RESULT=$?

if [ $RESULT -ne 0 ]; then
  echo "[$(date)] FORENSIC HARD-FAIL — full collapse (eff_rank < 30 OR cos > 0.85 like v8). Aborting finetune. See $FORENSIC_JSON."
  exit 1
fi

echo "[$(date)] FORENSIC PASSED — launching 3-seed finetune in parallel."
mkdir -p "$STAGING"

for SEED in 42 123 456; do
  GPU=$(( (SEED == 42) * 0 + (SEED == 123) * 1 + (SEED == 456) * 2 ))
  OUT=$STAGING/v9_c6area_s${SEED}
  LOG=.ops/v9_finetune_s${SEED}.log
  echo "[$(date)] GPU $GPU seed $SEED → $OUT"
  CUDA_VISIBLE_DEVICES=$GPU nohup python3 tools/finetune_seg.py \
    --pretrained "$CKPT" --config "$CFG" \
    --output_dir "$OUT" \
    --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --load_decoder --loss ce_dice --dice_weight 0.5 \
    --rpe_mode equivariant --n_gauges 6 --seed $SEED \
    > "$LOG" 2>&1 &
done

wait
echo "[$(date)] All 3 finetune sessions complete. Check $STAGING/v9_c6area_s{42,123,456}/results.pth"
