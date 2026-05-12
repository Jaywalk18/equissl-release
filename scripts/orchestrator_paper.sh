#!/bin/bash
# Autonomous orchestrator: fill paper-side data gaps with zero hand-holding.
#
# Phase 1: 7 missing finetune seeds (random-init, ce_dice).
#   c6_noarea {123,456}, c2 {123,456}, none {123,456}, standard {456}
# Phase 2: rotation-curve eval (0/10/20/35/45/60/90) on the seed-42 best_model
#   of each of 6 variants. Re-uses existing checkpoints, no retraining.
#
# Queue-driven: 3 workers (one per GPU) pull jobs from a shared queue with
# flock. Skips jobs whose results.pth already exists (idempotent restart).
# Survives container restarts: just relaunch and it picks up where it left.

set -u
cd ${EQUISSL_ROOT}
export PYTHONPATH=${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export TMPDIR=/mnt/ssd/tmp

ORCH_LOG=.ops/orchestrator.log
QUEUE_DIR=.ops
mkdir -p "$QUEUE_DIR"

CONFIG=configs/pretrain_v8_large.yaml
EMPTY_CKPT=/mnt/ssd/tmp/empty_ckpt.pth

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }

# ---------- Phase 1: finetune seg, 7 jobs ----------
P1_QUEUE="$QUEUE_DIR/p1_queue.txt"
P1_LOCK="$QUEUE_DIR/p1.lock"

if [ ! -f "$P1_QUEUE.init" ]; then
  cat > "$P1_QUEUE" <<'EOF'
c6_noarea|123|--rpe_mode equivariant --n_gauges 6 --no_area_weight
c6_noarea|456|--rpe_mode equivariant --n_gauges 6 --no_area_weight
c2|123|--rpe_mode equivariant --n_gauges 2
c2|456|--rpe_mode equivariant --n_gauges 2
none|123|--rpe_mode none
none|456|--rpe_mode none
standard|456|--rpe_mode standard
EOF
  touch "$P1_QUEUE.init"
fi

run_finetune() {
  local GPU=$1 VARIANT=$2 SEED=$3 EXTRA="$4"
  local OUT="outputs/${VARIANT}_seed${SEED}_v9"
  local LOG=".ops/finetune_${VARIANT}_seed${SEED}_v9.log"

  if [ -f "$OUT/results.pth" ]; then
    log "SKIP finetune $VARIANT s${SEED}: results.pth exists"
    return 0
  fi

  mkdir -p "$OUT"
  log "START finetune $VARIANT s${SEED} GPU=$GPU EXTRA=$EXTRA"

  CUDA_VISIBLE_DEVICES=$GPU python3 -u tools/finetune_seg.py \
    --pretrained "$EMPTY_CKPT" \
    --config "$CONFIG" \
    --output_dir "$OUT" \
    --epochs 350 --lr 1e-4 --batch_size 8 --num_workers 8 \
    --loss ce_dice --dice_weight 0.5 \
    --seed "$SEED" $EXTRA \
    > "$LOG" 2>&1
  local EXIT=$?

  if [ -f "$OUT/results.pth" ]; then
    local VAL=$(python3 -c "import torch; r=torch.load('$OUT/results.pth', weights_only=False); print(round(r['best_val_miou']*100,2))" 2>/dev/null)
    log "DONE finetune $VARIANT s${SEED} val=${VAL} (exit=$EXIT)"
  else
    log "FAIL finetune $VARIANT s${SEED} (exit=$EXIT) — see $LOG"
  fi
}

p1_worker() {
  local GPU=$1
  while true; do
    JOB=""
    {
      flock 9
      JOB=$(head -1 "$P1_QUEUE" 2>/dev/null)
      if [ -n "$JOB" ]; then
        sed -i "1d" "$P1_QUEUE"
      fi
    } 9>"$P1_LOCK"
    [ -z "$JOB" ] && break
    IFS='|' read -r VARIANT SEED EXTRA <<< "$JOB"
    run_finetune "$GPU" "$VARIANT" "$SEED" "$EXTRA"
  done
  log "P1 worker GPU=$GPU exiting (queue empty)"
}

# ---------- Phase 2: rotation-curve eval, single seed per variant ----------
P2_QUEUE="$QUEUE_DIR/p2_queue.txt"
P2_LOCK="$QUEUE_DIR/p2.lock"

build_p2_queue() {
  cat > "$P2_QUEUE" <<'EOF'
c4|outputs/rpe_ablation_c4_v2|--rpe_mode equivariant --n_gauges 4
c6area|outputs/rpe_ablation_equivariant_v2|--rpe_mode equivariant --n_gauges 6
c6_noarea|outputs/rpe_ablation_c6_noarea|--rpe_mode equivariant --n_gauges 6 --no_area_weight
c2|outputs/rpe_ablation_c2_v2|--rpe_mode equivariant --n_gauges 2
none|outputs/rpe_ablation_none|--rpe_mode none
standard|outputs/finetune_v8_random_s|--rpe_mode standard
EOF
  touch "$P2_QUEUE.init"
}

run_rotation() {
  local GPU=$1 VARIANT=$2 CKPT_DIR=$3 EXTRA="$4"
  local CKPT="$CKPT_DIR/best_model.pth"
  local DONE_MARK="$CKPT_DIR/rot_curve_v9.done"

  if [ -f "$DONE_MARK" ]; then
    log "SKIP rotation $VARIANT: $DONE_MARK exists"
    return 0
  fi
  if [ ! -f "$CKPT" ]; then
    log "FAIL rotation $VARIANT: no $CKPT"
    return 1
  fi

  log "START rotation $VARIANT GPU=$GPU"
  for ANGLE in 0 10 20 35 45 60 90; do
    local LOG="$CKPT_DIR/rot_v9_angle${ANGLE}_val.log"
    if grep -q "SO(3) mIoU:" "$LOG" 2>/dev/null; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU python3 -u tools/eval_pose35.py \
      --checkpoint "$CKPT" \
      --config "$CONFIG" \
      --max_angle "$ANGLE" \
      --num_rotations 10 \
      --num_repeats 3 \
      --batch_size 4 \
      --num_workers 8 \
      --split val \
      $EXTRA \
      > "$LOG" 2>&1
  done
  touch "$DONE_MARK"
  log "DONE rotation $VARIANT"
}

p2_worker() {
  local GPU=$1
  while true; do
    JOB=""
    {
      flock 9
      JOB=$(head -1 "$P2_QUEUE" 2>/dev/null)
      if [ -n "$JOB" ]; then
        sed -i "1d" "$P2_QUEUE"
      fi
    } 9>"$P2_LOCK"
    [ -z "$JOB" ] && break
    IFS='|' read -r VARIANT CKPT_DIR EXTRA <<< "$JOB"
    run_rotation "$GPU" "$VARIANT" "$CKPT_DIR" "$EXTRA"
  done
  log "P2 worker GPU=$GPU exiting (queue empty)"
}

# ---------- main ----------
log "===== Orchestrator starting ====="
log "Phase 1 queue: $(wc -l < $P1_QUEUE) jobs"

p1_worker 0 &
p1_worker 1 &
p1_worker 2 &
wait
log "Phase 1 finished"

build_p2_queue
log "Phase 2 queue: $(wc -l < $P2_QUEUE) jobs"

p2_worker 0 &
p2_worker 1 &
p2_worker 2 &
wait
log "Phase 2 finished"

log "===== ALL DONE ====="
log "Final summary:"
python3 - <<'PY' 2>&1 | tee -a "$ORCH_LOG"
import torch, os, glob
print("\n3-seed seg results:")
for variant_dirs, name in [
    (sorted(glob.glob("outputs/c4_seed*") + ["outputs/rpe_ablation_c4_v2"]), "C4"),
    (sorted(glob.glob("outputs/c6_seed*") + ["outputs/rpe_ablation_equivariant_v2"]), "C6 area"),
    (sorted(glob.glob("outputs/c6_noarea_seed*_v9") + ["outputs/rpe_ablation_c6_noarea"]), "C6 no-area"),
    (sorted(glob.glob("outputs/c2_seed*_v9") + ["outputs/rpe_ablation_c2_v2"]), "C2"),
    (sorted(glob.glob("outputs/none_seed*_v9") + ["outputs/rpe_ablation_none"]), "No-RPE"),
    (sorted(glob.glob("outputs/standard_seed*") + ["outputs/finetune_v8_random_s"]), "Standard"),
]:
    vals, tests = [], []
    for d in variant_dirs:
        try:
            r = torch.load(f"{d}/results.pth", weights_only=False)
            vals.append(r["best_val_miou"]*100); tests.append(r["test_miou"]*100)
        except: pass
    if vals:
        import statistics
        n = len(vals)
        vm = sum(vals)/n; tm = sum(tests)/n
        vs = statistics.stdev(vals) if n>1 else 0.0
        ts = statistics.stdev(tests) if n>1 else 0.0
        print(f"  {name:<12} n={n} val={vm:.2f}±{vs:.2f} test={tm:.2f}±{ts:.2f}")
PY
