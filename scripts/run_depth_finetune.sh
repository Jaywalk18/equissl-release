#!/bin/bash
# EquiSSL Depth Estimation 2-stage finetuning on Stanford2D3D
#   GPU 0: v3 backbone + depth (2-stage)
#   GPU 1: v8 backbone + depth (2-stage)
# Uses the pretrained encoders (student.* keys) as starting point.
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export TMPDIR=/mnt/ssd/tmp

run_depth_2stage() {
    local GPU=$1 NAME=$2 PRETRAINED=$3 CFG=$4 S1_EPOCHS=$5 S2_EPOCHS=$6
    local S1="outputs/depth_${NAME}_s1" S2="outputs/depth_${NAME}_s2"
    mkdir -p "$S1" "$S2"

    if [ ! -f "$PRETRAINED" ]; then
        echo "[${NAME} | GPU $GPU] ERROR: $PRETRAINED not found"
        return 1
    fi

    echo "[${NAME} | GPU $GPU] Stage1: freeze encoder, ${S1_EPOCHS} epochs"
    CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_depth.py \
        --pretrained "$PRETRAINED" --config "$CFG" \
        --output_dir "$S1" --epochs $S1_EPOCHS --lr 5e-4 --batch_size 8 --num_workers 8 \
        --freeze --load_decoder \
        2>&1 | tee "${S1}/stage1.log"

    echo "[${NAME} | GPU $GPU] Stage2: full finetune, ${S2_EPOCHS} epochs"
    CUDA_VISIBLE_DEVICES=$GPU python tools/finetune_depth.py \
        --pretrained "${S1}/best_model.pth" --config "$CFG" \
        --output_dir "$S2" --epochs $S2_EPOCHS --lr 1e-4 --batch_size 8 --num_workers 8 \
        --load_decoder \
        2>&1 | tee "${S2}/stage2.log"

    echo "[${NAME} | GPU $GPU] Done"
}

echo "============================================"
echo "EquiSSL Depth 2-stage finetuning"
echo "  GPU 0: v3 backbone depth (50+200 epoch)"
echo "  GPU 1: v8 backbone depth (50+200 epoch)"
echo "  Start: $(date)"
echo "============================================"

# GPU 0: v3 backbone
run_depth_2stage 0 "v3" \
    "outputs/pretrain_v3/checkpoint_epoch99.pth" \
    "configs/pretrain.yaml" \
    50 200 &
PID_V3=$!

# GPU 1: v8 backbone
run_depth_2stage 1 "v8" \
    "outputs/pretrain_v8/checkpoint_epoch99.pth" \
    "configs/pretrain_v8_large.yaml" \
    50 200 &
PID_V8=$!

wait $PID_V3 || echo "[WARN] v3 depth exited with error"
wait $PID_V8 || echo "[WARN] v8 depth exited with error"

echo ""
echo "============================================"
echo "=== Depth Results ==="
echo "============================================"
for exp in v3 v8; do
    echo ""
    echo "--- ${exp} ---"
    grep -E "Best val delta1|Test results:|delta1:|delta2:|delta3:|rmse:|mae:|abs_rel:" \
        "outputs/depth_${exp}_s2/stage2.log" 2>/dev/null | tail -10
done

echo ""
echo "All done at $(date)"
