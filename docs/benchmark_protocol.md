# EquiSSL Benchmark Protocol (SIGGRAPH Asia 2026)

## 1. Goal

Reproducible evaluation standard for all EquiSSL experiments. Every result in the paper must follow this protocol.

## 2. Dataset

**Stanford2D3D Semantic Segmentation** on icosphere (13 semantic classes + 1 ignored).

| Split | Samples | Usage |
|-------|---------|-------|
| train | 1,013 | Training only |
| val | 40 | Hyperparameter selection, reported in paper |
| test | 373 | Final evaluation, reported in paper |

Splits follow SphereUFormer's official partition (Area 5 as test).

## 3. Metrics

### 3.1 Base mIoU
- Standard mean Intersection-over-Union across 13 classes
- Class 0 (unknown) is ignored
- Evaluated on the model's best checkpoint (selected by val mIoU)

### 3.2 SO(3) mIoU (rotation evaluation)
- Apply random SO(3) rotations (max angle ±θ_max) to input
- Paper default: **θ_max = 90°**, 10 random rotations per sample, 3 repeats
- Smaller-angle sweeps (e.g., θ_max = 35°) use the same script with `--max_angle 35.0`
- Report mean ± std across repeats
- Historical note: the evaluation script is named `eval_pose35.py` for backward
  compatibility; the script is parametric in `--max_angle` and the paper headline
  uses 90°

### 3.3 Rotation Drop
- `Drop = (Base mIoU - SO(3) mIoU) / Base mIoU × 100%`
- Lower is better; measures rotation robustness

### 3.4 Joint Score (internal use)
- `Joint = val_base_mIoU - 0.5 × val_drop`
- Used for quick experiment comparison, not reported in paper

## 4. Evaluation Commands

```bash
# Rotation evaluation at the paper-default θ_max = 90° (val)
bash scripts/eval_pose35.sh <checkpoint> <gpu_id> val

# Rotation evaluation at the paper-default θ_max = 90° (test)
bash scripts/eval_pose35.sh <checkpoint> <gpu_id> test

# Or directly via the Python entry point with an explicit angle:
python tools/eval_pose35.py \
    --checkpoint <ckpt> --max_angle 90.0 --num_rotations 10 --num_repeats 3
```

## 5. Experiment Naming Convention

```
finetune_{version}_{strategy}/
  ├── stage1.log / finetune.log    # Training log
  ├── stage2.log                   # Stage 2 log (2-stage only)
  ├── best_model.pth               # Best checkpoint (by val mIoU)
  ├── results.pth                  # {best_val_miou, best_epoch, test_miou, per_class_iou}
  ├── pose35_val.log               # Pose35 val results
  ├── pose35_test.log              # Pose35 test results
  └── pose35_results.pth           # {base_miou, so3_miou, drop}
```

## 6. Baseline Alignment Checklist

Before claiming any gap vs SphereUFormer (67.53 val):

- [ ] Same val/test split (Area 5 test)
- [ ] Same 13-class label mapping
- [ ] Same icosphere resolution (img_rank=7)
- [ ] Same eval script (single-scale, no TTA)
- [ ] Confirm SphereUFormer number is from their official code, not paper table

## 7. Rules

1. **No val/test leakage**: Never tune hyperparameters on test set
2. **Single checkpoint**: Report best val checkpoint's test performance
3. **No cherry-picking**: Report all experiments, including negative results (e.g., v6)
4. **Reproducibility**: Every experiment must have a config + log + results.pth
