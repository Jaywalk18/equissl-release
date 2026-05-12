"""
TTA + Ensemble evaluation for EquiSSL segmentation.

Loads multiple (config, checkpoint) pairs, runs yaw-rotation TTA on each,
and averages logits for the final ensemble prediction.

Yaw TTA is chosen because ERP yaw rotation is equivariant (horizontal roll),
and we can implement it as a deterministic permutation on the icosphere.

Usage:
    python tools/eval_tta_ensemble.py \
        --model configs/pretrain.yaml:outputs/finetune_exp_g_tuned_s2/best_model.pth \
        --model configs/pretrain_v8_large.yaml:outputs/finetune_v8_g_s2/best_model.pth \
        --model configs/pretrain_v8_large.yaml:outputs/finetune_v8_f_s2/best_model.pth \
        --tta_yaws 0,90,180,270 --split val
"""

import os
import sys
import argparse
import yaml
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation
from trimesh_utils import IcoSphereRef


def load_model(config_path, ckpt_path, device):
    """Load an EquiSSLSegUNet from config + checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    encoder = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=mc["rel_pos_bias"],
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    )

    num_classes = Stanford2D3DSeg.NUM_CLASSES
    model = EquiSSLSegUNet(
        encoder=encoder, num_classes=num_classes,
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=mc["rel_pos_bias"],
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, cfg


def build_yaw_permutations(yaws_deg, img_normals, proj_normals, device):
    """Precompute (img_perm_fwd, proj_perm_bwd) for each yaw in degrees.

    - img_perm_fwd: forward rotation permutation, applied to input RGB
        rgb_rot[i] = rgb[img_perm_fwd[i]]
    - proj_perm_bwd: backward rotation permutation, applied to logits at proj_rank
        logits_orig[i] = logits_rot[proj_perm_bwd[i]]

    Using two independent KD-tree queries (forward by R and backward by R.T)
    avoids the non-bijection artifacts of `inverse_permutation(forward_perm)`
    on nearest-neighbor rotation permutations.
    """
    perms = {}
    for yaw in yaws_deg:
        if yaw == 0:
            # Identity — use arange to avoid KD-tree noise
            img_perm_fwd = np.arange(len(img_normals))
            proj_perm_bwd = np.arange(len(proj_normals))
        else:
            R = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            R_inv = R.T  # R is orthogonal so R^-1 = R.T
            img_perm_fwd = compute_rotation_permutation(img_normals, R)
            proj_perm_bwd = compute_rotation_permutation(proj_normals, R_inv)
        perms[yaw] = (
            torch.tensor(img_perm_fwd, dtype=torch.long, device=device),
            torch.tensor(proj_perm_bwd, dtype=torch.long, device=device),
        )
    return perms


@torch.no_grad()
def evaluate_tta_ensemble(models, dataloader, num_classes, yaw_perms, device,
                          ignore_index=0):
    """Run TTA + ensemble: accumulate logits across all models and all yaw views,
    averaged and argmax-ed at the end.
    """
    intersection = torch.zeros(num_classes, device="cpu")
    union = torch.zeros(num_classes, device="cpu")
    n_views = len(models) * len(yaw_perms)

    t_start = time.time()
    n_batches = 0
    for batch in dataloader:
        rgb = batch["sphere_rgb"].to(device)           # (B, N_img, 3)
        labels = batch["sphere_gt_sem"].long()         # (B, N_proj)

        acc_logits = None
        for model in models:
            for yaw, (img_perm_fwd_t, proj_perm_bwd_t) in yaw_perms.items():
                # Rotate input by yaw
                rgb_rot = rgb[:, img_perm_fwd_t]
                logits_rot = model(rgb_rot)             # (B, N_proj, C)
                # Un-rotate logits back to original frame via backward permutation
                logits_unrot = logits_rot[:, proj_perm_bwd_t].float()

                if acc_logits is None:
                    acc_logits = logits_unrot
                else:
                    acc_logits = acc_logits + logits_unrot

        acc_logits = acc_logits / n_views
        preds = acc_logits.argmax(dim=-1).cpu()

        for c in range(num_classes):
            if c == ignore_index:
                continue
            pred_c = preds == c
            label_c = labels == c
            intersection[c] += (pred_c & label_c).sum().item()
            union[c] += (pred_c | label_c).sum().item()

        n_batches += 1

    iou = intersection / union.clamp(min=1)
    valid = union > 0
    valid[ignore_index] = False
    miou = iou[valid].mean().item()
    per_class = iou.tolist()
    elapsed = time.time() - t_start
    return miou, per_class, elapsed, n_batches


@torch.no_grad()
def evaluate_single_tta(model, dataloader, num_classes, yaw_perms, device,
                        ignore_index=0):
    """Single-model TTA evaluation (for per-model ablation table)."""
    return evaluate_tta_ensemble([model], dataloader, num_classes, yaw_perms,
                                  device, ignore_index=ignore_index)


def parse_args():
    p = argparse.ArgumentParser(description="EquiSSL TTA + Ensemble Eval")
    p.add_argument("--model", action="append", required=True,
                   help="Format: CONFIG:CHECKPOINT. Can repeat for multi-model ensemble.")
    p.add_argument("--data_dir", type=str, default="${STANFORD2D3D_PATH}")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tta_yaws", type=str, default="0,90,180,270",
                   help="Comma-separated yaw angles for TTA. Use '0' for no-TTA baseline.")
    p.add_argument("--output", type=str, default="outputs/ensemble_tta/results.txt")
    p.add_argument("--report_per_model", action="store_true",
                   help="Also report each individual model's TTA result for ablation.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse models
    specs = []
    for m in args.model:
        if ":" not in m:
            raise ValueError(f"Bad --model spec: {m}, expected CONFIG:CHECKPOINT")
        cfg_path, ckpt_path = m.split(":", 1)
        specs.append((cfg_path, ckpt_path))

    yaws = [int(x) for x in args.tta_yaws.split(",") if x.strip()]

    print("=" * 60)
    print("EquiSSL TTA + Ensemble Evaluation")
    print(f"  Split: {args.split}")
    print(f"  TTA yaws: {yaws}")
    print(f"  Models ({len(specs)}):")
    for i, (c, k) in enumerate(specs):
        print(f"    [{i}] {os.path.basename(c)} + {k}")
    print("=" * 60)

    # Load all models
    models = []
    ref_cfg = None
    for cfg_path, ckpt_path in specs:
        print(f"Loading {ckpt_path} ...")
        model, cfg = load_model(cfg_path, ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  OK: {n_params:.1f}M params")
        models.append(model)
        if ref_cfg is None:
            ref_cfg = cfg

    # Build dataset (use first model's config for data/normalize settings —
    # all our models share the same input spec).
    ref_mc = ref_cfg["model"]
    ds_kwargs = dict(
        data_dir=args.data_dir, img_rank=ref_mc["img_rank"],
        node_type=ref_mc["node_type"], num_scales=ref_mc["num_scales"],
        in_scale_factor=ref_mc["in_scale_factor"],
        normalize_mean=ref_cfg["data"]["normalize_mean"],
        normalize_std=ref_cfg["data"]["normalize_std"],
    )
    dataset = Stanford2D3DSeg(split=args.split, **ds_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    num_classes = Stanford2D3DSeg.NUM_CLASSES

    # Build TTA permutations
    icosphere_ref = IcoSphereRef(ref_mc["node_type"])
    img_normals = icosphere_ref.get_normals(ref_mc["img_rank"])
    proj_rank = ref_mc["img_rank"] - 1 if ref_mc.get("in_scale_factor", 2) == 2 else ref_mc["img_rank"]
    proj_normals = icosphere_ref.get_normals(proj_rank)
    yaw_perms = build_yaw_permutations(yaws, img_normals, proj_normals, device)
    print(f"Built {len(yaw_perms)} yaw permutations.")

    results_lines = []
    results_lines.append(f"# EquiSSL TTA+Ensemble Results")
    results_lines.append(f"Split: {args.split}")
    results_lines.append(f"TTA yaws: {yaws}")
    results_lines.append(f"Models: {len(specs)}")
    for i, (c, k) in enumerate(specs):
        results_lines.append(f"  [{i}] {c} + {k}")
    results_lines.append("")

    # Per-model TTA ablation
    if args.report_per_model and len(models) > 1:
        print(f"\n--- Per-model TTA ablation ({args.split}) ---")
        results_lines.append(f"## Per-model TTA ablation ({args.split})")
        for i, (model, (c, k)) in enumerate(zip(models, specs)):
            miou, per_class, elapsed, nb = evaluate_single_tta(
                model, loader, num_classes, yaw_perms, device
            )
            line = f"  [{i}] {os.path.basename(k):<30}  TTA mIoU={miou:.4f}  ({elapsed:.0f}s)"
            print(line)
            results_lines.append(line)
        results_lines.append("")

    # Full ensemble
    print(f"\n--- Full TTA+Ensemble ({args.split}) ---")
    miou, per_class, elapsed, nb = evaluate_tta_ensemble(
        models, loader, num_classes, yaw_perms, device
    )
    print(f"  Ensemble mIoU: {miou:.4f}  ({elapsed:.0f}s, {nb} batches)")
    results_lines.append(f"## Full TTA+Ensemble ({args.split})")
    results_lines.append(f"mIoU: {miou:.4f}")
    results_lines.append(f"Per-class IoU:")
    for c, iou in enumerate(per_class):
        results_lines.append(f"  class {c:2d}: {iou:.4f}")
    results_lines.append("")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a") as f:
        f.write("\n".join(results_lines) + "\n")
    print(f"\nResults saved to {args.output}")

    # Print big banner for mIoU
    print("\n" + "=" * 60)
    print(f"  FINAL {args.split} mIoU: {miou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
