"""
EquiSSL Pose35 Evaluation: SO(3) rotation robustness on Stanford2D3D.
Uses EquiSSLSegUNet (encoder + decoder) for evaluation.

Usage:
    python tools/eval_pose35.py \
        --checkpoint outputs/finetune_unet_full/best_model.pth
"""

import os
import sys
import argparse
import yaml
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import (
    sample_so3_rotation_bounded,
    compute_rotation_permutation,
    apply_rotation_to_features,
)
from trimesh_utils import IcoSphereRef


@torch.no_grad()
def evaluate_with_rotation(
    model, dataloader, num_classes,
    img_normals, proj_normals,
    max_angle_deg=35.0, num_rotations=10, ignore_index=0,
):
    model.eval()
    device = next(model.parameters()).device

    intersection = torch.zeros(num_classes, device="cpu")
    union = torch.zeros(num_classes, device="cpu")

    for batch in dataloader:
        rgb = batch["sphere_rgb"].to(device)
        labels = batch["sphere_gt_sem"].long()

        n_rot = 1 if max_angle_deg == 0.0 else num_rotations
        for _ in range(n_rot):
            if max_angle_deg > 0.0:
                rot = sample_so3_rotation_bounded(max_angle_deg)
                img_perm = compute_rotation_permutation(img_normals, rot)
                img_perm_t = torch.tensor(img_perm, dtype=torch.long, device=device)
                rgb_input = apply_rotation_to_features(rgb, img_perm_t)

                proj_perm = compute_rotation_permutation(proj_normals, rot)
                proj_perm_t = torch.tensor(proj_perm, dtype=torch.long)
                labels_rot = labels[:, proj_perm_t]
            else:
                rgb_input = rgb
                labels_rot = labels

            logits = model(rgb_input)
            pred = logits.argmax(dim=-1).cpu()

            for c in range(num_classes):
                if c == ignore_index:
                    continue
                pred_c = pred == c
                label_c = labels_rot == c
                intersection[c] += (pred_c & label_c).sum().item()
                union[c] += (pred_c | label_c).sum().item()

    iou = intersection / union.clamp(min=1)
    valid = union > 0
    valid[ignore_index] = False
    return iou[valid].mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="EquiSSL Pose35 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/pretrain_default.yaml")
    parser.add_argument("--data_dir", type=str, default="${STANFORD2D3D_PATH}")
    parser.add_argument("--max_angle", type=float, default=35.0)
    parser.add_argument("--num_rotations", type=int, default=10)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="test", help="val or test")
    parser.add_argument("--rpe_mode", type=str, default="config",
                        choices=["config", "none", "standard", "equivariant"],
                        help="RPE mode: config=follow yaml, none=disable RPE, "
                             "standard=plain RPE, equivariant=GE-RPE from config")
    parser.add_argument("--n_gauges", type=int, default=None,
                        help="Override n_gauges for GE-RPE (default: from config, typically 6)")
    parser.add_argument("--no_area_weight", action="store_true",
                        help="Disable area weighting in GE-RPE")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("EquiSSL Pose35 SO(3) Robustness Evaluation (U-Net)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Max angle: {args.max_angle} deg")
    print(f"  Rotations: {args.num_rotations}, Repeats: {args.num_repeats}")
    print("=" * 60)

    model_cfg = cfg["model"]
    n_gauges_val = args.n_gauges or model_cfg.get("n_gauges", 6)
    use_area_weighted = (not args.no_area_weight) and model_cfg.get("area_weighted", True)

    if args.rpe_mode == "none":
        use_rel_pos_bias = False
        use_equivariant_rpe = False
    elif args.rpe_mode == "standard":
        use_rel_pos_bias = True
        use_equivariant_rpe = False
    elif args.rpe_mode == "equivariant":
        use_rel_pos_bias = True
        use_equivariant_rpe = True
    else:
        use_rel_pos_bias = model_cfg["rel_pos_bias"]
        use_equivariant_rpe = model_cfg.get("equivariant_rpe", False)
    print(f"  RPE mode: {args.rpe_mode} (rel_pos_bias={use_rel_pos_bias}, equivariant={use_equivariant_rpe}, n_gauges={n_gauges_val})")

    encoder = SphericalEncoder(
        img_rank=model_cfg["img_rank"], node_type=model_cfg["node_type"],
        embed_dim=model_cfg["embed_dim"], num_scales=model_cfg["num_scales"],
        in_scale_factor=model_cfg["in_scale_factor"], enc_depths=model_cfg["enc_depths"],
        bottleneck_depth=model_cfg["bottleneck_depth"], enc_num_heads=model_cfg["enc_num_heads"],
        d_head_coef=model_cfg["d_head_coef"], win_size_coef=model_cfg["win_size_coef"],
        mlp_ratio=model_cfg["mlp_ratio"], qkv_bias=model_cfg["qkv_bias"],
        drop_path_rate=0.0,
        abs_pos_enc_in=model_cfg["abs_pos_enc_in"], abs_pos_enc=model_cfg["abs_pos_enc"],
        rel_pos_bias=use_rel_pos_bias,
        rel_pos_bias_size=model_cfg.get("rel_pos_bias_size", 7),
        equivariant_rpe=use_equivariant_rpe,
        n_gauges=n_gauges_val,
        area_weighted=use_area_weighted,
    )

    num_classes = Stanford2D3DSeg.NUM_CLASSES
    model = EquiSSLSegUNet(
        encoder=encoder, num_classes=num_classes,
        dec_depths=tuple(model_cfg.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(model_cfg.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=model_cfg["d_head_coef"], win_size_coef=model_cfg["win_size_coef"],
        mlp_ratio=model_cfg["mlp_ratio"], qkv_bias=model_cfg["qkv_bias"],
        abs_pos_enc=model_cfg["abs_pos_enc"], rel_pos_bias=use_rel_pos_bias,
        rel_pos_bias_size=model_cfg.get("rel_pos_bias_size", 7),
        equivariant_rpe=use_equivariant_rpe,
        n_gauges=n_gauges_val,
        area_weighted=use_area_weighted,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().eval()
    print("Model loaded.")

    icosphere_ref = IcoSphereRef(model_cfg["node_type"])
    img_normals = icosphere_ref.get_normals(model_cfg["img_rank"])
    proj_rank = model_cfg["img_rank"] - 1 if model_cfg.get("in_scale_factor", 2) == 2 else model_cfg["img_rank"]
    proj_normals = icosphere_ref.get_normals(proj_rank)

    ds_kwargs = dict(
        data_dir=args.data_dir, img_rank=model_cfg["img_rank"],
        node_type=model_cfg["node_type"], num_scales=model_cfg["num_scales"],
        in_scale_factor=model_cfg["in_scale_factor"],
        normalize_mean=cfg["data"]["normalize_mean"],
        normalize_std=cfg["data"]["normalize_std"],
    )
    eval_split = args.split
    test_dataset = Stanford2D3DSeg(split=eval_split, **ds_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"\nBase evaluation (no rotation)...")
    base_miou = evaluate_with_rotation(
        model, test_loader, num_classes, img_normals, proj_normals,
        max_angle_deg=0.0, num_rotations=1,
    )
    print(f"  Base mIoU: {base_miou:.4f}")

    print(f"\nPose35 evaluation (max {args.max_angle} deg)...")
    mious = []
    for rep in range(args.num_repeats):
        t0 = time.time()
        miou = evaluate_with_rotation(
            model, test_loader, num_classes, img_normals, proj_normals,
            max_angle_deg=args.max_angle, num_rotations=args.num_rotations,
        )
        mious.append(miou)
        print(f"  Repeat {rep+1}/{args.num_repeats}: SO(3) mIoU={miou:.4f} ({time.time()-t0:.0f}s)")

    mean_miou = np.mean(mious)
    std_miou = np.std(mious)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Base mIoU:  {base_miou:.4f}")
    print(f"  SO(3) mIoU: {mean_miou:.4f} +/- {std_miou:.4f}")
    print(f"  Drop:       {base_miou - mean_miou:.4f} ({(base_miou - mean_miou)/max(base_miou,1e-6)*100:.1f}%)")
    print(f"{'='*60}")

    out_dir = os.path.dirname(args.checkpoint)
    results = {
        "base_miou": base_miou, "so3_miou_mean": mean_miou,
        "so3_miou_std": std_miou, "so3_mious": mious,
        "max_angle_deg": args.max_angle,
    }
    torch.save(results, os.path.join(out_dir, "pose35_results.pth"))
    print(f"Saved to {out_dir}/pose35_results.pth")


if __name__ == "__main__":
    main()
