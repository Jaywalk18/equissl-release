"""
SphereUFormer Pose35 Evaluation: SO(3) rotation robustness.
Loads SphereUFormer model and evaluates with random rotations.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

from network.sphere_model import SphereUFormer
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
        rgb = batch["sphere_rgb"].to(device)  # (B, N, 3) in [0, 1]
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

            # SphereUFormer normalization: (x - 0.5) / 0.225
            rgb_normed = (rgb_input - 0.5) / 0.225
            logits = model(rgb_normed)  # (B, N, C)
            preds = logits.argmax(dim=-1).cpu()  # (B, N)

            for c in range(1, num_classes):
                pred_c = (preds == c)
                label_c = (labels_rot == c)
                intersection[c] += (pred_c & label_c).sum().float()
                union[c] += (pred_c | label_c).sum().float()

    iou = intersection[1:] / (union[1:] + 1e-6)
    return iou.mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="SphereUFormer Pose35 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="${STANFORD2D3D_PATH}")
    parser.add_argument("--max_angle", type=float, default=35.0)
    parser.add_argument("--num_rotations", type=int, default=10)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="test")
    # Model config (match SphereUFormer defaults)
    parser.add_argument("--img_rank", type=int, default=7)
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--scale_depth", type=int, default=2)
    parser.add_argument("--win_size_coef", type=int, default=2)
    parser.add_argument("--d_head_coef", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SphereUFormer Pose35 SO(3) Robustness Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Max angle: {args.max_angle} deg")
    print(f"  Rotations: {args.num_rotations}, Repeats: {args.num_repeats}")
    print("=" * 60)

    num_classes = 14  # Stanford2D3D

    model = SphereUFormer(
        img_rank=args.img_rank,
        node_type="vertex",
        in_channels=3,
        out_channels=num_classes,
        in_scale_factor=args.scale_factor,
        num_scales=args.num_scales,
        win_size_coef=args.win_size_coef,
        enc_depths=args.scale_depth,
        dec_depths=args.scale_depth,
        bottleneck_depth=args.scale_depth,
        d_head_coef=args.d_head_coef,
        enc_num_heads=[2, 4, 8, 16],
        dec_num_heads=[16, 16, 8, 4],
        abs_pos_enc_in=True,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=1.0,
        downsample="center",
        upsample="interpolate",
        use_checkpoint=True,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Handle both state_dict formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.cuda().eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Params: {total_params/1e6:.1f}M")

    icosphere_ref = IcoSphereRef("vertex")
    img_normals = icosphere_ref.get_normals(args.img_rank)
    # SphereUFormer's seg head outputs at img_rank (no downsample),
    # so labels and the projection-side rotation permutation must also be at img_rank.
    proj_normals = img_normals

    # Use EquiSSL's dataset; disable in-dataset normalization (no-op)
    # because evaluate_with_rotation() applies SphereUFormer's (x-0.5)/0.225 itself.
    # Force in_scale_factor=1 so labels are kept at img_rank (matches model output).
    ds_kwargs = dict(
        data_dir=args.data_dir, img_rank=args.img_rank,
        node_type="vertex", num_scales=args.num_scales,
        in_scale_factor=1,
        normalize_mean=0.0,
        normalize_std=1.0,
    )
    test_dataset = Stanford2D3DSeg(split=args.split, **ds_kwargs)
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


if __name__ == "__main__":
    main()
