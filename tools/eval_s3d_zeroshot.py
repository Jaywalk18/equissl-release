"""Zero-shot cross-dataset evaluation on Structured3D.

Loads a Stanford2D3D-trained EquiSSLSegUNet checkpoint and evaluates it on
Structured3D panoramas (with labels remapped to the S2D3D 13-class taxonomy).
No fine-tuning — this tests whether the gauge-equivariance inductive bias
transfers across datasets.

Optional: pass --max_angle > 0 to also compute the SO(3) rotation drop on S3D.
"""
import os, sys, argparse, time, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
from torch.utils.data import DataLoader

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.structured3d_seg import Structured3DSeg
from equissl.utils.sphere import (
    sample_so3_rotation_bounded, compute_rotation_permutation,
    apply_rotation_to_features,
)
from trimesh_utils import IcoSphereRef


@torch.no_grad()
def eval_miou(model, loader, num_classes, img_normals, proj_normals,
              max_angle_deg=0.0, num_rotations=10, ignore_index=0):
    model.eval()
    device = next(model.parameters()).device

    inter = torch.zeros(num_classes, device="cpu")
    union = torch.zeros(num_classes, device="cpu")
    for batch in loader:
        rgb = batch["sphere_rgb"].to(device)
        labels = batch["sphere_gt_sem"].long()

        n_rot = 1 if max_angle_deg == 0.0 else num_rotations
        for _ in range(n_rot):
            if max_angle_deg > 0.0:
                rot = sample_so3_rotation_bounded(max_angle_deg)
                img_perm = torch.tensor(compute_rotation_permutation(img_normals, rot),
                                         dtype=torch.long, device=device)
                rgb_input = apply_rotation_to_features(rgb, img_perm)
                proj_perm = torch.tensor(compute_rotation_permutation(proj_normals, rot),
                                          dtype=torch.long)
                labels_rot = labels[:, proj_perm]
            else:
                rgb_input = rgb
                labels_rot = labels

            logits = model(rgb_input)
            pred = logits.argmax(dim=-1).cpu()
            for c in range(num_classes):
                if c == ignore_index:
                    continue
                pc = pred == c
                lc = labels_rot == c
                inter[c] += (pc & lc).sum().item()
                union[c] += (pc | lc).sum().item()

    iou = inter / union.clamp(min=1)
    valid = union > 0
    valid[ignore_index] = False
    return iou[valid].mean().item(), iou, valid


def build_model(ckpt_path, cfg, rpe_mode, n_gauges, area_weighted, num_classes=14):
    mc = cfg["model"]
    if rpe_mode == "none":
        rp, eq = False, False
    elif rpe_mode == "standard":
        rp, eq = True, False
    else:  # equivariant
        rp, eq = True, True

    encoder = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=rp, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=area_weighted,
    )
    model = EquiSSLSegUNet(
        encoder=encoder, num_classes=num_classes,
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=area_weighted,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/pretrain_v8_large.yaml")
    ap.add_argument("--rpe_mode", choices=["none", "standard", "equivariant"], required=True)
    ap.add_argument("--n_gauges", type=int, default=6)
    ap.add_argument("--no_area_weight", action="store_true")
    ap.add_argument("--data_dir", default="${STRUCTURED3D_PATH}_new/Structured3D")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_angle", type=float, default=0.0)
    ap.add_argument("--num_rotations", type=int, default=10)
    ap.add_argument("--num_repeats", type=int, default=1)
    ap.add_argument("--out_tag", default=None,
                    help="Tag for output filename (default: derived from rpe_mode)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    area_w = (not args.no_area_weight) and mc.get("area_weighted", True)

    print("=" * 60)
    print(f"S3D zero-shot eval")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  RPE mode:   {args.rpe_mode} (n_gauges={args.n_gauges}, area_w={area_w})")
    print(f"  Split:      {args.split} (max_samples={args.max_samples})")
    print(f"  Max angle:  {args.max_angle}°")
    print("=" * 60)

    model = build_model(args.checkpoint, cfg, args.rpe_mode, args.n_gauges, area_w)
    print(f"Model loaded.")

    ds = Structured3DSeg(
        data_dir=args.data_dir, split=args.split,
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
        normalize_mean=cfg["data"]["normalize_mean"],
        normalize_std=cfg["data"]["normalize_std"],
        max_samples=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    ref = IcoSphereRef(mc["node_type"])
    img_normals = ref.get_normals(mc["img_rank"])
    proj_rank = mc["img_rank"] - 1 if mc.get("in_scale_factor", 2) == 2 else mc["img_rank"]
    proj_normals = ref.get_normals(proj_rank)

    # Base (no rotation)
    t0 = time.time()
    base_miou, iou, valid = eval_miou(model, loader, ds.NUM_CLASSES,
                                       img_normals, proj_normals, max_angle_deg=0.0)
    CLS = ["unknown","beam","board","bookcase","ceiling","chair","clutter",
           "column","door","floor","sofa","table","wall","window"]
    print(f"\nBase mIoU (S3D {args.split}, {len(ds)} panoramas): {base_miou:.4f}  ({time.time()-t0:.0f}s)")
    print("Per-class IoU:")
    for c in range(ds.NUM_CLASSES):
        mark = "✓" if valid[c] else " "
        print(f"  {mark} {c:2d} {CLS[c]:<10}: {iou[c]:.4f}  (valid={valid[c].item()})")

    results = {"base_miou": base_miou, "per_class_iou": iou.tolist(),
               "valid_classes": valid.tolist(), "n_samples": len(ds)}

    if args.max_angle > 0:
        mious = []
        for rep in range(args.num_repeats):
            t = time.time()
            m, _, _ = eval_miou(model, loader, ds.NUM_CLASSES, img_normals, proj_normals,
                                max_angle_deg=args.max_angle, num_rotations=args.num_rotations)
            mious.append(m)
            print(f"  Repeat {rep+1}: SO(3) mIoU={m:.4f} ({time.time()-t:.0f}s)")
        mean, std = float(np.mean(mious)), float(np.std(mious))
        print(f"\nSO(3) mIoU: {mean:.4f} ± {std:.4f}")
        print(f"Drop: {base_miou - mean:.4f} ({(base_miou - mean)/max(base_miou,1e-6)*100:.1f}%)")
        results["so3_miou_mean"] = mean
        results["so3_miou_std"] = std
        results["so3_mious"] = mious
        results["max_angle_deg"] = args.max_angle

    tag = args.out_tag or args.rpe_mode
    out_dir = os.path.dirname(args.checkpoint)
    out = os.path.join(out_dir, f"s3d_zeroshot_{args.split}_{tag}.pth")
    torch.save(results, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
