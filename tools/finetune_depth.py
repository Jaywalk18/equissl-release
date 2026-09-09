"""
EquiSSL Downstream Finetuning: Depth Estimation with U-Net decoder.

Predicts per-vertex metric depth (meters) on Stanford2D3D icosphere,
using the same encoder+decoder architecture as segmentation finetuning
but with a 1-channel regression head.

Usage:
    python tools/finetune_depth.py \
        --pretrained outputs/pretrain_v3/checkpoint_epoch99.pth \
        --config configs/pretrain.yaml \
        --output_dir outputs/finetune_depth_v3_s1 \
        --epochs 50 --lr 5e-4 --batch_size 8 \
        --freeze --load_decoder
"""

import os
import sys
import argparse
import yaml
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_depth import Stanford2D3DDepth
from equissl.utils.training import save_checkpoint


MIN_DEPTH = 0.1   # meters — eval clamp to avoid div-by-zero on invalid regions
MAX_DEPTH = 5.12  # meters — matches Stanford2D3DDepth.MAX_DEPTH_M and SphereUFormer


def masked_l1_loss(pred, target, mask):
    """L1 loss on valid pixels only.

    Args:
        pred: (B, N)
        target: (B, N)
        mask: (B, N) bool
    """
    diff = (pred - target).abs()
    diff = diff * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_valid = 0
    total_abs_err = 0.0

    for step, batch in enumerate(dataloader):
        rgb = batch["sphere_rgb"].cuda(non_blocking=True)
        target = batch["sphere_gt_depth"].cuda(non_blocking=True)           # (B, N)
        mask = batch["sphere_valid_mask"].cuda(non_blocking=True)            # (B, N) bool

        pred = model(rgb).squeeze(-1)  # (B, N, 1) -> (B, N)
        loss = masked_l1_loss(pred, target, mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            diff = (pred - target).abs() * mask.float()
            total_abs_err += diff.sum().item()
            total_valid += mask.float().sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    mae = total_abs_err / max(total_valid, 1)
    return avg_loss, mae


@torch.no_grad()
def evaluate(model, dataloader):
    """Return dict of depth metrics: delta1/2/3, rmse, mae, abs_rel."""
    model.eval()

    n_valid = 0
    sum_d1 = 0.0
    sum_d2 = 0.0
    sum_d3 = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    sum_abs_rel = 0.0

    for batch in dataloader:
        rgb = batch["sphere_rgb"].cuda(non_blocking=True)
        target = batch["sphere_gt_depth"].cuda(non_blocking=True)
        mask = batch["sphere_valid_mask"].cuda(non_blocking=True)

        pred = model(rgb).squeeze(-1)
        pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
        target = target.clamp(MIN_DEPTH, MAX_DEPTH)

        mask_f = mask.float()

        thresh = torch.max(target / pred, pred / target)
        sum_d1 += ((thresh < 1.25).float() * mask_f).sum().item()
        sum_d2 += ((thresh < 1.25 ** 2).float() * mask_f).sum().item()
        sum_d3 += ((thresh < 1.25 ** 3).float() * mask_f).sum().item()

        diff = (pred - target)
        sum_sq += ((diff ** 2) * mask_f).sum().item()
        sum_abs += (diff.abs() * mask_f).sum().item()
        sum_abs_rel += ((diff.abs() / target) * mask_f).sum().item()

        n_valid += mask_f.sum().item()

    n = max(n_valid, 1)
    return {
        "delta1": sum_d1 / n,
        "delta2": sum_d2 / n,
        "delta3": sum_d3 / n,
        "rmse": (sum_sq / n) ** 0.5,
        "mae": sum_abs / n,
        "abs_rel": sum_abs_rel / n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="EquiSSL Depth Estimation Finetuning")
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--data_dir", type=str, default="${STANFORD2D3D_PATH}")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune_depth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze", action="store_true", help="freeze encoder (linear probe)")
    parser.add_argument("--load_decoder", action="store_true", help="load pretrained decoder weights")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--layer_decay", type=float, default=None,
                        help="layer-wise lr decay rate (e.g. 0.65). None = uniform lr")
    parser.add_argument("--rpe_mode", type=str, default="config",
                        choices=["config", "none", "standard", "equivariant"],
                        help="RPE mode: config=follow yaml, none=disable RPE, "
                             "standard=plain RPE, equivariant=GE-RPE from config")
    parser.add_argument("--n_gauges", type=int, default=None,
                        help="Override n_gauges for GE-RPE (default: from config, typically 6)")
    parser.add_argument("--no_area_weight", action="store_true",
                        help="Disable area-weighted attention bias in GE-RPE")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _build_layer_decay_params(model, base_lr, layer_decay):
    """Same as seg finetune — deeper layers get higher lr."""
    param_groups = {}
    num_enc_scales = len(model.encoder.enc_blocks) if hasattr(model.encoder, 'enc_blocks') else 4
    num_layers = num_enc_scales + 1
    max_layer_id = num_layers

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "enc_blocks" in name:
            layer_id = int(name.split("enc_blocks.")[1].split(".")[0])
        elif "bottleneck" in name:
            layer_id = num_enc_scales
        else:
            layer_id = max_layer_id

        scale = layer_decay ** (max_layer_id - layer_id)
        group_key = f"layer_{layer_id}_scale_{scale:.4f}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [], "lr": base_lr * scale, "weight_decay": 0.01,
            }
        param_groups[group_key]["params"].append(param)

    groups = list(param_groups.values())
    for g in groups:
        print(f"  lr={g['lr']:.6f}, params={len(g['params'])}")
    return groups


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("EquiSSL U-Net Depth Finetuning")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Freeze encoder: {args.freeze}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}, BS: {args.batch_size}")
    print("=" * 60)

    model_cfg = cfg["model"]

    # Resolve RPE settings based on --rpe_mode
    if args.rpe_mode == "none":
        use_rel_pos_bias = False
        use_equivariant_rpe = False
    elif args.rpe_mode == "standard":
        use_rel_pos_bias = True
        use_equivariant_rpe = False
    elif args.rpe_mode == "equivariant":
        use_rel_pos_bias = True
        use_equivariant_rpe = True
    else:  # "config" — follow yaml
        use_rel_pos_bias = model_cfg["rel_pos_bias"]
        use_equivariant_rpe = model_cfg.get("equivariant_rpe", False)

    n_gauges_val = args.n_gauges or model_cfg.get("n_gauges", 6)
    use_area_weighted = (not args.no_area_weight) and model_cfg.get("area_weighted", True)

    encoder = SphericalEncoder(
        img_rank=model_cfg["img_rank"], node_type=model_cfg["node_type"],
        embed_dim=model_cfg["embed_dim"], num_scales=model_cfg["num_scales"],
        in_scale_factor=model_cfg["in_scale_factor"], enc_depths=model_cfg["enc_depths"],
        bottleneck_depth=model_cfg["bottleneck_depth"], enc_num_heads=model_cfg["enc_num_heads"],
        d_head_coef=model_cfg["d_head_coef"], win_size_coef=model_cfg["win_size_coef"],
        mlp_ratio=model_cfg["mlp_ratio"], qkv_bias=model_cfg["qkv_bias"],
        drop_path_rate=model_cfg["drop_path_rate"],
        abs_pos_enc_in=model_cfg["abs_pos_enc_in"], abs_pos_enc=model_cfg["abs_pos_enc"],
        rel_pos_bias=use_rel_pos_bias,
        rel_pos_bias_size=model_cfg.get("rel_pos_bias_size", 7),
        equivariant_rpe=use_equivariant_rpe,
        n_gauges=n_gauges_val,
        area_weighted=use_area_weighted,
    )
    print(f"  RPE mode: {args.rpe_mode} (rel_pos_bias={use_rel_pos_bias}, equivariant={use_equivariant_rpe}, n_gauges={n_gauges_val}, area_weighted={use_area_weighted})")

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Handle both pretrain ckpts (student.xxx) and seg-finetune ckpts (encoder.xxx).
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("student."):
            encoder_state[k.replace("student.", "")] = v
        elif k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "")] = v
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded encoder: {len(encoder_state)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    # Reuse EquiSSLSegUNet with num_classes=1 for depth regression.
    num_classes = 1
    model = EquiSSLSegUNet(
        encoder=encoder,
        num_classes=num_classes,
        dec_depths=tuple(model_cfg.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(model_cfg.get("dec_num_heads", [16, 16, 8, 4])),
        freeze_encoder=args.freeze,
        d_head_coef=model_cfg["d_head_coef"],
        win_size_coef=model_cfg["win_size_coef"],
        mlp_ratio=model_cfg["mlp_ratio"],
        qkv_bias=model_cfg["qkv_bias"],
        abs_pos_enc=model_cfg["abs_pos_enc"],
        rel_pos_bias=use_rel_pos_bias,
        rel_pos_bias_size=model_cfg.get("rel_pos_bias_size", 7),
        equivariant_rpe=use_equivariant_rpe,
        n_gauges=n_gauges_val,
        area_weighted=use_area_weighted,
    )

    if args.load_decoder:
        decoder_state = {}
        for k, v in state.items():
            if k.startswith("decoder."):
                decoder_state[k] = v
        if decoder_state:
            dec_missing, dec_unexpected = model.decoder.load_state_dict(
                {k.replace("decoder.", ""): v for k, v in decoder_state.items()},
                strict=False,
            )
            print(f"Loaded pretrained decoder: {len(decoder_state)} keys, {len(dec_missing)} missing")
        else:
            print("No decoder weights found in checkpoint, using random init")

    # Restore depth regression head (seg_head) from stage-1 checkpoint.
    seg_head_state = {
        k.replace("seg_head.", "", 1): v
        for k, v in state.items()
        if k.startswith("seg_head.")
    }
    if seg_head_state:
        try:
            sh_missing, sh_unexpected = model.seg_head.load_state_dict(seg_head_state, strict=False)
            print(f"Loaded depth head: {len(seg_head_state)} keys, {len(sh_missing)} missing")
        except RuntimeError as e:
            print(f"Depth head shape mismatch (expected for cross-task), using random init: {e}")

    model = model.cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    ds_kwargs = dict(
        data_dir=args.data_dir,
        img_rank=model_cfg["img_rank"],
        node_type=model_cfg["node_type"],
        num_scales=model_cfg["num_scales"],
        in_scale_factor=model_cfg["in_scale_factor"],
        normalize_mean=cfg["data"]["normalize_mean"],
        normalize_std=cfg["data"]["normalize_std"],
    )

    train_dataset = Stanford2D3DDepth(split="train", label_fraction=args.label_fraction, augment=True, **ds_kwargs)
    val_dataset = Stanford2D3DDepth(split="val", **ds_kwargs)
    test_dataset = Stanford2D3DDepth(split="test", **ds_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("Loss: masked L1 (in meters)")

    if args.layer_decay is not None and not args.freeze:
        param_groups = _build_layer_decay_params(model, args.lr, args.layer_decay)
        print(f"Optimizer: AdamW with layer_decay={args.layer_decay}, {len(param_groups)} groups")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_delta1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, epoch)
        scheduler.step()

        m = evaluate(model, val_loader)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs-1} ({elapsed:.0f}s) "
              f"loss={train_loss:.4f} train_mae={train_mae:.4f} "
              f"val_d1={m['delta1']:.4f} val_rmse={m['rmse']:.4f} "
              f"val_abs_rel={m['abs_rel']:.4f} lr={lr_now:.6f}")

        if m["delta1"] > best_delta1:
            best_delta1 = m["delta1"]
            best_epoch = epoch
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pth"),
                epoch, model, optimizer, best_metric=best_delta1,
            )
            print(f"  -> New best! delta1={best_delta1:.4f} "
                  f"(d2={m['delta2']:.4f} d3={m['delta3']:.4f} "
                  f"rmse={m['rmse']:.4f} mae={m['mae']:.4f} abs_rel={m['abs_rel']:.4f})")

        save_checkpoint(
            os.path.join(args.output_dir, "checkpoint_latest.pth"),
            epoch, model, optimizer, best_metric=m["delta1"],
        )

    print(f"\nBest val delta1: {best_delta1:.4f} at epoch {best_epoch}")

    print("\nEvaluating best model on test set...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model = model.cuda()

    test_m = evaluate(model, test_loader)
    print(f"Test results:")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "best_val_delta1": best_delta1,
        "val_delta1": best_delta1,  # alias for paper-side aggregation scripts
        "best_epoch": best_epoch,
        "test": test_m,
        "freeze": args.freeze,
        "label_fraction": args.label_fraction,
        "model": "EquiSSLDepthUNet",
    }
    torch.save(results, os.path.join(args.output_dir, "results.pth"))
    print(f"\nResults saved to {args.output_dir}/results.pth")


if __name__ == "__main__":
    main()
