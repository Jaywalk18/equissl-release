"""
EquiSSL Downstream Finetuning: Semantic Segmentation with U-Net decoder.

Usage:
    # Full fine-tune
    python tools/finetune_seg.py \
        --pretrained outputs/pretrain/checkpoint_epoch99.pth \
        --output_dir outputs/finetune_unet_full

    # Linear probe (freeze encoder)
    python tools/finetune_seg.py \
        --pretrained outputs/pretrain/checkpoint_epoch99.pth \
        --output_dir outputs/finetune_unet_linear --freeze
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
from equissl.data.structured3d_seg import Structured3DSeg
from equissl.losses.segmentation import FocalLoss, CEDiceLoss, build_class_weights
from equissl.utils.training import save_checkpoint
from equissl.losses.segmentation import FocalLoss, CEDiceLoss, build_class_weights


def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0

    for step, batch in enumerate(dataloader):
        rgb = batch["sphere_rgb"].cuda(non_blocking=True)
        labels = batch["sphere_gt_sem"].cuda(non_blocking=True).long()

        logits = model(rgb)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        valid = labels != 0
        total_correct += (pred[valid] == labels[valid]).sum().item()
        total_pixels += valid.sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = total_correct / max(total_pixels, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, num_classes, ignore_index=0):
    model.eval()
    intersection = torch.zeros(num_classes, device="cpu")
    union = torch.zeros(num_classes, device="cpu")

    for batch in dataloader:
        rgb = batch["sphere_rgb"].cuda(non_blocking=True)
        labels = batch["sphere_gt_sem"].long()

        logits = model(rgb)
        pred = logits.argmax(dim=-1).cpu()

        for c in range(num_classes):
            if c == ignore_index:
                continue
            pred_c = pred == c
            label_c = labels == c
            intersection[c] += (pred_c & label_c).sum().item()
            union[c] += (pred_c | label_c).sum().item()

    iou = intersection / union.clamp(min=1)
    valid = union > 0
    valid[ignore_index] = False
    miou = iou[valid].mean().item()
    per_class = {i: iou[i].item() for i in range(num_classes) if valid[i]}
    return miou, per_class


def parse_args():
    parser = argparse.ArgumentParser(description="EquiSSL Segmentation Finetuning (U-Net)")
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--data_dir", type=str, default="${STANFORD2D3D_PATH}")
    parser.add_argument("--dataset", type=str, default="s2d3d", choices=["s2d3d", "s3d"],
                        help="Dataset: s2d3d=Stanford2D3D, s3d=Structured3D")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune_seg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze", action="store_true", help="freeze encoder (linear probe)")
    parser.add_argument("--load_decoder", action="store_true", help="load pretrained decoder weights")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    # Loss options
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "focal", "ce_dice", "ce_weighted"],
                        help="loss function: ce, focal, ce_dice, ce_weighted")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="focal loss gamma")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="dice loss weight in ce_dice")
    # Layer-wise lr decay
    parser.add_argument("--layer_decay", type=float, default=None,
                        help="layer-wise lr decay rate (e.g. 0.65). None = uniform lr")
    # RPE ablation
    parser.add_argument("--rpe_mode", type=str, default="config",
                        choices=["config", "none", "standard", "equivariant"],
                        help="RPE mode: config=follow yaml, none=disable RPE, "
                             "standard=plain RPE, equivariant=GE-RPE from config")
    parser.add_argument("--n_gauges", type=int, default=None,
                        help="Override n_gauges for GE-RPE (default: from config, typically 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_area_weight", action="store_true",
                        help="Disable area weighting in GE-RPE")
    return parser.parse_args()


def _build_layer_decay_params(model, base_lr, layer_decay):
    """Build param groups with layer-wise lr decay.
    Deeper layers get higher lr, shallow layers get lower lr.
    Standard technique from MAE/iBOT finetuning."""
    param_groups = {}
    # Assign layer ids: encoder blocks get increasing ids, decoder/head get max id
    num_enc_scales = len(model.encoder.enc_blocks) if hasattr(model.encoder, 'enc_blocks') else 4
    num_layers = num_enc_scales + 1  # +1 for bottleneck
    max_layer_id = num_layers

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Determine layer id
        if "enc_blocks" in name:
            # e.g. encoder.enc_blocks.0.xxx -> layer 0
            layer_id = int(name.split("enc_blocks.")[1].split(".")[0])
        elif "bottleneck" in name:
            layer_id = num_enc_scales
        else:
            # decoder, seg_head, input_proj -> max lr
            layer_id = max_layer_id

        scale = layer_decay ** (max_layer_id - layer_id)
        group_key = f"layer_{layer_id}_scale_{scale:.4f}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.01,
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
    print("EquiSSL U-Net Segmentation Finetuning")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Freeze encoder: {args.freeze}")
    print(f"  Label fraction: {args.label_fraction}")
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
        n_gauges=args.n_gauges or model_cfg.get("n_gauges", 6),
        area_weighted=use_area_weighted,
    )
    n_gauges_val = args.n_gauges or model_cfg.get("n_gauges", 6)
    print(f"  RPE mode: {args.rpe_mode} (rel_pos_bias={use_rel_pos_bias}, equivariant={use_equivariant_rpe}, n_gauges={n_gauges_val})")

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Handle two checkpoint formats:
    #   (a) SSL pretrain checkpoint from EquiSSLEncoder -> keys prefixed "student."
    #   (b) EquiSSLSegUNet finetune checkpoint (stage1) -> keys prefixed "encoder."
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("student."):
            encoder_state[k.replace("student.", "", 1)] = v
        elif k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "", 1)] = v
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded encoder: {len(encoder_state)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    num_classes = Structured3DSeg.NUM_CLASSES if args.dataset == "s3d" else Stanford2D3DSeg.NUM_CLASSES
    print(f"Num classes: {num_classes} (dataset={args.dataset})")
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
                {k.replace("decoder.", "", 1): v for k, v in decoder_state.items()},
                strict=False,
            )
            print(f"Loaded pretrained decoder: {len(decoder_state)} keys, {len(dec_missing)} missing")
        else:
            print("No decoder weights found in checkpoint, using random init")

    # Also restore the seg_head when continuing from a stage1 finetune checkpoint.
    # Shapes match only when num_classes is the same across runs (14 for Stanford2D3D).
    seg_head_state = {
        k.replace("seg_head.", "", 1): v
        for k, v in state.items()
        if k.startswith("seg_head.")
    }
    if seg_head_state:
        try:
            sh_missing, sh_unexpected = model.seg_head.load_state_dict(
                seg_head_state, strict=False
            )
            print(f"Loaded seg_head: {len(seg_head_state)} keys, {len(sh_missing)} missing")
        except (AttributeError, RuntimeError) as e:
            print(f"seg_head restore skipped ({e})")

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

    if args.dataset == "s3d":
        s3d_kwargs = dict(
            data_dir=args.data_dir, img_rank=model_cfg["img_rank"],
            node_type=model_cfg["node_type"], num_scales=model_cfg["num_scales"],
            in_scale_factor=model_cfg["in_scale_factor"],
            normalize_mean=cfg["data"]["normalize_mean"],
            normalize_std=cfg["data"]["normalize_std"],
        )
        train_max = int(18362 * args.label_fraction) if args.label_fraction < 1.0 else -1
        train_dataset = Structured3DSeg(split="train", max_samples=train_max, augment=True, **s3d_kwargs)
        val_dataset = Structured3DSeg(split="val", max_samples=-1, **s3d_kwargs)
        test_dataset = Structured3DSeg(split="test", max_samples=-1, **s3d_kwargs)
    else:
        train_dataset = Stanford2D3DSeg(split="train", label_fraction=args.label_fraction, augment=True, **ds_kwargs)
        val_dataset = Stanford2D3DSeg(split="val", **ds_kwargs)
        test_dataset = Stanford2D3DSeg(split="test", **ds_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Build loss function ---
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, ignore_index=0).cuda()
        print(f"Loss: FocalLoss(gamma={args.focal_gamma})")
    elif args.loss == "ce_dice":
        criterion = CEDiceLoss(dice_weight=args.dice_weight, ignore_index=0).cuda()
        print(f"Loss: CEDiceLoss(dice_weight={args.dice_weight})")
    elif args.loss == "ce_weighted":
        # Pre-computed inverse-frequency weights from Stanford2D3D training set
        # Avoids slow dataset traversal at startup
        # Classes: 0=unknown, 1=beam, 2=board, 3=bookcase, 4=ceiling, 5=chair,
        #          6=clutter, 7=column, 8=door, 9=floor, 10=sofa, 11=table, 12=wall, 13=window
        class_weights = torch.tensor([
            0.0,     # 0: unknown (ignored)
            8.5,     # 1: beam (very rare)
            2.1,     # 2: board
            1.8,     # 3: bookcase
            0.3,     # 4: ceiling (very common)
            1.5,     # 5: chair
            0.8,     # 6: clutter
            6.0,     # 7: column (rare)
            3.5,     # 8: door (rare)
            0.3,     # 9: floor (very common)
            4.0,     # 10: sofa (rare)
            1.2,     # 11: table
            0.2,     # 12: wall (most common)
            2.0,     # 13: window
        ], dtype=torch.float32).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
        print(f"Loss: CrossEntropyLoss(class_weighted)")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        print("Loss: CrossEntropyLoss")

    # --- Build optimizer (with optional layer-wise lr decay) ---
    if args.layer_decay is not None and not args.freeze:
        param_groups = _build_layer_decay_params(model, args.lr, args.layer_decay)
        print(f"Optimizer: AdamW with layer_decay={args.layer_decay}, {len(param_groups)} groups")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        scheduler.step()

        val_miou, _ = evaluate(model, val_loader, num_classes)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs-1} ({elapsed:.0f}s) "
              f"loss={train_loss:.4f} acc={train_acc:.4f} val_mIoU={val_miou:.4f} lr={lr_now:.6f}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pth"),
                epoch, model, optimizer, best_metric=best_miou,
            )
            print(f"  -> New best! mIoU={best_miou:.4f}")

        save_checkpoint(
            os.path.join(args.output_dir, "checkpoint_latest.pth"),
            epoch, model, optimizer, best_metric=val_miou,
        )

    print(f"\nBest val mIoU: {best_miou:.4f} at epoch {best_epoch}")

    print("\nEvaluating best model on test set...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model = model.cuda()

    test_miou, per_class = evaluate(model, test_loader, num_classes)
    print(f"Test mIoU: {test_miou:.4f}")
    print("Per-class IoU:")
    for c, iou in sorted(per_class.items()):
        print(f"  Class {c}: {iou:.4f}")

    results = {
        "best_val_miou": best_miou, "best_epoch": best_epoch,
        "test_miou": test_miou, "per_class_iou": per_class,
        "freeze": args.freeze, "label_fraction": args.label_fraction,
        "model": "EquiSSLSegUNet",
    }
    torch.save(results, os.path.join(args.output_dir, "results.pth"))
    print(f"\nResults saved to {args.output_dir}/results.pth")


if __name__ == "__main__":
    main()
