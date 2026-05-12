"""
EquiSSL Pretraining Script.
Equivariant Self-Supervised Learning on Icosphere.

Usage:
    python tools/pretrain.py --config configs/pretrain_default.yaml
    
    # Multi-GPU:
    torchrun --nproc_per_node=3 tools/pretrain.py --config configs/pretrain_default.yaml
"""

import os
import sys
import argparse
import yaml
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from equissl.models.encoder import EquiSSLEncoder
from equissl.losses.distillation import EquiSSLLoss
from equissl.data.structured3d import Structured3DSSL
from equissl.utils.training import (
    setup_distributed,
    cosine_scheduler,
    momentum_scheduler,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="EquiSSL Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain_default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="override config batch_size per GPU")
    parser.add_argument("--max_samples", type=int, default=None, help="limit dataset size for smoke test")
    return parser.parse_args()


def build_model(cfg):
    model_cfg = cfg["model"]
    ssl_cfg = cfg["ssl"]

    model = EquiSSLEncoder(
        img_rank=model_cfg["img_rank"],
        node_type=model_cfg["node_type"],
        embed_dim=model_cfg["embed_dim"],
        num_scales=model_cfg["num_scales"],
        in_scale_factor=model_cfg["in_scale_factor"],
        enc_depths=model_cfg["enc_depths"],
        bottleneck_depth=model_cfg["bottleneck_depth"],
        enc_num_heads=model_cfg["enc_num_heads"],
        d_head_coef=model_cfg["d_head_coef"],
        win_size_coef=model_cfg["win_size_coef"],
        mlp_ratio=model_cfg["mlp_ratio"],
        qkv_bias=model_cfg["qkv_bias"],
        drop_path_rate=model_cfg["drop_path_rate"],
        abs_pos_enc_in=model_cfg["abs_pos_enc_in"],
        abs_pos_enc=model_cfg["abs_pos_enc"],
        rel_pos_bias=model_cfg["rel_pos_bias"],
        rel_pos_bias_size=model_cfg.get("rel_pos_bias_size", 7),
        proj_dim=ssl_cfg["proj_dim"],
        out_dim=ssl_cfg["out_dim"],
        mask_ratio=ssl_cfg["mask_ratio"],
        teacher_momentum=ssl_cfg["teacher_momentum"],
        teacher_temp=ssl_cfg["teacher_temp"],
        student_temp=ssl_cfg["student_temp"],
        center_momentum=ssl_cfg["center_momentum"],
        dec_depths=model_cfg.get("dec_depths", [2, 2, 2, 2]),
        dec_num_heads=model_cfg.get("dec_num_heads", [16, 16, 8, 4]),
        rot_bins=ssl_cfg.get("rot_bins", 36),
        equivariant_rpe=model_cfg.get("equivariant_rpe", False),
        n_gauges=model_cfg.get("n_gauges", 6),
        area_weighted=model_cfg.get("area_weighted", True),
    )
    return model


def build_dataset(cfg, split="train", max_samples=None):
    data_cfg = cfg["data"]
    dataset = Structured3DSSL(
        root_dir=data_cfg["root_dir"],
        img_rank=cfg["model"]["img_rank"],
        node_type=cfg["model"]["node_type"],
        split=split,
        image_key=data_cfg["image_key"],
        normalize_mean=data_cfg["normalize_mean"],
        normalize_std=data_cfg["normalize_std"],
        color_augment=data_cfg.get("color_augment", True),
        max_samples=max_samples,
    )
    return dataset


def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "center" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=train_cfg["lr"])
    return optimizer


def train_one_epoch(
    model, loss_fn, dataloader, optimizer,
    epoch, lr_schedule, momentum_schedule, cfg,
    rank=0, accum_steps=1,
):
    model.train()
    train_cfg = cfg["training"]
    total_steps = len(dataloader)
    log_interval = train_cfg.get("log_interval", 50)
    clip_grad = train_cfg.get("clip_grad", 3.0)

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_patch_loss = 0.0
    epoch_contrast_loss = 0.0
    epoch_rot_loss = 0.0
    epoch_koleo_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        global_step = epoch * total_steps + step

        lr = lr_schedule[epoch]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        momentum = momentum_schedule[epoch]

        sphere_rgb = batch["sphere_rgb"].cuda(non_blocking=True)

        outputs = model(sphere_rgb)

        losses = loss_fn(outputs)
        loss = losses["total"] / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()

            if hasattr(model, "module"):
                model.module.update_teacher(momentum)
            else:
                model.update_teacher(momentum)

        epoch_loss += losses["total"].item()
        epoch_cls_loss += losses["cls"].item()
        epoch_patch_loss += losses["patch"].item()
        epoch_contrast_loss += losses.get("contrast", torch.tensor(0.0)).item()
        epoch_rot_loss += losses.get("rot", torch.tensor(0.0)).item()
        epoch_koleo_loss += losses.get("koleo", torch.tensor(0.0)).item()
        num_batches += 1

        if rank == 0 and (step + 1) % log_interval == 0:
            avg_loss = epoch_loss / num_batches
            avg_cls = epoch_cls_loss / num_batches
            avg_patch = epoch_patch_loss / num_batches
            avg_contrast = epoch_contrast_loss / num_batches
            avg_rot = epoch_rot_loss / num_batches
            avg_koleo = epoch_koleo_loss / num_batches
            print(
                f"  [{step+1}/{total_steps}] "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} patch={avg_patch:.4f} "
                f"contrast={avg_contrast:.4f} rot={avg_rot:.4f} koleo={avg_koleo:.4f} "
                f"lr={lr:.6f} mom={momentum:.4f}"
            )

    return epoch_loss / max(num_batches, 1)


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    output_dir = args.output_dir or cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    if is_main:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if is_main:
        print("=" * 60)
        print("EquiSSL Pretraining")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Output dir: {output_dir}")

    model = build_model(cfg)
    model = model.cuda()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    dataset = build_dataset(cfg, split="train", max_samples=args.max_samples)

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    optimizer = build_optimizer(model, cfg)
    loss_fn = EquiSSLLoss(
        student_temp=cfg["ssl"]["student_temp"],
        teacher_temp=cfg["ssl"]["teacher_temp"],
        cls_weight=cfg["ssl"]["cls_weight"],
        patch_weight=cfg["ssl"]["patch_weight"],
        contrast_weight=cfg["ssl"].get("contrast_weight", 0.0),
        contrast_temp=cfg["ssl"].get("contrast_temp", 0.2),
        rot_weight=cfg["ssl"].get("rot_weight", 0.5),
        koleo_weight=cfg["ssl"].get("koleo_weight", 0.01),
        rot_bins=cfg["ssl"].get("rot_bins", 36),
    )

    train_cfg = cfg["training"]
    epochs = train_cfg["epochs"]
    lr_schedule = cosine_scheduler(
        base_value=train_cfg["lr"],
        final_value=train_cfg["min_lr"],
        epochs=epochs,
        warmup_epochs=train_cfg["warmup_epochs"],
    )
    mom_schedule = momentum_scheduler(
        base_momentum=cfg["ssl"]["teacher_momentum"],
        final_momentum=train_cfg["teacher_momentum_end"],
        epochs=epochs,
    )

    start_epoch = 0
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1

    accum_steps = train_cfg.get("gradient_accumulation", 1)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
        print(f"Dataset: {len(dataset)} images")
        print(f"Epochs: {start_epoch} -> {epochs}")
        print(f"Batch: {train_cfg['batch_size']} x {world_size} GPU x {accum_steps} accum = {train_cfg['batch_size'] * world_size * accum_steps}")
        print("=" * 60)

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main:
            print(f"\nEpoch {epoch}/{epochs-1}")

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, loss_fn, dataloader, optimizer,
            epoch, lr_schedule, mom_schedule, cfg,
            rank=rank, accum_steps=accum_steps,
        )
        elapsed = time.time() - t0

        if is_main:
            print(f"  Epoch {epoch} done in {elapsed:.0f}s, avg_loss={avg_loss:.4f}")

            if (epoch + 1) % train_cfg.get("save_freq", 10) == 0 or epoch == epochs - 1:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth")
                save_model = model.module if hasattr(model, "module") else model
                save_checkpoint(ckpt_path, epoch, save_model, optimizer)
                print(f"  Saved checkpoint: {ckpt_path}")
                # Clean up old periodic checkpoints (keep only latest + final)
                import glob
                for old_ckpt in sorted(glob.glob(os.path.join(output_dir, "checkpoint_epoch*.pth"))):
                    if old_ckpt != ckpt_path:
                        os.remove(old_ckpt)
                        print(f"  Removed old checkpoint: {old_ckpt}")

            latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
            save_model = model.module if hasattr(model, "module") else model
            save_checkpoint(latest_path, epoch, save_model, optimizer)

    if is_main:
        print("\nPretraining complete!")


if __name__ == "__main__":
    main()
