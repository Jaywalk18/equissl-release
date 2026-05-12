"""
Training utilities: schedulers, distributed setup, checkpoint management.
Reused from RotMask V6 with minimal changes.
"""

import os
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional, List


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: int = 0,
    warmup_value: float = 0.0,
) -> np.ndarray:
    """Cosine annealing with linear warmup."""
    warmup = np.linspace(warmup_value, base_value, warmup_epochs)
    t = np.arange(epochs - warmup_epochs)
    cosine = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(math.pi * t / (epochs - warmup_epochs))
    )
    return np.concatenate([warmup, cosine])


def momentum_scheduler(
    base_momentum: float,
    final_momentum: float,
    epochs: int,
) -> np.ndarray:
    """Cosine schedule for teacher EMA momentum (increasing)."""
    t = np.arange(epochs)
    return final_momentum - 0.5 * (final_momentum - base_momentum) * (
        1 + np.cos(math.pi * t / epochs)
    )


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: Optional[float] = None,
    **extra,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best_metric is not None:
        state["best_metric"] = best_metric
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Handle DDP module. prefix mismatch
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if len(model_keys) > 0 and len(ckpt_keys) > 0:
        model_has_module = any(k.startswith("module.") for k in model_keys)
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)
        if model_has_module and not ckpt_has_module:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not model_has_module and ckpt_has_module:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
