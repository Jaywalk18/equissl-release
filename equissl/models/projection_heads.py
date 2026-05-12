"""
DINO/iBOT-style projection heads for self-distillation.
Directly reused from RotMask V6 — geometry-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """3-layer MLP -> L2 normalize -> linear projection for CLS token."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.normal_(self.last_layer.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> [B, out_dim]"""
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class PatchProjectionHead(nn.Module):
    """Lightweight 2-layer MLP for patch-level distillation."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 65536,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim] -> [B, N, out_dim]"""
        return self.mlp(x)
