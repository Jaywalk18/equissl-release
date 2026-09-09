"""
Masking strategies for icosphere-based self-supervised learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from ..utils.sphere import compute_node_areas, IcoSphereRef


class IcosphereMasking(nn.Module):
    """
    Generate random masks on icosphere nodes.

    Since icosphere vertices are already near-uniformly distributed on the sphere,
    simple random masking is approximately area-uniform.
    Optionally weight by Voronoi dual area for exact uniformity.
    """

    def __init__(
        self,
        num_nodes: int,
        mask_ratio: float = 0.75,
        area_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.mask_ratio = mask_ratio
        self.num_masked = int(num_nodes * mask_ratio)

        if area_weights is not None:
            probs = area_weights / area_weights.sum()
            self.register_buffer("mask_probs", probs.unsqueeze(0), persistent=False)
        else:
            self.register_buffer(
                "mask_probs",
                torch.ones(1, num_nodes) / num_nodes,
                persistent=False,
            )

    @torch.no_grad()
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate random masks.

        Returns:
            mask: [B, N] bool — True = masked (loss computed here)
        """
        noise = torch.rand(batch_size, self.num_nodes, device=device)
        weighted_noise = noise / self.mask_probs.to(device)

        _, indices = weighted_noise.topk(self.num_masked, dim=-1, largest=False)

        mask = torch.zeros(batch_size, self.num_nodes, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)
        return mask


class IcosphereBlockMasking(nn.Module):
    """
    Block (geodesic neighborhood) masking on icosphere.
    Masks connected regions on the sphere surface.
    """

    def __init__(
        self,
        icosphere_ref: IcoSphereRef,
        rank: int,
        mask_ratio: float = 0.75,
        num_seeds: Tuple[int, int] = (2, 6),
        expansion_depth: int = 3,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_seeds = num_seeds
        self.expansion_depth = expansion_depth

        neighbors = icosphere_ref.get_neighbor_mapping(rank, depth=1)
        self.num_nodes = len(neighbors)
        self.num_masked = int(self.num_nodes * mask_ratio)

        adj_list = []
        max_neighbors = max(len(n) for n in neighbors)
        for n_set in neighbors:
            padded = list(n_set) + [list(n_set)[0]] * (max_neighbors - len(n_set))
            adj_list.append(padded)
        self.register_buffer(
            "adj", torch.tensor(adj_list, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            n_seeds = torch.randint(
                self.num_seeds[0], self.num_seeds[1] + 1, (1,)
            ).item()
            seed_indices = torch.randperm(self.num_nodes, device=device)[:n_seeds]

            selected = set(seed_indices.tolist())
            frontier = set(seed_indices.tolist())

            for _ in range(self.expansion_depth):
                if len(selected) >= self.num_masked:
                    break
                new_frontier = set()
                for idx in frontier:
                    neighbors_of_idx = self.adj[idx].tolist()
                    new_frontier.update(neighbors_of_idx)
                new_frontier -= selected
                selected.update(new_frontier)
                frontier = new_frontier

            selected_list = list(selected)[: self.num_masked]
            if len(selected_list) < self.num_masked:
                remaining = set(range(self.num_nodes)) - set(selected_list)
                extra = list(remaining)[: self.num_masked - len(selected_list)]
                selected_list.extend(extra)

            mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
            mask[torch.tensor(selected_list, device=device)] = True
            masks.append(mask)

        return torch.stack(masks, dim=0)
