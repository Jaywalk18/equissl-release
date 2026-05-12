"""
Gauge-Equivariant Relative Position Encoding for icosphere attention.

The key idea: instead of computing relative position bias in a single
(gauge-dependent) local coordinate frame, we compute it under multiple
gauge choices (C_6 rotations around the vertex normal) and average.
This makes the bias invariant to the gauge choice, achieving SO(3) equivariance.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import IcoSphereRef, asSpherical
from network.position_encoding import get_rotation_matrices


def _rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation matrix around unit axis by angle (radians)."""
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12)
    K = np.zeros((*axis.shape[:-1], 3, 3))
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


class GaugeEquivariantRPE(nn.Module):
    """
    Gauge-equivariant relative position bias via C_n gauge pooling.

    For each query vertex:
    1. Compute a base local frame (same as SphereUFormer)
    2. Rotate the frame by 2*pi*k/n_gauges around the vertex normal (k=0..n_gauges-1)
    3. In each rotated frame, compute relative (phi, theta) of neighbors
    4. Look up bias from learnable grid for each gauge
    5. Average over all gauges -> gauge-invariant bias

    Also supports area-weighted attention via log(area) additive bias.
    """

    def __init__(
        self,
        rank: int,
        icosphere_ref: IcoSphereRef,
        win_size_coef: int,
        rel_pos_bias_size: int,
        num_heads: int,
        n_gauges: int = 6,
        area_weighted: bool = True,
        init_variance: float = 10.0,
    ):
        super().__init__()

        self.rank = rank
        self.n_gauges = n_gauges
        self.area_weighted = area_weighted

        normals = icosphere_ref.get_normals(rank)
        normals_rphitheta = asSpherical(normals)
        base_rot_mat = get_rotation_matrices(normals_rphitheta)

        mapping = icosphere_ref.get_neighbor_mapping(rank=rank, depth=win_size_coef)
        self.num_nodes = len(mapping)
        self.num_keys = max(len(_) for _ in mapping)

        idx = torch.arange(0, self.num_nodes).unsqueeze(1).expand(-1, self.num_keys).clone()
        idx_mask = torch.zeros(self.num_nodes, self.num_keys).bool()
        for i, keys in tqdm(enumerate(mapping), desc=f"GaugeEquivariantRPE - index mapping {rank}"):
            idx[i, :len(keys)] = torch.tensor(list(keys))
            idx_mask[i, :len(keys)] = 1

        self.register_buffer("idx", idx[None, None, :, :, None], persistent=False)
        self.register_buffer("idx_mask", idx_mask[None, None, :, :], persistent=False)

        expanded_normals = np.tile(normals[:, None, :], (1, self.num_keys, 1))
        expanded_idx_np = idx.numpy()
        aligned_neighbors = np.zeros((self.num_nodes, self.num_keys, 3))
        for i in range(self.num_nodes):
            for j in range(self.num_keys):
                aligned_neighbors[i, j] = normals[expanded_idx_np[i, j]]

        gauge_angles = [2 * np.pi * k / n_gauges for k in range(n_gauges)]
        gauge_rot_mats = _rotation_around_axis(normals, 0)

        all_gauge_coords = []
        for g_idx, angle in enumerate(gauge_angles):
            gauge_rot = _rotation_around_axis(normals, angle)
            combined_rot = base_rot_mat @ gauge_rot

            rotated = np.einsum('nij,nkj->nki', combined_rot, aligned_neighbors)
            rotated_flat = rotated.reshape(-1, 3)
            norms = np.linalg.norm(rotated_flat, axis=-1, keepdims=True)
            rotated_flat = rotated_flat / (norms + 1e-12)
            sph = asSpherical(rotated_flat)
            rel_coords = sph[:, 1:] - np.array([[90, 0]])
            rel_coords = rel_coords.reshape(self.num_nodes, self.num_keys, 2)
            all_gauge_coords.append(rel_coords)

        all_gauge_coords = np.stack(all_gauge_coords, axis=0)
        self.register_buffer(
            "gauge_relative_coords",
            torch.tensor(all_gauge_coords, dtype=torch.float32),
            persistent=False,
        )

        self.bias_grid = nn.Parameter(
            init_variance * torch.randn(1, num_heads, rel_pos_bias_size, rel_pos_bias_size),
            requires_grad=True,
        )

        if area_weighted:
            ico = icosphere_ref.get_icosphere(rank, refine=True)
            vertices = ico.vertices
            faces = ico.faces
            areas = np.zeros(len(vertices))
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                for vi in face:
                    areas[vi] += face_area / 3.0

            area_bias = np.zeros((self.num_nodes, self.num_keys))
            for i in range(self.num_nodes):
                for j in range(self.num_keys):
                    area_bias[i, j] = np.log(areas[expanded_idx_np[i, j]] + 1e-12)
            area_bias = area_bias - area_bias.mean()
            self.register_buffer(
                "area_bias",
                torch.tensor(area_bias, dtype=torch.float32)[None, None, :, :],
                persistent=False,
            )

    def get_neighbor_idx(self):
        return self.idx, self.idx_mask

    def forward(self, keys: Tensor):
        N, H, D, K, C_H = keys.shape

        total_bias = 0.0
        for g in range(self.n_gauges):
            coords = self.gauge_relative_coords[g].unsqueeze(0)
            coords_norm = coords / (coords.abs().max() + 1e-8)
            bias = F.grid_sample(self.bias_grid, grid=coords_norm, align_corners=True)
            total_bias = total_bias + bias

        avg_bias = total_bias / self.n_gauges

        if self.area_weighted:
            avg_bias = avg_bias + self.area_bias

        return self.gauge_relative_coords[0].unsqueeze(0), avg_bias
