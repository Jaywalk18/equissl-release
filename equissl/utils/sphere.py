"""
Icosphere utilities and SO(3) rotation operations for EquiSSL.
Wraps SphereUFormer's trimesh_utils with additional rotation support.
"""

import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import (
    IcoSphereRef,
    get_icosphere,
    find_vertex_neighbors,
    find_face_neighbors,
    asSpherical,
    asCartesian,
)


def sample_so3_rotation() -> np.ndarray:
    """Sample a uniform random SO(3) rotation matrix (Haar measure)."""
    return R.random().as_matrix()


def sample_so3_rotation_bounded(max_angle_deg: float = 35.0) -> np.ndarray:
    """
    Sample a random SO(3) rotation with magnitude bounded by max_angle_deg.
    Uses axis-angle uniform sampling: random axis + uniform angle in [0, max_angle].
    """
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, max_angle_deg) * np.pi / 180.0
    return R.from_rotvec(axis * angle).as_matrix()


def compute_rotation_permutation(
    normals: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the nearest-neighbor node permutation induced by a 3D rotation.

    For each rotated node position, find the closest original node.
    This gives an index mapping: new_features[i] = old_features[perm[i]].

    Uses KD-tree for O(N log N) instead of O(N^2) full dot-product matrix,
    which would be infeasible for high-rank icospheres (rank 7 has 163842 nodes).

    Args:
        normals: (N, 3) unit vectors for icosphere nodes
        rotation_matrix: (3, 3) SO(3) rotation matrix

    Returns:
        perm: (N,) int64 — permutation indices
    """
    from scipy.spatial import cKDTree
    rotated = (rotation_matrix @ normals.T).T
    tree = cKDTree(normals)
    _, perm = tree.query(rotated, k=1)
    return perm.astype(np.int64)


def apply_rotation_to_features(
    features: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a rotation (as node permutation) to icosphere features.

    Args:
        features: (B, N, C) or (N, C)
        perm: (N,) long — permutation from compute_rotation_permutation

    Returns:
        rotated features: same shape as input
    """
    if features.dim() == 2:
        return features[perm]
    elif features.dim() == 3:
        return features[:, perm]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {features.dim()}D")


def compute_inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Compute the inverse of a permutation."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def erp_to_icosphere_grid(
    icosphere_ref: IcoSphereRef,
    rank: int,
) -> torch.Tensor:
    """
    Precompute the grid_sample coordinates for ERP -> icosphere resampling.

    Returns:
        grid: (1, 1, N, 2) — normalized coords for F.grid_sample
    """
    normals = icosphere_ref.get_normals(rank)
    rphitheta = asSpherical(normals)

    theta = rphitheta[:, 2]  # [-180, 180]
    phi = rphitheta[:, 1]    # [0, 180]

    w_norm = theta / 180.0           # [-1, 1]
    h_norm = phi / 180.0 * 2 - 1    # [-1, 1]

    grid = np.stack([w_norm, h_norm], axis=1)
    grid = torch.tensor(grid, dtype=torch.float32).reshape(1, 1, -1, 2)
    return grid


def compute_node_areas(
    icosphere_ref: IcoSphereRef,
    rank: int,
) -> torch.Tensor:
    """
    Compute the approximate spherical area weight for each icosphere vertex.
    Uses the Voronoi dual area: sum of 1/3 of each adjacent face's area.

    Returns:
        areas: (N,) float — area weights (sum to ~4*pi for unit sphere)
    """
    ico = icosphere_ref.get_icosphere(rank, refine=True)
    vertices = ico.vertices
    faces = ico.faces
    num_vertices = len(vertices)

    areas = np.zeros(num_vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        for vi in face:
            areas[vi] += face_area / 3.0

    return torch.tensor(areas, dtype=torch.float32)
