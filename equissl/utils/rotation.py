"""
SO(3) rotation utilities for icosphere.

On an icosphere, an SO(3) rotation acts as a permutation of nodes
(via nearest-neighbor assignment after rotating 3D coordinates).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional


def sample_uniform_rotation() -> np.ndarray:
    """Sample a uniformly random SO(3) rotation matrix (Haar measure)."""
    return R.random().as_matrix()


def sample_rotation_within(max_angle_deg: float = 35.0) -> np.ndarray:
    """Sample a random rotation with magnitude <= max_angle_deg."""
    angle = np.random.uniform(0, np.radians(max_angle_deg))
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    return R.from_rotvec(angle * axis).as_matrix()


def rotation_to_permutation(
    rot_matrix: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """
    Convert an SO(3) rotation to a node permutation on the icosphere.

    For each node i, find the node j closest to R @ normal_i.
    This is a nearest-neighbor approximation of the continuous rotation.

    Args:
        rot_matrix: (3, 3) rotation matrix
        normals: (N, 3) unit normals of icosphere nodes

    Returns:
        perm: (N,) int array — perm[i] = j means node i maps to node j
    """
    rotated = normals @ rot_matrix.T
    dots = rotated @ normals.T
    perm = np.argmax(dots, axis=1)
    return perm


def apply_permutation(
    features: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a node permutation to features.

    Args:
        features: (B, N, C) or (N, C)
        perm: (N,) long tensor

    Returns:
        permuted features with same shape
    """
    if features.dim() == 2:
        return features[perm]
    elif features.dim() == 3:
        return features[:, perm]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {features.dim()}D")


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Compute the inverse of a permutation."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def precompute_rotation_permutations(
    normals: np.ndarray,
    num_rotations: int = 64,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute a bank of random SO(3) rotations and their node permutations.

    Args:
        normals: (N, 3) icosphere node normals
        num_rotations: number of rotations to precompute
        seed: random seed for reproducibility

    Returns:
        rot_matrices: (num_rotations, 3, 3)
        permutations: (num_rotations, N) int
    """
    if seed is not None:
        np.random.seed(seed)

    rot_matrices = np.stack([sample_uniform_rotation() for _ in range(num_rotations)])
    permutations = np.stack([
        rotation_to_permutation(rot_matrices[i], normals)
        for i in range(num_rotations)
    ])
    return rot_matrices, permutations
