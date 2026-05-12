"""
Stanford2D3D dataset for depth estimation on icosphere.
Mirrors stanford2d3d_seg.py but returns `sphere_gt_depth` (in meters) + a validity mask.

Depth format: uint16 PNG, unit = millimeters, invalid = 0 or > MAX_DEPTH_MM.
SphereUFormer convention: MAX_DEPTH = 5120 mm (5.12 m), same here for consistency.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import IcoSphereRef

from ..utils.sphere import erp_to_icosphere_grid

SPHERE_UFORMER_DATA = "${SPHERE_UFORMER_SRC}/data"


class Stanford2D3DDepth(Dataset):
    """
    Stanford2D3D dataset for dense depth estimation on icosphere.
    Depth is returned in meters; invalid pixels marked by `sphere_valid_mask=False`.
    """

    MAX_DEPTH_M = 5.12  # matches SphereUFormer's MAX_DEPTH = 5120 mm

    def __init__(
        self,
        data_dir: str = "${STANFORD2D3D_PATH}",
        split: str = "train",
        img_rank: int = 7,
        node_type: str = "vertex",
        num_scales: int = 4,
        in_scale_factor: int = 2,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.225,
        label_fraction: float = 1.0,
        augment: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augment = augment and (split == "train")

        self.valid_mask = cv2.imread(
            os.path.join(SPHERE_UFORMER_DATA, "stanford2d3d_mask_pretty.png"), -1
        ) > 0

        self.icosphere_ref = IcoSphereRef(node_type)
        self.sphere_grid = erp_to_icosphere_grid(self.icosphere_ref, img_rank)

        proj_rank = img_rank - 1 if in_scale_factor == 2 else img_rank
        self.label_grid = erp_to_icosphere_grid(self.icosphere_ref, proj_rank)

        split_file = os.path.join(
            SPHERE_UFORMER_DATA, "splits_2d3d", f"stanford2d3d_{split}.txt"
        )
        self.samples = self._load_split(split_file)

        if label_fraction < 1.0 and split == "train":
            n = max(1, int(len(self.samples) * label_fraction))
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(self.samples))[:n]
            self.samples = [self.samples[i] for i in sorted(indices)]

        print(f"Stanford2D3DDepth: {len(self.samples)} samples ({split})")

    def _load_split(self, split_file: str):
        samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rgb_rel, depth_rel = parts[0], parts[1]
                rgb_path = os.path.join(self.data_dir, rgb_rel)
                depth_path = os.path.join(self.data_dir, depth_rel)
                if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                    samples.append((rgb_path, depth_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, depth_path = self.samples[idx]

        try:
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            depth = cv2.imread(depth_path, -1).astype(np.float32)  # uint16, in mm
        except Exception as e:
            print(f"Error loading {rgb_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

        # mm -> meters; mark invalid (0 or saturated/too-far) as 0
        depth = depth / 1000.0  # meters
        invalid_pixel = (depth <= 0.01) | (depth > self.MAX_DEPTH_M)

        # Apply ERP boundary validity mask
        mask = self.valid_mask.copy()
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0])) > 0
        valid_erp = mask & ~invalid_pixel
        depth[~valid_erp] = 0.0  # zeros flow through nearest grid_sample safely

        if self.augment:
            H, W = rgb.shape[:2]
            # Yaw rotation (ERP is yaw-equivariant via horizontal roll)
            shift = np.random.randint(0, W)
            rgb = np.roll(rgb, shift, axis=1)
            depth = np.roll(depth, shift, axis=1)
            valid_erp = np.roll(valid_erp, shift, axis=1)
            # Horizontal flip
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                depth = np.ascontiguousarray(depth[:, ::-1])
                valid_erp = np.ascontiguousarray(valid_erp[:, ::-1])

        # RGB -> sphere_rgb at img_rank (bilinear)
        rgb_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        sphere_rgb = F.grid_sample(
            rgb_tensor, self.sphere_grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        ).squeeze(0).squeeze(1).T  # (N_img, 3)

        # Depth -> sphere_gt_depth at proj_rank (nearest to avoid blending invalid)
        depth_tensor = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        sphere_depth = F.grid_sample(
            depth_tensor, self.label_grid,
            mode="nearest", padding_mode="border", align_corners=False,
        ).squeeze(0).squeeze(0).squeeze(0).float()  # (N_proj,)

        # Validity mask -> sphere_valid_mask at proj_rank (nearest)
        valid_tensor = torch.tensor(valid_erp.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        sphere_valid = F.grid_sample(
            valid_tensor, self.label_grid,
            mode="nearest", padding_mode="border", align_corners=False,
        ).squeeze(0).squeeze(0).squeeze(0)
        sphere_valid = (sphere_valid > 0.5) & (sphere_depth > 0.01)  # (N_proj,) bool

        sphere_rgb = (sphere_rgb - self.normalize_mean) / self.normalize_std

        return {
            "sphere_rgb": sphere_rgb,
            "sphere_gt_depth": sphere_depth,
            "sphere_valid_mask": sphere_valid,
        }
