"""
Stanford2D3D dataset for semantic segmentation on icosphere.
Loads ERP panoramic RGB + semantic labels, resamples to icosphere.
References SphereUFormer's data loading pipeline.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import IcoSphereRef, asSpherical

from ..utils.sphere import erp_to_icosphere_grid

SPHERE_UFORMER_DATA = "${SPHERE_UFORMER_SRC}/data"


class Stanford2D3DSeg(Dataset):
    """
    Stanford2D3D dataset for semantic segmentation on icosphere.
    13 semantic classes + 1 unknown (class 0, ignored).
    """

    NUM_CLASSES = 14

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

        self.id2label = np.load(os.path.join(SPHERE_UFORMER_DATA, "stanford2d3d_id2label.npy"))
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

        print(f"Stanford2D3DSeg: {len(self.samples)} samples ({split})")

    def _load_split(self, split_file: str):
        samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rgb_rel, depth_rel = parts[0], parts[1]
                sem_rel = depth_rel.replace("/depth/", "/semantic/").replace(
                    "_depth.png", "_semantic.png"
                )
                rgb_path = os.path.join(self.data_dir, rgb_rel)
                sem_path = os.path.join(self.data_dir, sem_rel)
                if os.path.isfile(rgb_path) and os.path.isfile(sem_path):
                    samples.append((rgb_path, sem_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _semantic_to_labels(self, sem_rgb: np.ndarray) -> np.ndarray:
        idx = sem_rgb[..., 1].astype(np.int32) * 256 + sem_rgb[..., 2].astype(np.int32)
        label = self.id2label[idx]
        unk = sem_rgb[..., 0] != 0
        label[unk] = 0
        return label.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, sem_path = self.samples[idx]

        try:
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            sem_rgb = cv2.imread(sem_path)
            sem_rgb = cv2.cvtColor(sem_rgb, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {rgb_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

        sem_labels = self._semantic_to_labels(sem_rgb)

        mask = self.valid_mask.copy()
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0])) > 0
        sem_labels[~mask] = 0

        if self.augment:
            H, W = rgb.shape[:2]
            # Yaw rotation: random horizontal shift (ERP equivariant to yaw)
            shift = np.random.randint(0, W)
            rgb = np.roll(rgb, shift, axis=1)
            sem_labels = np.roll(sem_labels, shift, axis=1)
            mask = np.roll(mask, shift, axis=1)
            # Horizontal flip with 50% probability
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                sem_labels = np.ascontiguousarray(sem_labels[:, ::-1])

        rgb_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        sphere_rgb = F.grid_sample(
            rgb_tensor, self.sphere_grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        ).squeeze(0).squeeze(1).T

        sem_tensor = torch.tensor(sem_labels).unsqueeze(0).unsqueeze(0)
        sphere_sem = F.grid_sample(
            sem_tensor, self.label_grid,
            mode="nearest", padding_mode="border", align_corners=False,
        ).squeeze(0).squeeze(0).squeeze(0).long()

        sphere_rgb = (sphere_rgb - self.normalize_mean) / self.normalize_std

        return {
            "sphere_rgb": sphere_rgb,
            "sphere_gt_sem": sphere_sem,
        }
