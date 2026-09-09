"""
Structured3D dataset for EquiSSL pretraining.
Loads ERP panoramic images and resamples to icosphere.
No labels needed for SSL — only RGB.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Dict

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import IcoSphereRef, asSpherical

from ..utils.sphere import erp_to_icosphere_grid


class Structured3DSSL(Dataset):
    """
    Structured3D dataset for self-supervised pretraining on icosphere.
    Only loads RGB panoramas — no depth/semantic labels needed.
    """

    def __init__(
        self,
        root_dir: str = "${STRUCTURED3D_PATH}_new/Structured3D",
        img_rank: int = 7,
        node_type: str = "vertex",
        split: str = "train",
        image_key: str = "rgb_rawlight",
        normalize_mean: float = 0.5,
        normalize_std: float = 0.225,
        color_augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_rank = img_rank
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.color_augment = color_augment
        self.image_key = image_key

        self.icosphere_ref = IcoSphereRef(node_type)
        self.sphere_grid = erp_to_icosphere_grid(self.icosphere_ref, img_rank)

        self.image_paths = self._collect_panoramas(split)
        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]
        print(f"Structured3DSSL: found {len(self.image_paths)} panoramas ({split})")

    def _collect_panoramas(self, split: str):
        scenes = sorted(os.listdir(self.root_dir))
        scenes = [s for s in scenes if s.startswith("scene_")]

        num_scenes = len(scenes)
        if split == "train":
            scenes = scenes[: int(num_scenes * 0.9)]
        elif split == "val":
            scenes = scenes[int(num_scenes * 0.9):]
        else:
            raise ValueError(f"Unknown split: {split}")

        paths = []
        for scene in scenes:
            rendering_dir = os.path.join(self.root_dir, scene, "2D_rendering")
            if not os.path.isdir(rendering_dir):
                continue
            for room in os.listdir(rendering_dir):
                img_path = os.path.join(
                    rendering_dir, room, "panorama", "full", f"{self.image_key}.png"
                )
                if os.path.isfile(img_path):
                    paths.append(img_path)
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}, using random index")
            return self.__getitem__(np.random.randint(len(self)))

        img = np.array(img, dtype=np.float32) / 255.0

        if self.color_augment and np.random.rand() < 0.5:
            img = self._color_jitter(img)

        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        sphere_rgb = F.grid_sample(
            img_tensor,
            self.sphere_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sphere_rgb = sphere_rgb.squeeze(0).squeeze(1).T

        normalized = (sphere_rgb - self.normalize_mean) / self.normalize_std

        return {
            "sphere_rgb": normalized,
        }

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 1)

        if np.random.rand() < 0.5:
            gray = img.mean(axis=-1, keepdims=True)
            saturation = np.random.uniform(0.5, 1.5)
            img = np.clip(gray + saturation * (img - gray), 0, 1)

        return img
