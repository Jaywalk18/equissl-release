"""
Structured3D dataset for semantic segmentation on icosphere.

Native NYU40 40-class taxonomy (+ class 0 = unlabeled / ignored).
Matches SphereUFormer / PanoFormer / EGFormer evaluation protocol for
direct comparison with published S3D SOTA numbers.
"""
import os, cv2, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
from trimesh_utils import IcoSphereRef

from ..utils.sphere import erp_to_icosphere_grid


# --- NYU40 RGB palette (Structured3D's semantic.png encoding) ---
# idx 0 is "unlabeled"; idx 1-40 are the standard NYU40 semantic classes.
NYU40_RGB = {
    0: (0, 0, 0),            # unlabeled
    1: (174, 199, 232),      # wall
    2: (152, 223, 138),      # floor
    3: (31, 119, 180),       # cabinet
    4: (255, 187, 120),      # bed
    5: (188, 189, 34),       # chair
    6: (140, 86, 75),        # sofa
    7: (255, 152, 150),      # table
    8: (214, 39, 40),        # door
    9: (197, 176, 213),      # window
    10: (148, 103, 189),     # bookshelf
    11: (196, 156, 148),     # picture
    12: (23, 190, 207),      # counter
    13: (178, 76, 76),       # blinds
    14: (247, 182, 210),     # desk
    15: (66, 188, 102),      # shelves
    16: (219, 219, 141),     # curtain
    17: (140, 57, 197),      # dresser
    18: (202, 185, 52),      # pillow
    19: (51, 176, 203),      # mirror
    20: (92, 193, 61),       # floor_mat
    21: (78, 71, 183),       # clothes
    22: (172, 114, 82),      # ceiling
    23: (255, 127, 14),      # books
    24: (91, 163, 138),      # refrigerator
    25: (153, 98, 156),      # television
    26: (140, 153, 101),     # paper
    27: (158, 218, 229),     # towel
    28: (100, 125, 154),     # shower_curtain
    29: (178, 127, 135),     # box
    30: (120, 185, 128),     # whiteboard
    31: (146, 111, 194),     # person
    32: (44, 160, 44),       # nightstand
    33: (112, 128, 144),     # toilet
    34: (96, 207, 209),      # sink
    35: (227, 119, 194),     # lamp
    36: (213, 92, 176),      # bathtub
    37: (94, 106, 211),      # bag
    38: (82, 84, 163),       # otherstructure
    39: (100, 85, 144),      # otherfurniture
    40: (148, 156, 196),     # otherprop
}


def _build_rgb_lookup() -> np.ndarray:
    """256^3 lookup: R*65536 + G*256 + B → NYU40 class id."""
    lut = np.zeros(256 * 256 * 256, dtype=np.uint8)
    for cid, rgb in NYU40_RGB.items():
        idx = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        lut[idx] = cid
    return lut


class Structured3DSeg(Dataset):
    """Structured3D panorama segmentation with native NYU40 40-class taxonomy.

    Splits (official scene-level):
      train  : scene_00000 - scene_02999 (~18k panoramas)
      val    : scene_03000 - scene_03249 (~1.8k panoramas)
      test   : scene_03250 - scene_03499 (~1.7k panoramas)

    NUM_CLASSES = 41 (0=unlabeled + 40 NYU classes); class 0 is ignore_index.
    """

    NUM_CLASSES = 41

    def __init__(
        self,
        data_dir: str = "${STRUCTURED3D_PATH}_new/Structured3D",
        split: str = "train",
        img_rank: int = 7,
        node_type: str = "vertex",
        num_scales: int = 4,
        in_scale_factor: int = 2,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.225,
        lighting: str = "rawlight",   # rawlight / warmlight / coldlight
        rendering: str = "full",       # full / simple / empty
        augment: bool = False,
        max_samples: int = -1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.lighting = lighting
        self.rendering = rendering
        self.augment = augment and (split == "train")

        if split == "train":
            lo, hi = 0, 3000
        elif split == "val":
            lo, hi = 3000, 3250
        elif split == "test":
            lo, hi = 3250, 3500
        else:
            raise ValueError(f"Unknown split: {split}")

        self.icosphere_ref = IcoSphereRef(node_type)
        self.sphere_grid = erp_to_icosphere_grid(self.icosphere_ref, img_rank)
        proj_rank = img_rank - 1 if in_scale_factor == 2 else img_rank
        self.label_grid = erp_to_icosphere_grid(self.icosphere_ref, proj_rank)

        self.rgb_lut = _build_rgb_lookup()

        self.samples: List[Tuple[str, str]] = []
        for scene_idx in range(lo, hi):
            scene_dir = os.path.join(data_dir, f"scene_{scene_idx:05d}")
            rendering_root = os.path.join(scene_dir, "2D_rendering")
            if not os.path.isdir(rendering_root):
                continue
            for room in sorted(os.listdir(rendering_root)):
                pano_dir = os.path.join(rendering_root, room, "panorama", rendering)
                rgb_path = os.path.join(pano_dir, f"rgb_{lighting}.png")
                sem_path = os.path.join(pano_dir, "semantic.png")
                if os.path.isfile(rgb_path) and os.path.isfile(sem_path):
                    self.samples.append((rgb_path, sem_path))
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        print(f"Structured3DSeg: {len(self.samples)} panoramas ({split}, "
              f"scenes {lo}-{hi-1}, lighting={lighting}, rendering={rendering}, "
              f"augment={self.augment})")

    def __len__(self) -> int:
        return len(self.samples)

    def _semantic_to_labels(self, sem_bgr: np.ndarray) -> np.ndarray:
        """BGR semantic image → (H, W) NYU40 class ids."""
        r = sem_bgr[..., 2].astype(np.int64)
        g = sem_bgr[..., 1].astype(np.int64)
        b = sem_bgr[..., 0].astype(np.int64)
        flat = r * 65536 + g * 256 + b
        return self.rgb_lut[flat].astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, sem_path = self.samples[idx]
        try:
            rgb_bgr = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            sem_bgr = cv2.imread(sem_path)
        except Exception as e:
            print(f"Error loading {rgb_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        sem_labels = self._semantic_to_labels(sem_bgr)

        if self.augment:
            H, W = rgb.shape[:2]
            # Yaw rotation: random horizontal shift (ERP is yaw-equivariant via roll)
            shift = np.random.randint(0, W)
            rgb = np.roll(rgb, shift, axis=1)
            sem_labels = np.roll(sem_labels, shift, axis=1)
            # Horizontal flip with 50% probability
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                sem_labels = np.ascontiguousarray(sem_labels[:, ::-1])

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
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
