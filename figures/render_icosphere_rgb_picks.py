"""Render icosphere RGB views for multiple Stanford2D3D val samples.

Quick browser for the author to pick a non-Japanese-looking scene as
the replacement for `image.png` / `figures/figs/real_icosphere_rgb.png`
(currently val idx=28 which has a corridor-with-wooden-doors look the
author dislikes).

Re-uses the supersample + antialiased + LANCZOS pipeline from
render_clean_spheres.py: SEG_RANK=6, AA=True, 2× supersample, then
PIL-LANCZOS downsample to 1024×1024. Source ERP polar caps are also
cropped so the bilinear-sampling near the poles doesn't propagate
black bands onto the visible part of the disc.

Run:
    GPU_ID=0 python figures/render_icosphere_rgb_picks.py
"""
import os
import sys
import io

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from trimesh_utils import IcoSphereRef, asSpherical

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]


# --------- Config ---------
CANDIDATES = [
    ("val",  0),    # open conference area, bright
    ("val", 13),    # lecture hall with podium
    ("val", 16),    # wooden auditorium / bookshelves (Western library)
    ("val", 22),    # clutter-rich large hall
]
SPLIT_TO_DIR_NAME = {"val": "val"}

CFG          = "configs/pretrain_v8_large.yaml"
RENDER_RANK  = 6     # matches render_clean_spheres.py AA pipeline
CANVAS_PX    = 1024
SUPERSAMPLE  = 2
ZOOM         = 1.55  # disc fills ~85% of canvas

OUT_LOCAL    = "figures/figs"
OUT_PPT      = "figures/ppt_assets/server_delivered"

# View / lighting (matches render_clean_spheres.py for visual consistency)
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW  = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])


# --------- Init ---------
with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

ref = IcoSphereRef("vertex")
ico_vis  = ref.get_icosphere(RENDER_RANK, False)
verts_vis = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
faces_vis = np.asarray(ico_vis.faces, dtype=np.int64)
print(f"Render icosphere rank {RENDER_RANK}: {len(verts_vis)} verts, "
      f"{len(faces_vis)} faces")

# Precompute visibility + shading (camera + light fixed)
vf = verts_vis[faces_vis]
centroids = vf.mean(axis=1)
nrm = np.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
flip = (centroids * nrm).sum(axis=1) < 0
nrm[flip] = -nrm[flip]
visible   = (nrm @ VIEW) > -0.05
vf_vis    = vf[visible]
shade_vis = 0.75 + 0.25 * np.clip(nrm[visible] @ LIGHT, 0, 1)

# Sample ERP → per-vertex RGB
sph = asSpherical(verts_vis)
phi_deg   = sph[:, 1]
theta_deg = sph[:, 2]


def erp_to_vertex_rgb(erp_rgb_uint8):
    """Bilinear sample ERP onto the icosphere vertices."""
    H, W = erp_rgb_uint8.shape[:2]
    v = (phi_deg / 180.0) * (H - 1)
    u = ((theta_deg + 180.0) / 360.0) * (W - 1)
    v = np.clip(v, 0, H - 1).astype(np.float32)
    u = np.clip(u, 0, W - 1).astype(np.float32)
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, W - 1);    v1 = np.minimum(v0 + 1, H - 1)
    du = (u - u0)[:, None]; dv = (v - v0)[:, None]
    f = erp_rgb_uint8.astype(np.float32)
    c00 = f[v0, u0]; c01 = f[v0, u1]
    c10 = f[v1, u0]; c11 = f[v1, u1]
    out = ((1 - du) * (1 - dv) * c00 + du * (1 - dv) * c01 +
           (1 - du) * dv * c10 + du * dv * c11) / 255.0
    return out


def crop_polar_caps(erp):
    """Remove pure-black polar caps before sampling — otherwise the
    bilinear interpolator drags black into the highest-latitude vertices."""
    rm = erp.mean(axis=(1, 2))
    v = np.where(rm > 5)[0]
    if not len(v):
        return erp
    top, bot = max(0, v[0] - 4), min(erp.shape[0] - 1, v[-1] + 4)
    return erp[top:bot + 1]


def render_clean_icosphere_rgb(face_rgb, out_path,
                               canvas_px=CANVAS_PX, supersample=SUPERSAMPLE,
                               zoom=ZOOM):
    shaded = (face_rgb[visible] * shade_vis[:, None]).clip(0, 1)
    super_px = canvas_px * supersample
    figsize = (super_px / 100.0, super_px / 100.0)
    fig = plt.figure(figsize=figsize, dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.add_collection3d(Poly3DCollection(
        vf_vis, facecolors=shaded, edgecolors=shaded,
        linewidths=0.4, antialiased=True))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    try:
        ax.set_box_aspect([1, 1, 1], zoom=zoom)
    except TypeError:
        ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != (canvas_px, canvas_px):
        img = img.resize((canvas_px, canvas_px), Image.LANCZOS)
    img.save(out_path, format="PNG", optimize=True)


def face_colors_from_verts(vrgb):
    return vrgb[faces_vis].mean(axis=1).clip(0, 1)


# --------- Per-sample dataset (matches render_real_figs.py) ---------
ds_kwargs = dict(
    data_dir="${STANFORD2D3D_PATH}", img_rank=mc["img_rank"],
    node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])

_datasets = {}
def _get_ds(split):
    if split not in _datasets:
        _datasets[split] = Stanford2D3DSeg(split=split, **ds_kwargs)
    return _datasets[split]


os.makedirs(OUT_LOCAL, exist_ok=True)
os.makedirs(OUT_PPT, exist_ok=True)
manifest = []

for split, idx in CANDIDATES:
    ds = _get_ds(split)
    rgb_path, _ = ds.samples[idx]
    print(f"\n=== {split} idx={idx} : {os.path.basename(rgb_path)} ===")
    erp = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    erp = crop_polar_caps(erp)
    print(f"  ERP cropped to {erp.shape}")
    vert_rgb = erp_to_vertex_rgb(erp)
    face_rgb = face_colors_from_verts(vert_rgb)

    tag = f"{split}{idx}"
    paths = [
        os.path.join(OUT_LOCAL, f"icosphere_rgb_{tag}.png"),
        os.path.join(OUT_PPT,   f"icosphere_rgb_{tag}.png"),
    ]
    render_clean_icosphere_rgb(face_rgb, paths[0])
    # Copy to the second path so we don't render twice
    import shutil
    shutil.copy2(paths[0], paths[1])
    for p in paths:
        sz_kb = os.path.getsize(p) / 1024
        manifest.append((p, sz_kb))
        print(f"  wrote {p} ({sz_kb:.1f} KB)")


print("\n=== Manifest ===")
for p, kb in manifest:
    print(f"  {p}  ({kb:.1f} KB)")
