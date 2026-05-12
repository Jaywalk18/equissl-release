"""Supplementary: Feature PCA visualization.

Renders the decoder output of our EquiSSL model on a Stanford2D3D val
sample. Per-vertex features (40962 tokens at proj_rank=6) are projected
to 3D via PCA and mapped to RGB. Rendered on the rank-7 icosphere mesh
for visual continuity with Figure 1.

This demonstrates that the learned representation is spatially coherent
(neighbouring vertices have similar PCA colours) while encoding
semantic structure (walls vs. floor vs. clutter form distinct clusters
in the PCA space).

Output: figures/figs/supp_feature_pca.{png,pdf}
"""
import os, sys, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

SAMPLE_IDX = 28
SPLIT = "val"
CFG = "configs/pretrain_v8_large.yaml"
CKPT = "outputs/rpe_ablation_c4_v2/best_model.pth"
OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1
RENDER_RANK = 7

# ---- Icosphere refs ----
ref = IcoSphereRef("vertex")
img_normals   = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)
proj_normals  = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
render_verts  = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
render_faces  = np.asarray(ref.get_icosphere(RENDER_RANK, False).faces,
                           dtype=np.int64)

_, idx_render_from_proj = cKDTree(proj_normals).query(render_verts, k=1)


# ---- Load sample ----
print(f"Loading sample: {SPLIT} idx={SAMPLE_IDX}")
ds = Stanford2D3DSeg(
    split=SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"],
    num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
rgb_in = ds[SAMPLE_IDX]["sphere_rgb"].unsqueeze(0).cuda()


# ---- Build + load EquiSSL ----
print(f"Loading EquiSSL ckpt: {CKPT}")
enc = SphericalEncoder(
    img_rank=IMG_RANK, node_type=mc["node_type"],
    embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
    bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
    abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
    rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True)

model = EquiSSLSegUNet(
    encoder=enc, num_classes=14,
    dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
    dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
    abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
    rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda().eval()


# ---- Forward: capture decoder output (pre-seg-head features) ----
print("Running forward pass...")
with torch.no_grad():
    enc_result = model.encoder(rgb_in, mask=None, return_enc_outs=True)
    dec_out = model.decoder(enc_result["patch"], enc_result["enc_outs"])
features = dec_out[0].cpu().numpy().astype(np.float32)
print(f"  decoder features: {features.shape} (N_proj, D)")

# ---- PCA to 3D -> RGB ----
print("Running PCA(3)...")
pca = PCA(n_components=3, random_state=42)
pca_3d = pca.fit_transform(features)  # (N_proj, 3)
print(f"  explained var ratio: {pca.explained_variance_ratio_}")
print(f"  cumulative: {pca.explained_variance_ratio_.cumsum()}")

# Normalize per-component to [0,1] robustly (2nd/98th percentile)
rgb = np.empty_like(pca_3d)
for c in range(3):
    lo, hi = np.percentile(pca_3d[:, c], [2, 98])
    rgb[:, c] = np.clip((pca_3d[:, c] - lo) / (hi - lo + 1e-8), 0, 1)

# Upsample to render rank 7 (nearest neighbour)
render_rgb = rgb[idx_render_from_proj]  # (N_render, 3)


# ---- Render ----
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])


def render_sphere(ax, face_rgb, edge_alpha=0.0):
    vf = render_verts[render_faces]
    centroids = vf.mean(axis=1)
    nrm = np.cross(vf[:, 1]-vf[:, 0], vf[:, 2]-vf[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    flip = (centroids * nrm).sum(axis=1) < 0
    nrm[flip] = -nrm[flip]
    visible = (nrm @ VIEW) > -0.05
    vf_vis = vf[visible]
    n_vis = nrm[visible]

    # Per-face colour = mean of its 3 vertex colours (indexed via faces)
    face_rgb_vis = face_rgb[visible]
    shade = 0.75 + 0.25 * np.clip(n_vis @ LIGHT, 0, 1)
    shaded = (face_rgb_vis * shade[:, None]).clip(0, 1)

    ax.add_collection3d(Poly3DCollection(
        vf_vis, facecolors=shaded,
        edgecolors=(0.3, 0.3, 0.3, edge_alpha),
        linewidths=0.15, antialiased=False))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()


# Per-face RGB: average over each face's 3 vertex colours
face_rgb = render_rgb[render_faces].mean(axis=1)

fig = plt.figure(figsize=(6.5, 6.0), dpi=200)
ax = fig.add_subplot(111, projection="3d")
render_sphere(ax, face_rgb)
ax.set_title(
    f"Learned decoder features (EquiSSL)\n"
    f"PC1/2/3 → RGB, cumulative var = "
    f"{pca.explained_variance_ratio_.sum()*100:.1f}%",
    fontsize=12.5, pad=6)

plt.savefig(f"{OUT}/supp_feature_pca.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/supp_feature_pca.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT}/supp_feature_pca.{{png,pdf}}")
print(f"  sample: {SPLIT} idx={SAMPLE_IDX}")
print(f"  PCA explained variance: "
      f"{pca.explained_variance_ratio_.round(3).tolist()} "
      f"(cum {pca.explained_variance_ratio_.sum()*100:.1f}%)")
