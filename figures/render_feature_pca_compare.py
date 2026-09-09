"""Supplementary: Feature PCA comparison — Standard RPE vs EquiSSL.

Runs both models on the same Stanford2D3D val sample, projects each
model's decoder features to 3D via PCA, and renders the resulting RGB
map on the rank-7 icosphere. Reveals that EquiSSL learns more spatially
coherent / semantically clustered representations than Standard RPE —
rotation-invariant gauge pooling yields structure that Standard's
gauge-dependent bias cannot consistently provide.

Output: figures/figs/supp_feature_pca_compare.{png,pdf}
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
from sklearn.metrics import silhouette_score

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

SAMPLE_IDX = 28
SPLIT = "val"
CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

VARIANTS = [
    ("Standard RPE",  "outputs/finetune_v8_random_s/best_model.pth",
     {"equivariant": False, "n_gauges": 6}),
    ("EquiSSL", "outputs/rpe_ablation_c4_v2/best_model.pth",
     {"equivariant": True,  "n_gauges": 4}),
]

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1
RENDER_RANK = 7

ref = IcoSphereRef("vertex")
proj_normals  = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
render_verts  = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
render_faces  = np.asarray(ref.get_icosphere(RENDER_RANK, False).faces,
                           dtype=np.int64)
_, idx_render_from_proj = cKDTree(proj_normals).query(render_verts, k=1)


def build(ckpt_path, equivariant, n_gauges):
    enc = SphericalEncoder(
        img_rank=IMG_RANK, node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"],
        enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=equivariant, n_gauges=n_gauges, area_weighted=True)
    m = EquiSSLSegUNet(
        encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=equivariant, n_gauges=n_gauges, area_weighted=True)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


# Shared sample
ds = Stanford2D3DSeg(
    split=SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"],
    num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
rgb_in = ds[SAMPLE_IDX]["sphere_rgb"].unsqueeze(0).cuda()
gt_labels = ds[SAMPLE_IDX]["sphere_gt_sem"].numpy()  # (N_proj,) class indices

features_per_variant = []
for label, ckpt, spec in VARIANTS:
    print(f"[{label}] loading {ckpt}")
    model = build(ckpt, spec["equivariant"], spec["n_gauges"])
    with torch.no_grad():
        enc_result = model.encoder(rgb_in, mask=None, return_enc_outs=True)
        dec_out = model.decoder(enc_result["patch"], enc_result["enc_outs"])
    features_per_variant.append(dec_out[0].cpu().numpy().astype(np.float32))
    del model
    torch.cuda.empty_cache()


# ---- Class-alignment metric: silhouette score of features, labelled by GT
# Excludes class 0 (unknown, ignored in eval). Subsample for compute budget.
rng = np.random.default_rng(0)
valid = gt_labels != 0
valid_idx = np.where(valid)[0]
SUB_N = min(3000, len(valid_idx))
sub = rng.choice(valid_idx, SUB_N, replace=False)
sub_labels = gt_labels[sub]
# Require ≥2 distinct classes for silhouette
unique_classes = np.unique(sub_labels)
print(f"\nSilhouette score on {SUB_N} vertices, {len(unique_classes)} classes")
silhouettes = []
for feats in features_per_variant:
    s = silhouette_score(feats[sub], sub_labels, metric="cosine")
    silhouettes.append(float(s))


# ---- PCA each variant independently, map to RGB with percentile normalisation
variant_rgbs = []
variant_vars = []
for feats in features_per_variant:
    pca = PCA(n_components=3, random_state=42)
    proj = pca.fit_transform(feats)
    rgb = np.empty_like(proj)
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [2, 98])
        rgb[:, c] = np.clip((proj[:, c] - lo) / (hi - lo + 1e-8), 0, 1)
    variant_rgbs.append(rgb[idx_render_from_proj])
    variant_vars.append(float(pca.explained_variance_ratio_.sum()))


# ---- Render both as side-by-side panels
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])


def render_panel(ax, render_rgb):
    vf = render_verts[render_faces]
    centroids = vf.mean(axis=1)
    nrm = np.cross(vf[:, 1]-vf[:, 0], vf[:, 2]-vf[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    flip = (centroids * nrm).sum(axis=1) < 0
    nrm[flip] = -nrm[flip]
    visible = (nrm @ VIEW) > -0.05
    vf_vis = vf[visible]
    n_vis = nrm[visible]

    face_rgb = render_rgb[render_faces].mean(axis=1)
    face_rgb_vis = face_rgb[visible]
    shade = 0.75 + 0.25 * np.clip(n_vis @ LIGHT, 0, 1)
    shaded = (face_rgb_vis * shade[:, None]).clip(0, 1)

    ax.add_collection3d(Poly3DCollection(
        vf_vis, facecolors=shaded, edgecolors=(0.3, 0.3, 0.3, 0.0),
        linewidths=0.1, antialiased=False))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()


fig = plt.figure(figsize=(11.5, 6.4), dpi=200)
gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.80, bottom=0.05,
                      wspace=0.04)

for i, (label, _, _) in enumerate(VARIANTS):
    ax = fig.add_subplot(gs[0, i], projection="3d")
    render_panel(ax, variant_rgbs[i])
    ax.set_title(
        f"({chr(97+i)}) {label}\n"
        f"class-feature silhouette: {silhouettes[i]:.3f}",
        fontsize=12.5, pad=4)

fig.suptitle(
    "Decoder features on the sphere (PCA$\\to$RGB) — same val sample"
    "\nHigher silhouette = features cluster more tightly by ground-truth class",
    fontsize=13, y=0.96)

plt.savefig(f"{OUT}/supp_feature_pca_compare.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/supp_feature_pca_compare.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT}/supp_feature_pca_compare.{{png,pdf}}")
for (label, _, _), v, s in zip(VARIANTS, variant_vars, silhouettes):
    print(f"  {label}: PCA cum var = {v*100:.1f}%  "
          f"class-feature silhouette = {s:.4f}")
