"""Render 4 icosphere-segmentation PNGs for paper Figure 1 motivation:
    seg_std_upright.png, seg_std_rot90.png  — SphereUFormer (published baseline, -53% drop)
    seg_ge_upright.png, seg_ge_rot90.png    — EquiSSL (ours, -2.5% drop)

Contrast: SphereUFormer's original checkpoint exhibits severe rotation-induced
degradation (gauge-dependent RPE); our EquiSSL remains visually consistent.

All renders: val split idx=28 (matches Figure 5 seg_comparison), azim=30°, elev=20°.
Rotation: fixed 90° around world y-axis (tips up-axis sideways), applied as
icosphere node permutation (same protocol as eval_pose35).
"""
import os, sys, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

# Ours
from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
# SphereUFormer baseline
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

SAMPLE_IDX = 28
SPLIT = "val"
CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"       # SphereUFormer original
CKPT_OURS     = "outputs/rpe_ablation_c4_v2/best_model.pth"           # EquiSSL-C₄ seed 42 (3-seed mean 67.31)
print("=== Contrastive render: SphereUFormer baseline vs our EquiSSL ===")
print(f"  Baseline: {CKPT_BASELINE}")
print(f"  Ours:     {CKPT_OURS}")

S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60],  [0.90, 0.75, 0.25],  [0.20, 0.60, 0.80],
    [0.55, 0.35, 0.20],  [0.85, 0.85, 0.95],  [0.95, 0.45, 0.45],
    [0.75, 0.55, 0.75],  [0.40, 0.40, 0.60],  [0.95, 0.75, 0.55],
    [0.55, 0.75, 0.45],  [0.85, 0.35, 0.60],  [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85],  [0.30, 0.55, 0.85],
], dtype=np.float32)

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK
RENDER_RANK = 7

# ---- Icosphere references ----
ref = IcoSphereRef("vertex")
img_normals   = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)    # 163842
proj_normals  = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)   # 40962
render_verts  = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
render_faces  = np.asarray(ref.get_icosphere(RENDER_RANK, False).faces, dtype=np.int64)

# ---- Fixed 90° pitch (rotation around world y-axis) ----
ANGLE = np.pi / 2
R_pitch = np.array([[ np.cos(ANGLE), 0, np.sin(ANGLE)],
                    [ 0,             1, 0            ],
                    [-np.sin(ANGLE), 0, np.cos(ANGLE)]], dtype=np.float32)
print(f"\nR (90° y-axis pitch):\n{R_pitch}\n")

img_perm_np  = compute_rotation_permutation(img_normals, R_pitch)
proj_perm_np = compute_rotation_permutation(proj_normals, R_pitch)
img_perm  = torch.tensor(img_perm_np,  dtype=torch.long).cuda()
proj_perm = torch.tensor(proj_perm_np, dtype=torch.long)
print(f"img_perm identity-fraction: {(img_perm_np == np.arange(len(img_perm_np))).mean()*100:.1f}%")

# ---- Load sample ----
# Our dataset uses in_scale_factor=2 so labels live at proj_rank (rank 6).
# SphereUFormer expects output at img_rank (rank 7). We'll handle both cases at render time.
ds_ours = Stanford2D3DSeg(split=SPLIT,
    data_dir="${STANFORD2D3D_PATH}", img_rank=IMG_RANK,
    node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
rgb_cpu = ds_ours[SAMPLE_IDX]["sphere_rgb"].unsqueeze(0)  # (1, N_img, 3), already normalized
rgb = rgb_cpu.cuda()
print(f"Sample: {os.path.basename(ds_ours.samples[SAMPLE_IDX][0])}  shape={rgb.shape}")

# ---- Build our EquiSSL model ----
def build_ours():
    encoder = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=4, area_weighted=True,
    )
    model = EquiSSLSegUNet(
        encoder=encoder, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths", [2,2,2,2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16,16,8,4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=4, area_weighted=True,
    )
    ckpt = torch.load(CKPT_OURS, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.cuda().eval()

# ---- Build SphereUFormer baseline ----
def build_baseline():
    model = SphereUFormer(
        img_rank=IMG_RANK, node_type="vertex",
        in_channels=3, out_channels=14,
        in_scale_factor=2, num_scales=4,
        win_size_coef=2, enc_depths=2, dec_depths=2, bottleneck_depth=2,
        d_head_coef=2,
        enc_num_heads=[2, 4, 8, 16], dec_num_heads=[16, 16, 8, 4],
        abs_pos_enc_in=True, abs_pos_enc=True,
        rel_pos_bias=True, rel_pos_bias_size=7, rel_pos_init_variance=1.0,
        downsample="center", upsample="interpolate",
        use_checkpoint=True,
    )
    ckpt = torch.load(CKPT_BASELINE, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.cuda().eval()

# ---- Helpers ----
def upsample_from_rank(pred, src_normals):
    """NN upsample from src_normals resolution to RENDER_RANK."""
    tree = cKDTree(src_normals)
    _, nn_idx = tree.query(render_verts, k=1)
    return pred[nn_idx]

def render_icosphere(verts, faces, face_colors, out_path, azim=30, elev=20,
                     final_px=1500, ss=2.4, dpi=300):
    size_px = int(final_px * ss)
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    verts_faces = verts[faces]
    centroids = verts_faces.mean(axis=1)
    n_raw = np.cross(verts_faces[:,1] - verts_faces[:,0], verts_faces[:,2] - verts_faces[:,0])
    n_raw /= np.linalg.norm(n_raw, axis=1, keepdims=True) + 1e-9
    flip = (n_raw * centroids).sum(axis=1) < 0
    n_raw[flip] = -n_raw[flip]
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    view_dir = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    visible = (n_raw @ view_dir) > 0.01
    vf = verts_faces[visible]; nf = n_raw[visible]; fc = face_colors[visible]
    light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = 0.55 + 0.45 * np.clip(nf @ light, 0, 1)
    shaded = (fc[:, :3] * shade[:, None]).clip(0, 1)
    coll = Poly3DCollection(vf, facecolors=shaded, edgecolors=shaded,
                            linewidths=0.5, antialiased=False)
    ax.add_collection3d(coll)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_facecolor("none"); fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    from PIL import Image
    im = Image.open(out_path).convert("RGBA")
    bbox = im.getbbox()
    if bbox is not None:
        w, h = im.size
        mx = int((bbox[2]-bbox[0])*0.02); my = int((bbox[3]-bbox[1])*0.02)
        bbox = (max(0, bbox[0]-mx), max(0, bbox[1]-my),
                min(w, bbox[2]+mx), min(h, bbox[3]+my))
        im = im.crop(bbox)
        W = max(im.size)
        sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        sq.paste(im, ((W - im.size[0])//2, (W - im.size[1])//2))
        if W > final_px:
            sq = sq.resize((final_px, final_px), Image.LANCZOS)
        sq.save(out_path, optimize=True)
    print(f"Saved {out_path}")

def face_colors_from_verts(vrgb, faces):
    return vrgb[faces].mean(axis=1)

verts_render_upright = render_verts
verts_render_rotated = (R_pitch @ render_verts.T).T

# ---- Run both models × both conditions ----
@torch.no_grad()
def predict(model, rgb_in):
    return model(rgb_in).argmax(dim=-1).cpu().numpy()[0]

print("\n--- SphereUFormer baseline (Standard RPE, -53% drop) ---")
baseline = build_baseline()
pred_up_b  = predict(baseline, rgb)
rgb_rot    = apply_rotation_to_features(rgb, img_perm)
pred_rot_b = predict(baseline, rgb_rot)
# SphereUFormer outputs at img_rank=7, so predictions are already at RENDER_RANK
assert len(pred_up_b) == len(render_verts), f"baseline pred len {len(pred_up_b)} vs render {len(render_verts)}"
pred_up_render_b  = pred_up_b
pred_rot_render_b = pred_rot_b
print(f"  upright classes: {sorted(set(pred_up_b.tolist()))[:10]}")
print(f"  rotated classes: {sorted(set(pred_rot_b.tolist()))[:10]}")
# Equivariance agreement (at render/img rank, which == proj rank for SphereUFormer)
img_perm_ren = compute_rotation_permutation(render_verts, R_pitch)
expected_b = pred_up_b[img_perm_ren]
agree_b = (pred_rot_b == expected_b).mean() * 100
print(f"  equivariance agreement: {agree_b:.2f}%")

render_icosphere(verts_render_upright, render_faces,
                 face_colors_from_verts(S2D3D_COLORS[pred_up_render_b], render_faces),
                 f"{OUT}/seg_std_upright.png")
render_icosphere(verts_render_rotated, render_faces,
                 face_colors_from_verts(S2D3D_COLORS[pred_rot_render_b], render_faces),
                 f"{OUT}/seg_std_rot90.png")
del baseline; torch.cuda.empty_cache()

print("\n--- Our EquiSSL-C4 (3-seed mean val 67.31, drop -1.7%) ---")
ours = build_ours()
pred_up_o  = predict(ours, rgb)
pred_rot_o = predict(ours, rgb_rot)
# Our model outputs at proj_rank=6, need to upsample to render_rank=7
pred_up_render_o  = upsample_from_rank(pred_up_o,  proj_normals)
pred_rot_render_o = upsample_from_rank(pred_rot_o, proj_normals)
print(f"  upright classes: {sorted(set(pred_up_o.tolist()))[:10]}")
print(f"  rotated classes: {sorted(set(pred_rot_o.tolist()))[:10]}")
expected_o = pred_up_o[proj_perm.numpy()]
agree_o = (pred_rot_o == expected_o).mean() * 100
print(f"  equivariance agreement: {agree_o:.2f}%")

render_icosphere(verts_render_upright, render_faces,
                 face_colors_from_verts(S2D3D_COLORS[pred_up_render_o], render_faces),
                 f"{OUT}/seg_ge_upright.png")
render_icosphere(verts_render_rotated, render_faces,
                 face_colors_from_verts(S2D3D_COLORS[pred_rot_render_o], render_faces),
                 f"{OUT}/seg_ge_rot90.png")
del ours; torch.cuda.empty_cache()

print("\n==== Output summary ====")
for f in ["seg_std_upright.png", "seg_std_rot90.png",
          "seg_ge_upright.png",  "seg_ge_rot90.png"]:
    p = os.path.join(OUT, f)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
print(f"\nEquivariance agreement:")
print(f"  SphereUFormer (published): {agree_b:.2f}%   <- 'Standard' RPE in paper Figure 1")
print(f"  EquiSSL (ours):          {agree_o:.2f}%   <- 'EquiSSL' in paper Figure 1")
