"""Render 3 figures for paper pipeline (val split idx=28, matches Figure 5):
  1) real_panorama.png   — ERP RGB
  2) real_icosphere_rgb.png  — icosphere (rank 7) with panorama texture
  3) real_icosphere_seg.png  — icosphere with EquiSSL segmentation prediction
All icosphere renders: azim=30°, elev=20°, transparent bg.
"""
import os, sys, yaml, cv2
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import erp_to_icosphere_grid
from trimesh_utils import IcoSphereRef

SAMPLE_IDX = 28
SPLIT = "val"
CKPT = "outputs/rpe_ablation_c4_v2/best_model.pth"
CFG  = "configs/pretrain_v8_large.yaml"
OUT  = "figures/figs"
os.makedirs(OUT, exist_ok=True)

# Stanford2D3D 14-class colormap (RGB 0-1). Class 0 = unknown (gray).
S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60],  # 0 unknown
    [0.90, 0.75, 0.25],  # 1 beam    (amber)
    [0.20, 0.60, 0.80],  # 2 board   (teal)
    [0.55, 0.35, 0.20],  # 3 bookcase(brown)
    [0.85, 0.85, 0.95],  # 4 ceiling (very light blue)
    [0.95, 0.45, 0.45],  # 5 chair   (coral)
    [0.75, 0.55, 0.75],  # 6 clutter (mauve)
    [0.40, 0.40, 0.60],  # 7 column  (slate)
    [0.95, 0.75, 0.55],  # 8 door    (peach)
    [0.55, 0.75, 0.45],  # 9 floor   (sage green)
    [0.85, 0.35, 0.60],  # 10 sofa   (pink)
    [0.70, 0.50, 0.25],  # 11 table  (tan)
    [0.65, 0.80, 0.85],  # 12 wall   (pale teal)
    [0.30, 0.55, 0.85],  # 13 window (bright blue)
], dtype=np.float32)

# --- Load config & data ---
with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

ds_kwargs = dict(
    data_dir="${STANFORD2D3D_PATH}", img_rank=mc["img_rank"],
    node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"],
)
test_ds = Stanford2D3DSeg(split=SPLIT, **ds_kwargs)
rgb_path, sem_path = test_ds.samples[SAMPLE_IDX]
print(f"Sample {SAMPLE_IDX}: {os.path.basename(rgb_path)}")

# --- Fig 1: ERP panorama (save the raw RGB, crop polar black caps) ---
erp = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
print(f"ERP source shape: {erp.shape}")
# Stanford2D3D ERP carries pure-black caps at the top/bottom corresponding
# to lat ≈ ±90° (Matterport-style cameras cannot see straight up/down).
# Crop tight to non-black content + small margin, so the saved panorama
# is a clean visualisation strip rather than a stretched canvas with
# wasted black bands.
_row_mean = erp.mean(axis=(1, 2))
_valid    = np.where(_row_mean > 5)[0]
if len(_valid):
    _top, _bot = _valid[0], _valid[-1]
    _margin = 4
    _top = max(0, _top - _margin)
    _bot = min(erp.shape[0] - 1, _bot + _margin)
    erp_crop = erp[_top:_bot + 1]
    print(f"Crop polar black caps: rows [{_top}, {_bot}] of {erp.shape[0]} "
          f"-> {erp_crop.shape} (kept {100*erp_crop.shape[0]/erp.shape[0]:.1f}%)")
else:
    erp_crop = erp
cv2.imwrite(f"{OUT}/real_panorama.png", cv2.cvtColor(erp_crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 3])
print(f"Saved {OUT}/real_panorama.png  shape={erp_crop.shape}")

# --- Prepare icosphere ---
RENDER_RANK = 7  # upgrade to match model's img_rank for sharper detail (163k verts, 327k faces)
ref = IcoSphereRef("vertex")
ico_vis = ref.get_icosphere(RENDER_RANK, False)  # for topology (faces)
# trimesh icosphere verts are NOT unit-length — use get_normals() for unit sphere positions
verts_vis = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
faces_vis = np.asarray(ico_vis.faces, dtype=np.int64)
print(f"Render icosphere rank {RENDER_RANK}: {len(verts_vis)} verts (unit), {len(faces_vis)} faces")

# --- Get per-vertex RGB by sampling ERP ---
# asSpherical returns (r, phi, theta) in DEGREES:
#   phi   = vertical [0, 180]   (polar angle)
#   theta = horizontal [-180, 180] (azimuth)
from trimesh_utils import asSpherical
sph = asSpherical(verts_vis)
phi_deg   = sph[:, 1]
theta_deg = sph[:, 2]
H_, W_ = erp.shape[:2]
v_coord = (phi_deg / 180.0) * (H_ - 1)
u_coord = ((theta_deg + 180.0) / 360.0) * (W_ - 1)
v_coord = np.clip(v_coord, 0, H_ - 1).astype(np.float32)
u_coord = np.clip(u_coord, 0, W_ - 1).astype(np.float32)
# Manual bilinear ERP sampling (cv2.remap has SHRT_MAX limit; numpy is fine at 163k verts)
u0 = np.floor(u_coord).astype(np.int32)
v0 = np.floor(v_coord).astype(np.int32)
u1 = np.minimum(u0 + 1, W_ - 1)
v1 = np.minimum(v0 + 1, H_ - 1)
du = (u_coord - u0)[:, None]
dv = (v_coord - v0)[:, None]
erp_f = erp.astype(np.float32)
c00 = erp_f[v0, u0]; c01 = erp_f[v0, u1]
c10 = erp_f[v1, u0]; c11 = erp_f[v1, u1]
vert_rgb = ((1-du)*(1-dv)*c00 + du*(1-dv)*c01 + (1-du)*dv*c10 + du*dv*c11) / 255.0
print(f"vert_rgb range: [{vert_rgb.min():.3f}, {vert_rgb.max():.3f}], mean {vert_rgb.mean():.3f}")

# --- Build model & run inference at rank 7 ---
NUM_CLASSES = 14
encoder = SphericalEncoder(
    img_rank=mc["img_rank"], node_type=mc["node_type"],
    embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
    bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
    drop_path_rate=0.0,
    abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
    rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True,
)
model = EquiSSLSegUNet(
    encoder=encoder, num_classes=NUM_CLASSES,
    dec_depths=tuple(mc.get("dec_depths", [2,2,2,2])),
    dec_num_heads=tuple(mc.get("dec_num_heads", [16,16,8,4])),
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
    abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
    rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True,
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda().eval()

# Single-sample forward
sample = test_ds[SAMPLE_IDX]
rgb_sphere = sample["sphere_rgb"].unsqueeze(0).cuda()  # (1, Nproj_rank7, 3)
with torch.no_grad():
    logits = model(rgb_sphere)  # (1, Nproj_rank7, 14) — likely rank 6 due to in_scale_factor=2
pred = logits.argmax(dim=-1).cpu().numpy()[0]  # (Nproj_rank7,) of class ids
print(f"Prediction shape: {pred.shape}, unique classes: {np.unique(pred)}")

# The prediction is at proj rank (= img_rank - 1 since in_scale_factor=2) = rank 6 (40962 verts)
# which matches our RENDER_RANK=6.
proj_rank = mc["img_rank"] - 1 if mc.get("in_scale_factor", 2) == 2 else mc["img_rank"]
print(f"proj_rank (prediction resolution): {proj_rank}")

# Upsample prediction from proj_rank to RENDER_RANK via nearest-neighbor (parent->4-children subdivision)
if proj_rank != RENDER_RANK:
    proj_normals = np.asarray(ref.get_normals(proj_rank), dtype=np.float32)  # (N_proj, 3)
    # Nearest-neighbor lookup: for each render vertex, find nearest proj vertex
    from scipy.spatial import cKDTree
    tree = cKDTree(proj_normals)
    _, nn_idx = tree.query(verts_vis, k=1)
    pred_up = pred[nn_idx]
    print(f"Upsampled prediction from rank {proj_rank} ({len(pred)}) to rank {RENDER_RANK} ({len(pred_up)})")
    pred = pred_up
assert len(pred) == len(verts_vis), f"pred {len(pred)} != verts {len(verts_vis)}"

# per-vertex seg color
vert_seg_rgb = S2D3D_COLORS[pred]  # (Nv, 3)

# --- Per-face colors (mean of 3 vertices) ---
def face_colors_from_verts(vrgb, faces):
    return vrgb[faces].mean(axis=1)  # (Nf, 3)

face_rgb = face_colors_from_verts(vert_rgb, faces_vis)
face_seg = face_colors_from_verts(vert_seg_rgb, faces_vis)

# --- Render icosphere view: azimuth 30°, elevation 20° ---
def render_icosphere(verts, faces, face_colors, out_path, azim=30, elev=20,
                     final_px=1500, ss=2.4, dpi=300):
    """Super-sampled render: draw at ss×final_px, then downsample with Lanczos
    for anti-aliased edges against triangle flat-shading.
    """
    size_px = int(final_px * ss)
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    verts_faces = verts[faces]  # (Nf, 3, 3)
    # Compute face centroids and outward-pointing normals (unit sphere -> normal == centroid direction)
    centroids = verts_faces.mean(axis=1)
    n_raw = np.cross(verts_faces[:,1] - verts_faces[:,0], verts_faces[:,2] - verts_faces[:,0])
    n_raw /= np.linalg.norm(n_raw, axis=1, keepdims=True) + 1e-9
    # Ensure outward (agree with centroid direction)
    flip = (n_raw * centroids).sum(axis=1) < 0
    n_raw[flip] = -n_raw[flip]

    # View direction for azim/elev (camera looks at origin)
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    view_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    # Cull back-facing
    visible = (n_raw @ view_dir) > 0.01
    vf = verts_faces[visible]
    nf = n_raw[visible]
    fc = face_colors[visible]

    # Shading: light slightly from upper-right
    light = np.array([np.cos(np.deg2rad(225)) * np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225)) * np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = np.clip(nf @ light, 0, 1)
    shade = 0.55 + 0.45 * shade
    shaded = (fc[:, :3] * shade[:, None]).clip(0, 1)
    print(f"  Rendering {len(vf)} visible faces, shaded RGB range: [{shaded.min():.3f}, {shaded.max():.3f}], mean {shaded.mean():.3f}")

    coll = Poly3DCollection(vf, facecolors=shaded, edgecolors=shaded,
                            linewidths=0.5, antialiased=False)
    ax.add_collection3d(coll)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # Post-process: tight crop + square-pad + Lanczos downsample to final size
    from PIL import Image
    im = Image.open(out_path).convert("RGBA")
    bbox = im.getbbox()
    if bbox is not None:
        w, h = im.size
        mx = int((bbox[2]-bbox[0]) * 0.02)
        my = int((bbox[3]-bbox[1]) * 0.02)
        bbox = (max(0, bbox[0]-mx), max(0, bbox[1]-my),
                min(w, bbox[2]+mx), min(h, bbox[3]+my))
        im = im.crop(bbox)
        W = max(im.size)
        sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        sq.paste(im, ((W - im.size[0]) // 2, (W - im.size[1]) // 2))
        # High-quality downsample to final_px
        if W > final_px:
            sq = sq.resize((final_px, final_px), Image.LANCZOS)
        sq.save(out_path, optimize=True)
        print(f"  Super-sampled {W}x{W} → Lanczos {sq.size}")
    print(f"Saved {out_path}")

# --- Fig 2: icosphere with RGB ---
render_icosphere(verts_vis, faces_vis, face_rgb, f"{OUT}/real_icosphere_rgb.png", azim=30, elev=20)
# --- Fig 3: icosphere with segmentation ---
render_icosphere(verts_vis, faces_vis, face_seg, f"{OUT}/real_icosphere_seg.png", azim=30, elev=20)

print("\nDone. Files in", OUT)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    sz = os.path.getsize(p)
    print(f"  {f}  ({sz//1024} KB)")
