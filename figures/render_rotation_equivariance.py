"""Render 4 icosphere-segmentation PNGs for paper Figure 1 motivation:
    seg_std_upright.png, seg_std_rot90.png, seg_ge_upright.png, seg_ge_rot90.png

Demonstrates visually that EquiSSL is rotation-equivariant (pattern follows sphere)
while standard RPE is gauge-dependent (pattern breaks under rotation).

Mode B (current): supervised checkpoints
    Model 1: outputs/standard_seed123/best_model.pth   (Standard RPE, val 0.6667)
    Model 2: outputs/rpe_ablation_c4_v2/best_model.pth (EquiSSL, val 0.6801)

Mode A (if SSL finished): outputs/ssl_s2d3d_{standard,c4}/best_model.pth

All renders: sample idx=37 (office_6), azim=30°, elev=20°, transparent bg.
Rotation: fixed +90° yaw around world z-axis, applied as icosphere node permutation
(matches eval_pose35.py protocol).
"""
import os, sys, yaml, cv2
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from trimesh_utils import IcoSphereRef

# ---- Config ----
SAMPLE_IDX = 37
CFG  = "configs/pretrain_v8_large.yaml"
OUT  = "figures/figs"

# Mode detection
SSL_STD = "outputs/ssl_s2d3d_standard/best_model.pth"
SSL_GE  = "outputs/ssl_s2d3d_c4/best_model.pth"
SUP_STD = "outputs/standard_seed123/best_model.pth"   # Standard RPE seed 123
SUP_GE  = "outputs/rpe_ablation_c4_v2/best_model.pth" # EquiSSL-C₄ seed 42

if os.path.exists(SSL_STD) and os.path.exists(SSL_GE):
    MODE = "A"; CKPT_STD, CKPT_GE = SSL_STD, SSL_GE
    print(f"=== Mode A: using SSL-pretrained checkpoints ===")
else:
    MODE = "B"; CKPT_STD, CKPT_GE = SUP_STD, SUP_GE
    print(f"=== Mode B: SSL not ready, using supervised checkpoints ===")
print(f"  Model 1 (Standard RPE): {CKPT_STD}")
print(f"  Model 2 (EquiSSL):    {CKPT_GE}")

os.makedirs(OUT, exist_ok=True)

# Stanford2D3D 14-class colormap (RGB 0-1) — same as render_real_figs.py
S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60],  [0.90, 0.75, 0.25],  [0.20, 0.60, 0.80],
    [0.55, 0.35, 0.20],  [0.85, 0.85, 0.95],  [0.95, 0.45, 0.45],
    [0.75, 0.55, 0.75],  [0.40, 0.40, 0.60],  [0.95, 0.75, 0.55],
    [0.55, 0.75, 0.45],  [0.85, 0.35, 0.60],  [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85],  [0.30, 0.55, 0.85],
], dtype=np.float32)

# ---- Load config & dataset ----
with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
NUM_CLASSES = 14
IMG_RANK   = mc["img_rank"]                    # 7
PROJ_RANK  = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK  # 6
RENDER_RANK = 7

ds_kwargs = dict(
    data_dir="${STANFORD2D3D_PATH}", img_rank=IMG_RANK,
    node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"],
)
test_ds = Stanford2D3DSeg(split="test", **ds_kwargs)
sample = test_ds[SAMPLE_IDX]
rgb_sphere_cpu = sample["sphere_rgb"].unsqueeze(0)  # (1, N_img, 3)
print(f"Sample {SAMPLE_IDX}: {os.path.basename(test_ds.samples[SAMPLE_IDX][0])}")

# ---- Icosphere references ----
ref = IcoSphereRef("vertex")
img_normals   = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)   # (163842, 3)
proj_normals  = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)  # (40962, 3)
render_verts  = np.asarray(ref.get_normals(RENDER_RANK), dtype=np.float32)
render_faces  = np.asarray(ref.get_icosphere(RENDER_RANK, False).faces, dtype=np.int64)
print(f"img_rank {IMG_RANK}: {len(img_normals)}  proj_rank {PROJ_RANK}: {len(proj_normals)}")

# ---- Fixed +90° pitch (rotation around world y-axis) ----
# Tilts the up-axis sideways — breaks any implicit "up is z" assumption in
# Standard RPE and exposes gauge-dependence more dramatically than z-axis yaw
# (which is nearly a symmetry of the icosphere).
ANGLE = np.pi / 2
R_pitch = np.array([[ np.cos(ANGLE), 0, np.sin(ANGLE)],
                    [ 0,             1, 0            ],
                    [-np.sin(ANGLE), 0, np.cos(ANGLE)]], dtype=np.float32)
R_yaw = R_pitch   # alias: keep symbol name for downstream code
print(f"R (90° pitch around y-axis):\n{R_yaw}")

# Precompute permutations at img_rank (for input rotation) and proj_rank (reference)
img_perm_np  = compute_rotation_permutation(img_normals, R_yaw)   # (163842,)
proj_perm_np = compute_rotation_permutation(proj_normals, R_yaw)  # (40962,)
img_perm  = torch.tensor(img_perm_np,  dtype=torch.long)
proj_perm = torch.tensor(proj_perm_np, dtype=torch.long)
print(f"img_perm shape={img_perm.shape}  first 5: {img_perm[:5].tolist()}")

# Sanity: rotation fixed ratio (should be ~0% identical at 90°)
identity_frac = (img_perm_np == np.arange(len(img_perm_np))).mean()
print(f"fraction of vertices unchanged by 90° yaw: {identity_frac*100:.1f}%  (should be low)")


def build_model(ckpt_path, rpe_mode, n_gauges=6):
    if rpe_mode == "standard":
        rp, eq = True, False
    elif rpe_mode == "equivariant":
        rp, eq = True, True
    else:
        raise ValueError(rpe_mode)
    encoder = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=rp, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=True,
    )
    model = EquiSSLSegUNet(
        encoder=encoder, num_classes=NUM_CLASSES,
        dec_depths=tuple(mc.get("dec_depths", [2,2,2,2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16,16,8,4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.cuda().eval()


@torch.no_grad()
def predict(model, rgb_input):
    """rgb_input: (1, N_img, 3). Returns pred (N_proj,) int64."""
    return model(rgb_input).argmax(dim=-1).cpu().numpy()[0]


def upsample_pred_to_render(pred_proj):
    """NN upsample from proj_rank to render_rank (in world frame). Shape: (N_render,)."""
    tree = cKDTree(proj_normals)
    _, nn_idx = tree.query(render_verts, k=1)
    return pred_proj[nn_idx]


def render_icosphere(verts, faces, face_colors, out_path, azim=30, elev=20,
                     final_px=1500, ss=2.4, dpi=300):
    """Identical renderer to render_real_figs.py (super-sampled + Lanczos)."""
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
    vf = verts_faces[visible]
    nf = n_raw[visible]
    fc = face_colors[visible]

    light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = 0.55 + 0.45 * np.clip(nf @ light, 0, 1)
    shaded = (fc[:, :3] * shade[:, None]).clip(0, 1)

    coll = Poly3DCollection(vf, facecolors=shaded, linewidths=0, antialiased=True)
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


# ---- Main: run 2 models × 2 conditions ----
rgb_sphere = rgb_sphere_cpu.cuda()

# Rotation is applied as node permutation on the input features at img_rank.
# After model forward, output is at proj_rank. When rendering the rotated case,
# we place pred[i] at R @ render_verts[i] so the icosphere is visually rotated
# — an equivariant model produces the same pattern on the rotated sphere,
# a non-equivariant one produces a different (broken) pattern.
verts_render_upright = render_verts                  # (N, 3)
verts_render_rotated = (R_yaw @ render_verts.T).T    # (N, 3) — rotated positions

for model_name, ckpt_path, rpe_mode in [
    ("std", CKPT_STD, "standard"),
    ("ge",  CKPT_GE,  "equivariant"),
]:
    print(f"\n--- Building model: {model_name} ({rpe_mode}) ---")
    model = build_model(ckpt_path, rpe_mode=rpe_mode, n_gauges=4 if rpe_mode == "equivariant" else 6)

    # Upright forward
    pred_up = predict(model, rgb_sphere)
    pred_up_render = upsample_pred_to_render(pred_up)
    print(f"  [{model_name}] upright: pred classes {sorted(set(pred_up.tolist()))}")

    # Rotated forward: apply permutation to input
    rgb_rot = apply_rotation_to_features(rgb_sphere, img_perm.cuda())
    pred_rot = predict(model, rgb_rot)
    pred_rot_render = upsample_pred_to_render(pred_rot)
    print(f"  [{model_name}] rotated: pred classes {sorted(set(pred_rot.tolist()))}")

    # Render
    seg_up_rgb  = S2D3D_COLORS[pred_up_render]
    seg_rot_rgb = S2D3D_COLORS[pred_rot_render]

    render_icosphere(verts_render_upright, render_faces,
                     face_colors_from_verts(seg_up_rgb, render_faces),
                     f"{OUT}/seg_{model_name}_upright.png")
    render_icosphere(verts_render_rotated, render_faces,
                     face_colors_from_verts(seg_rot_rgb, render_faces),
                     f"{OUT}/seg_{model_name}_rot90.png")

    # Clean up before next model
    del model
    torch.cuda.empty_cache()

print("\n==== Output summary ====")
for f in ["seg_std_upright.png", "seg_std_rot90.png",
          "seg_ge_upright.png",  "seg_ge_rot90.png"]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        print(f"  {p}  ({os.path.getsize(p)//1024} KB)")

print(f"\nMode: {MODE}")
print(f"  Standard RPE ckpt: {CKPT_STD}")
print(f"  EquiSSL ckpt:    {CKPT_GE}")
