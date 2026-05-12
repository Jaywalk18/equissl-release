"""Figure 7 — Rotation sweep grid (PTH-only).

4 methods × 4 angles = 16 panels showing how SphereUFormer's output degrades
under rotation vs. EquiSSL's stability. Paper cannot regenerate this without
the model checkpoints, so we ship the rendered figure.

Methods: GT / SphereUFormer (CVPR'25) / Standard RPE / EquiSSL
Angles:  0° / 30° / 60° / 90° (yaw around world y-axis)

Output: figures/figs/fig_rotation_sweep.{png,pdf}
"""
import os, sys, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

SAMPLE_IDX = 28
SPLIT = "val"
CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

# ---- Load sample ----
with open(CFG) as f: cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK, PROJ_RANK = mc["img_rank"], mc["img_rank"] - 1
ref = IcoSphereRef("vertex")
img_normals = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
render_verts = np.asarray(ref.get_normals(7), dtype=np.float32)
render_faces = np.asarray(ref.get_icosphere(7, False).faces, dtype=np.int64)

ds = Stanford2D3DSeg(split=SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"])
rgb = ds[SAMPLE_IDX]["sphere_rgb"].unsqueeze(0).cuda()
gt_proj = ds[SAMPLE_IDX]["sphere_gt_sem"].numpy()

S2D3D_COLORS = np.array([
    [0.60,0.60,0.60],[0.90,0.75,0.25],[0.20,0.60,0.80],[0.55,0.35,0.20],
    [0.85,0.85,0.95],[0.95,0.45,0.45],[0.75,0.55,0.75],[0.40,0.40,0.60],
    [0.95,0.75,0.55],[0.55,0.75,0.45],[0.85,0.35,0.60],[0.70,0.50,0.25],
    [0.65,0.80,0.85],[0.30,0.55,0.85],
], dtype=np.float32)


def R_pitch(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]], dtype=np.float32)


def render_panel(pred_at_render, verts_render, out_path, size_px=640, dpi=200):
    fc = S2D3D_COLORS[pred_at_render][render_faces].mean(axis=1)
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    vf = verts_render[render_faces]
    cent = vf.mean(axis=1)
    n = np.cross(vf[:,1]-vf[:,0], vf[:,2]-vf[:,0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    flip = (n*cent).sum(axis=1) < 0; n[flip] = -n[flip]
    az, el = np.deg2rad(30), np.deg2rad(20)
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    vis = (n @ view) > 0.01
    light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = 0.55 + 0.45 * np.clip(n[vis] @ light, 0, 1)
    shaded = (fc[vis][:, :3] * shade[:, None]).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(vf[vis], facecolors=shaded, edgecolors=shaded,
                                         linewidths=0.5, antialiased=False))
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_zlim(-1.05,1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=20, azim=30)
    ax.set_axis_off(); fig.patch.set_alpha(0); ax.set_facecolor("none")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_ours(ckpt_path, rpe, ng):
    rp, eq = (True, False) if rpe=="standard" else (True, True)
    enc = SphericalEncoder(img_rank=IMG_RANK, node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=rp, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=ng, area_weighted=True)
    m = EquiSSLSegUNet(encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths",[2,2,2,2])),
        dec_num_heads=tuple(mc.get("dec_num_heads",[16,16,8,4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=ng, area_weighted=True)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def build_suformer(path):
    m = SphereUFormer(img_rank=IMG_RANK, node_type="vertex", in_channels=3, out_channels=14,
        in_scale_factor=2, num_scales=4, win_size_coef=2, enc_depths=2,
        dec_depths=2, bottleneck_depth=2, d_head_coef=2,
        enc_num_heads=[2,4,8,16], dec_num_heads=[16,16,8,4],
        abs_pos_enc_in=True, abs_pos_enc=True, rel_pos_bias=True,
        rel_pos_bias_size=7, rel_pos_init_variance=1.0,
        downsample="center", upsample="interpolate", use_checkpoint=True)
    c = torch.load(path, map_location="cpu", weights_only=False)
    state = c["model_state_dict"] if isinstance(c, dict) and "model_state_dict" in c else c
    m.load_state_dict(state)
    return m.cuda().eval()


def upsample(pred, src_nm):
    tree = cKDTree(src_nm)
    _, idx = tree.query(render_verts, k=1)
    return pred[idx]


@torch.no_grad()
def pred_fn(model, rgb_in):
    return model(rgb_in).argmax(dim=-1).cpu().numpy()[0]


# ---- Collect predictions: 3 trained models (SphereUFormer, Standard, C4) × 4 angles ----
angles = [0, 30, 60, 90]
methods = [
    ("SphereUFormer", "outputs/sphereuformer_baseline/best_model.pth", "suformer"),
    ("EquiSSL",   "outputs/rpe_ablation_c4_v2/best_model.pth",      "c4"),
]

panels = {}  # (method_name, angle) -> path to rendered PNG

# ---- Ground truth row (compute first so ignore masks are available for preds) ----
print("\n--- Ground truth ---")
gt_render_by_angle = {}
for ang in angles:
    if ang == 0:
        gt_pred = gt_proj
    else:
        R = R_pitch(ang)
        proj_perm = compute_rotation_permutation(proj_normals, R)
        gt_pred = gt_proj[proj_perm]
    gt_render = upsample(gt_pred, proj_normals)
    gt_render_by_angle[ang] = gt_render
    panel_path = f"/tmp/_sweep_gt_{ang}.png"
    render_panel(gt_render, render_verts, panel_path)
    panels[("Ground truth", ang)] = panel_path

# ---- Prediction rows: mask unlabeled regions to match GT's grey ----
for method_name, ckpt_path, kind in methods:
    print(f"\n--- {method_name} ---")
    if kind == "suformer":
        model = build_suformer(ckpt_path)
        src_nm = img_normals  # SphereUFormer outputs at img_rank
    else:
        model = build_ours(ckpt_path, "standard" if kind=="standard" else "equivariant",
                           4 if kind=="c4" else 6)
        src_nm = proj_normals  # Ours outputs at proj_rank

    for ang in angles:
        if ang == 0:
            pred = pred_fn(model, rgb)
        else:
            R = R_pitch(ang)
            img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                                    dtype=torch.long).cuda()
            rgb_rot = apply_rotation_to_features(rgb, img_perm)
            pred = pred_fn(model, rgb_rot)

        pred_render = pred if len(pred) == len(render_verts) else upsample(pred, src_nm)
        # Mask unlabeled regions to grey (class 0) — models never predict
        # class 0 because it's the ignore class, so this only affects the
        # visualization, matching the GT panel's grey regions.
        pred_render = pred_render.copy()
        pred_render[gt_render_by_angle[ang] == 0] = 0

        panel_path = f"/tmp/_sweep_{kind}_{ang}.png"
        render_panel(pred_render, render_verts, panel_path)
        panels[(method_name, ang)] = panel_path
        print(f"  angle {ang}°: rendered")

    del model; torch.cuda.empty_cache()

# ---- Input RGB row: actual scene texture, rotated at each angle ----
print("\n--- Input RGB ---")
import cv2
from trimesh_utils import asSpherical
erp = cv2.cvtColor(cv2.imread(ds.samples[SAMPLE_IDX][0]), cv2.COLOR_BGR2RGB)
sph = asSpherical(render_verts)
phi_deg, theta_deg = sph[:, 1], sph[:, 2]
H_, W_ = erp.shape[:2]
v = np.clip((phi_deg / 180.0) * (H_ - 1), 0, H_ - 1).astype(np.float32)
u = np.clip(((theta_deg + 180.0) / 360.0) * (W_ - 1), 0, W_ - 1).astype(np.float32)
u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
u1 = np.minimum(u0 + 1, W_ - 1);   v1 = np.minimum(v0 + 1, H_ - 1)
du = (u - u0)[:, None]; dv = (v - v0)[:, None]
e = erp.astype(np.float32)
vert_rgb = ((1-du)*(1-dv)*e[v0, u0] + du*(1-dv)*e[v0, u1]
            + (1-du)*dv*e[v1, u0] + du*dv*e[v1, u1]) / 255.0

def render_rgb_panel(vert_rgb_use, out_path, size_px=640, dpi=200):
    fc = vert_rgb_use[render_faces].mean(axis=1)
    verts = render_verts
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    vf = verts[render_faces]
    n = np.cross(vf[:,1]-vf[:,0], vf[:,2]-vf[:,0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    cent = vf.mean(axis=1)
    flip = (n*cent).sum(axis=1) < 0; n[flip] = -n[flip]
    az, el = np.deg2rad(30), np.deg2rad(20)
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    vis = (n @ view) > 0.01
    light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = 0.55 + 0.45 * np.clip(n[vis] @ light, 0, 1)
    shaded = (fc[vis][:, :3] * shade[:, None]).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(vf[vis], facecolors=shaded, edgecolors=shaded,
                                         linewidths=0.5, antialiased=False))
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_zlim(-1.05,1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=20, azim=30)
    ax.set_axis_off(); fig.patch.set_alpha(0); ax.set_facecolor("none")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

for ang in angles:
    if ang == 0:
        vert_rgb_use = vert_rgb
    else:
        render_perm = compute_rotation_permutation(render_verts, R_pitch(ang))
        vert_rgb_use = vert_rgb[render_perm]
    panel_path = f"/tmp/_sweep_rgb_{ang}.png"
    render_rgb_panel(vert_rgb_use, panel_path)
    panels[("Input RGB", ang)] = panel_path

# ---- Compose 4×4 grid ----
FONT_BOLD = "${HOME_DIR}/.fonts/TimesNewRoman-Bold.ttf"
FONT_REG = "${HOME_DIR}/.fonts/TimesNewRoman.ttf"

row_order = ["Input RGB", "Ground truth", "SphereUFormer", "EquiSSL"]
col_order = angles

panel_size = 340
gap = 18
col_header_h = 64
row_header_w = 240
total_w = row_header_w + len(col_order)*panel_size + (len(col_order)-1)*gap
total_h = col_header_h + len(row_order)*panel_size + (len(row_order)-1)*gap + 16

canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
f_col = ImageFont.truetype(FONT_BOLD, 32)
f_row = ImageFont.truetype(FONT_BOLD, 26)
f_small = ImageFont.truetype(FONT_REG, 20)

# Column headers (angles)
for c, ang in enumerate(col_order):
    x = row_header_w + c*(panel_size + gap) + panel_size//2
    draw.text((x, col_header_h//2 + 4), f"{ang}° rotation", fill=(20,20,20), font=f_col, anchor="mm")

# Row headers (method names) + panels
for r, row_name in enumerate(row_order):
    y = col_header_h + r*(panel_size + gap)
    # Row label + optional subtitle
    draw.text((row_header_w - 18, y + panel_size//2 - 10), row_name,
              fill=(20,20,20), font=f_row, anchor="rm")
    if row_name == "SphereUFormer":
        draw.text((row_header_w - 18, y + panel_size//2 + 20), "(CVPR'25)",
                  fill=(110,110,110), font=f_small, anchor="rm")
    elif row_name == "EquiSSL":
        draw.text((row_header_w - 18, y + panel_size//2 + 20), "(ours)",
                  fill=(110,110,110), font=f_small, anchor="rm")
    for c, ang in enumerate(col_order):
        x = row_header_w + c*(panel_size + gap)
        im = Image.open(panels[(row_name, ang)]).convert("RGBA")
        bbox = im.getbbox()
        if bbox: im = im.crop(bbox)
        W = max(im.size)
        sq = Image.new("RGBA", (W, W), (0,0,0,0))
        sq.paste(im, ((W-im.size[0])//2, (W-im.size[1])//2))
        sq = sq.resize((panel_size, panel_size), Image.LANCZOS)
        bg = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
        bg.paste(sq, (0, 0), sq)
        canvas.paste(bg, (x, y))

canvas.save(f"{OUT}/fig_rotation_sweep.png", optimize=True)
canvas.save(f"{OUT}/fig_rotation_sweep.pdf", "PDF", resolution=300)
print(f"\nSaved {OUT}/fig_rotation_sweep.{{png,pdf}}  {canvas.size}")
