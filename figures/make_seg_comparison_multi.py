"""Supplementary Figure 5b: qualitative seg comparison across 3 additional
samples (breadth validation — EquiSSL wins on more than just the main
Figure 5 sample).

Samples picked from figures/sample_ranking_all.json: pix-acc > 90% and C4
beats both baselines. Layout: 3 rows (samples) × 4 cols (GT / SphereUFormer
/ Standard RPE / EquiSSL).
"""
import os, sys, yaml, cv2
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
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

# (split, idx, label) — tops of the ranked list, excluding the main val-28 sample.
SAMPLES = [
    ("test", 225, "Sample A"),
    ("test",  59, "Sample B"),
    ("val",   26, "Sample C"),
]

CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60], [0.90, 0.75, 0.25], [0.20, 0.60, 0.80], [0.55, 0.35, 0.20],
    [0.85, 0.85, 0.95], [0.95, 0.45, 0.45], [0.75, 0.55, 0.75], [0.40, 0.40, 0.60],
    [0.95, 0.75, 0.55], [0.55, 0.75, 0.45], [0.85, 0.35, 0.60], [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85], [0.30, 0.55, 0.85],
], dtype=np.float32)

with open(CFG) as f: cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]; PROJ_RANK = IMG_RANK - 1

ref = IcoSphereRef("vertex")
render_verts = np.asarray(ref.get_normals(7), dtype=np.float32)
render_faces = np.asarray(ref.get_icosphere(7, False).faces, dtype=np.int64)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
img_normals  = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)

_tree_proj = cKDTree(proj_normals)
_, _idx_render_from_proj = _tree_proj.query(render_verts, k=1)
_tree_img = cKDTree(img_normals)
_, _idx_proj_from_img = _tree_img.query(proj_normals, k=1)

def up_proj_to_render(pred_proj):
    return pred_proj[_idx_render_from_proj]

def build_ours(ckpt_path, rpe_mode, n_gauges):
    rp, eq = (True, False) if rpe_mode == "standard" else (True, True)
    enc = SphericalEncoder(
        img_rank=IMG_RANK, node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=rp, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=True)
    m = EquiSSLSegUNet(encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths",[2,2,2,2])),
        dec_num_heads=tuple(mc.get("dec_num_heads",[16,16,8,4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=True)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def build_sphereuformer(path):
    m = SphereUFormer(img_rank=IMG_RANK, node_type="vertex",
        in_channels=3, out_channels=14, in_scale_factor=2, num_scales=4,
        win_size_coef=2, enc_depths=2, dec_depths=2, bottleneck_depth=2, d_head_coef=2,
        enc_num_heads=[2,4,8,16], dec_num_heads=[16,16,8,4],
        abs_pos_enc_in=True, abs_pos_enc=True, rel_pos_bias=True,
        rel_pos_bias_size=7, rel_pos_init_variance=1.0,
        downsample="center", upsample="interpolate", use_checkpoint=True)
    c = torch.load(path, map_location="cpu", weights_only=False)
    state = c["model_state_dict"] if isinstance(c, dict) and "model_state_dict" in c else c
    m.load_state_dict(state)
    return m.cuda().eval()

def render_ico(face_colors, ax):
    verts_faces = render_verts[render_faces]
    centroids = verts_faces.mean(axis=1)
    n_raw = np.cross(verts_faces[:,1]-verts_faces[:,0], verts_faces[:,2]-verts_faces[:,0])
    n_raw /= np.linalg.norm(n_raw, axis=1, keepdims=True) + 1e-9
    flip = (n_raw*centroids).sum(axis=1) < 0
    n_raw[flip] = -n_raw[flip]
    az, el = np.deg2rad(30), np.deg2rad(20)
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    visible = (n_raw @ view) > 0.01
    vf = verts_faces[visible]; nf = n_raw[visible]; fc = face_colors[visible]
    light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                      np.sin(np.deg2rad(55))])
    shade = 0.55 + 0.45 * np.clip(nf @ light, 0, 1)
    shaded = (fc[:, :3] * shade[:, None]).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(vf, facecolors=shaded, edgecolors=shaded,
                                         linewidths=0.5, antialiased=False))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=20, azim=30)
    ax.set_axis_off()

def class_face_colors(pred_render):
    return S2D3D_COLORS[pred_render][render_faces].mean(axis=1)

# Front-facing face mask for camera (azim=30, elev=20); used to filter legend to
# classes that actually occupy visible area.
_verts_faces_for_vis = render_verts[render_faces]
_centroids_for_vis = _verts_faces_for_vis.mean(axis=1)
_n_for_vis = np.cross(_verts_faces_for_vis[:,1]-_verts_faces_for_vis[:,0],
                       _verts_faces_for_vis[:,2]-_verts_faces_for_vis[:,0])
_n_for_vis /= np.linalg.norm(_n_for_vis, axis=1, keepdims=True) + 1e-9
_flip_for_vis = (_n_for_vis*_centroids_for_vis).sum(axis=1) < 0
_n_for_vis[_flip_for_vis] = -_n_for_vis[_flip_for_vis]
_az, _el = np.deg2rad(30), np.deg2rad(20)
_view_for_vis = np.array([np.cos(_el)*np.cos(_az), np.cos(_el)*np.sin(_az), np.sin(_el)])
_visible_faces = (_n_for_vis @ _view_for_vis) > 0.01

def visible_classes(label_render, min_frac=0.003):
    face_labels = label_render[render_faces]
    maj = np.array([np.bincount(face_labels[i], minlength=14).argmax()
                    for i in np.where(_visible_faces)[0]])
    n_visible = _visible_faces.sum()
    return [c for c in range(14) if (maj == c).sum() / n_visible >= min_frac]

def rgb_face_colors_from_erp(erp_img):
    from trimesh_utils import asSpherical
    sph = asSpherical(render_verts)
    phi_deg, theta_deg = sph[:, 1], sph[:, 2]
    H_, W_ = erp_img.shape[:2]
    v = np.clip((phi_deg / 180.0) * (H_ - 1), 0, H_ - 1).astype(np.float32)
    u = np.clip(((theta_deg + 180.0) / 360.0) * (W_ - 1), 0, W_ - 1).astype(np.float32)
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, W_ - 1);   v1 = np.minimum(v0 + 1, H_ - 1)
    du = (u - u0)[:, None]; dv = (v - v0)[:, None]
    e = erp_img.astype(np.float32)
    vert_rgb = ((1-du)*(1-dv)*e[v0, u0] + du*(1-dv)*e[v0, u1]
                + (1-du)*dv*e[v1, u0] + du*dv*e[v1, u1]) / 255.0
    return vert_rgb[render_faces].mean(axis=1)

def render_and_crop(face_colors, size_px=700, dpi=200):
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    render_ico(face_colors, ax)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = f"/tmp/_multi_{id(face_colors)}.png"
    fig.savefig(buf, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    im = Image.open(buf).convert("RGBA")
    bbox = im.getbbox()
    if bbox is not None:
        im = im.crop(bbox)
        W = max(im.size)
        sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        sq.paste(im, ((W-im.size[0])//2, (W-im.size[1])//2))
        return sq
    return im

@torch.no_grad()
def run_ours(m, rgb): return m(rgb).argmax(dim=-1).cpu().numpy()[0]
@torch.no_grad()
def run_sphereu(m, rgb):
    return m(rgb).argmax(dim=-1).cpu().numpy()[0][_idx_proj_from_img]

print("Loading models...")
m_sphereu  = build_sphereuformer("outputs/sphereuformer_baseline/best_model.pth")
m_standard = build_ours("outputs/standard_seed123/best_model.pth", "standard", 6)
m_c4       = build_ours("outputs/rpe_ablation_c4_v2/best_model.pth", "equivariant", 4)

# Render each sample × 4 methods
row_panels = []  # list of [gt_img, sphereu_img, std_img, c4_img]
row_classes = []  # set of classes present (for legend union)
for split, idx, label in SAMPLES:
    print(f"\n--- {split} idx={idx} ({label}) ---")
    ds = Stanford2D3DSeg(split=split, data_dir="${STANFORD2D3D_PATH}",
        img_rank=IMG_RANK, node_type=mc["node_type"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"],
        normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"])
    sample = ds[idx]
    rgb = sample["sphere_rgb"].unsqueeze(0).cuda()
    gt_proj = sample["sphere_gt_sem"].numpy()
    erp_path = ds.samples[idx][0]
    erp = cv2.cvtColor(cv2.imread(erp_path), cv2.COLOR_BGR2RGB)
    p_sphereu  = run_sphereu(m_sphereu, rgb)
    p_standard = run_ours(m_standard, rgb)
    p_c4       = run_ours(m_c4, rgb)

    gt_r       = up_proj_to_render(gt_proj)
    p_sphereu_r  = up_proj_to_render(p_sphereu)
    p_standard_r = up_proj_to_render(p_standard)
    p_c4_r       = up_proj_to_render(p_c4)

    mask_ignore = (gt_r == 0)
    p_sphereu_r[mask_ignore]  = 0
    p_standard_r[mask_ignore] = 0
    p_c4_r[mask_ignore]       = 0

    row_panels.append([
        render_and_crop(rgb_face_colors_from_erp(erp)),
        render_and_crop(class_face_colors(gt_r)),
        render_and_crop(class_face_colors(p_sphereu_r)),
        render_and_crop(class_face_colors(p_standard_r)),
        render_and_crop(class_face_colors(p_c4_r)),
    ])
    row_classes.append(set(visible_classes(gt_r, min_frac=0.003)))

# Compose
CLASSES = ["unknown","beam","board","bookcase","ceiling","chair","clutter",
           "column","door","floor","sofa","table","wall","window"]
try:
    font_bold = ImageFont.truetype("${HOME_DIR}/.fonts/TimesNewRoman-Bold.ttf", 34)
    font_row  = ImageFont.truetype("${HOME_DIR}/.fonts/TimesNewRoman-Bold.ttf", 28)
    font_leg  = ImageFont.truetype("${HOME_DIR}/.fonts/TimesNewRoman.ttf", 22)
except Exception:
    font_bold = ImageFont.load_default(); font_row = font_bold; font_leg = font_bold

psz          = 360
gap_col      = 18
gap_row      = 14
title_h      = 64
row_label_w  = 190
legend_h     = 130  # extra height so wrapped legend rows don't clip
n_rows       = len(SAMPLES)

n_cols = 5
W = row_label_w + n_cols*psz + (n_cols-1)*gap_col
H = title_h + n_rows*psz + (n_rows-1)*gap_row + legend_h
canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

col_titles = ["(a)  Input RGB", "(b)  Ground truth", "(c)  SphereUFormer",
              "(d)  Standard RPE", "(e)  EquiSSL"]
for c in range(n_cols):
    x = row_label_w + c*(psz + gap_col) + psz//2
    draw.text((x, title_h//2 + 4), col_titles[c],
              fill=(20,20,20), font=font_bold, anchor="mm")

for r, (_, _, label) in enumerate(SAMPLES):
    y = title_h + r*(psz + gap_row)
    draw.text((row_label_w - 18, y + psz//2), label,
              fill=(20,20,20), font=font_row, anchor="rm")
    for c in range(n_cols):
        img = row_panels[r][c]
        x = row_label_w + c*(psz + gap_col)
        resized = img.resize((psz, psz), Image.LANCZOS)
        bg = Image.new("RGB", (psz, psz), (255, 255, 255))
        bg.paste(resized, (0, 0), resized)
        canvas.paste(bg, (x, y))

# Legend: union of classes across samples, with auto-wrap to next line
all_classes = sorted(set().union(*row_classes))
swatch = 24
entry_spacing = 18
row_stride = swatch + 14
legend_y0 = title_h + n_rows*psz + (n_rows-1)*gap_row + 22
left_margin = 22
right_margin = 22
ly = legend_y0
lx = left_margin
for c in all_classes:
    if c == 0:
        col = (153, 153, 153); name = "unlabeled"
    else:
        col = tuple((S2D3D_COLORS[c] * 255).astype(int)); name = CLASSES[c]
    text_w = draw.textlength(name, font=font_leg)
    entry_w = swatch + 8 + text_w + entry_spacing
    if lx + entry_w > W - right_margin:
        lx = left_margin
        ly += row_stride
    draw.rectangle([lx, ly, lx+swatch, ly+swatch], fill=col, outline=(60,60,60))
    draw.text((lx+swatch+8, ly+swatch//2), name,
              fill=(30,30,30), font=font_leg, anchor="lm")
    lx += entry_w

out_path = f"{OUT}/seg_comparison_multi.png"
canvas.save(out_path, optimize=True)
canvas.save(out_path.replace(".png", ".pdf"), "PDF", resolution=300)
print(f"\nSaved {out_path}  {canvas.size}")
