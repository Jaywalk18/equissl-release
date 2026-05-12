"""Render Figure 5: seg_comparison 2×8 panel (predictions + error maps).

Sample selection: val split idx=28, picked by figures/rank_all_splits.py
(C4 pixel-acc = 94.1%, gap over best baseline = +2.7%).

Row 1: (a) GT / (b) SphereUFormer / (c) Standard RPE / (d) Tangent-Img /
       (e) PanoFormer / (f) HEAL-SWIN / (g) SO3UFormer / (h) EquiSSL (Ours)
       — class-colored predictions.
Row 2: input RGB on sphere + 7 error maps for the prediction panels
       (green=correct, red=wrong, grey=unlabeled, not scored).

NOTE on (d) Tangent-Img, (e) PanoFormer, (f) HEAL-SWIN, (g) SO3UFormer:
these are LAYOUT PLACEHOLDERS sourced from internal ablation checkpoints
(Standard seed-456, C6 area-weighted, No-RPE, C2 GE-RPE respectively).
They are not real baseline trainings — swap the checkpoints below once
those baselines are run.
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

# Chosen sample (see figures/rank_all_splits.py for selection rationale)
CANDIDATES = [
    ("val",  28, ""),
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
CLASSES = ["unknown","beam","board","bookcase","ceiling","chair","clutter",
           "column","door","floor","sofa","table","wall","window"]

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
    if rpe_mode == "standard":
        rp, eq = True, False
    elif rpe_mode == "none":
        rp, eq = False, False
    else:  # equivariant
        rp, eq = True, True
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
    # edgecolors=shaded fills sub-pixel gaps with face color; antialiased=False
    # avoids moire from edge-AA interference at rank-7 (327k faces).
    ax.add_collection3d(Poly3DCollection(vf, facecolors=shaded, edgecolors=shaded,
                                         linewidths=0.5, antialiased=False))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=20, azim=30)
    ax.set_axis_off()

def class_face_colors(pred_render):
    return S2D3D_COLORS[pred_render][render_faces].mean(axis=1)

# Precompute front-facing face mask for the shared camera view (azim=30, elev=20)
_verts_faces_for_vis = render_verts[render_faces]
_centroids_for_vis = _verts_faces_for_vis.mean(axis=1)
_n_for_vis = np.cross(_verts_faces_for_vis[:,1]-_verts_faces_for_vis[:,0],
                       _verts_faces_for_vis[:,2]-_verts_faces_for_vis[:,0])
_n_for_vis /= np.linalg.norm(_n_for_vis, axis=1, keepdims=True) + 1e-9
_flip_for_vis = (_n_for_vis*_centroids_for_vis).sum(axis=1) < 0
_n_for_vis[_flip_for_vis] = -_n_for_vis[_flip_for_vis]
_az_for_vis, _el_for_vis = np.deg2rad(30), np.deg2rad(20)
_view_for_vis = np.array([np.cos(_el_for_vis)*np.cos(_az_for_vis),
                           np.cos(_el_for_vis)*np.sin(_az_for_vis),
                           np.sin(_el_for_vis)])
_visible_faces = (_n_for_vis @ _view_for_vis) > 0.01

def visible_classes(label_render, min_frac=0.003):
    """Return sorted class ids that occupy >= min_frac of front-facing area."""
    face_labels = label_render[render_faces]  # (F, 3)
    maj = np.array([np.bincount(face_labels[i], minlength=14).argmax()
                    for i in np.where(_visible_faces)[0]])
    n_visible = _visible_faces.sum()
    present = []
    for c in range(14):
        if (maj == c).sum() / n_visible >= min_frac:
            present.append(c)
    return present

def rgb_face_colors_from_erp(erp_img):
    """Bilinear-sample the ERP at each render vertex (unit sphere),
    average 3 vertices per face. Returns (n_faces, 3) in [0,1]."""
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

def error_face_colors(pred_render, gt_render):
    """Green = correct, red = wrong, grey = unlabeled (gt=0, ignored in metric).
    Grey in error maps matches the grey in the GT panel — class 0 is
    Stanford2D3D's 'unknown' class and is never scored against the model."""
    correct = (pred_render == gt_render)
    ignore = (gt_render == 0)
    verts_correct = correct[render_faces].mean(axis=1) > 0.5
    verts_ignore  = ignore[render_faces].mean(axis=1) > 0.5
    green = np.array([0.45, 0.80, 0.45], dtype=np.float32)
    red   = np.array([0.90, 0.30, 0.30], dtype=np.float32)
    grey  = np.array([0.60, 0.60, 0.60], dtype=np.float32)
    return np.where(verts_ignore[:, None], grey,
                    np.where(verts_correct[:, None], green, red))

def render_and_crop(face_colors, size_px=800, dpi=200):
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    render_ico(face_colors, ax)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = f"/tmp/_render_{id(face_colors)}.png"
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
# (d) Tangent-Img — placeholder: Standard RPE seed 456. Replace when trained.
m_tangent  = build_ours("outputs/standard_seed456_v9/best_model.pth", "standard", 6)
# (e) PanoFormer — placeholder: C6 area-weighted GE-RPE. Replace when trained.
m_pano     = build_ours("outputs/c6_seed123/best_model.pth", "equivariant", 6)
# (f) HEAL-SWIN — placeholder: No-RPE ablation. Replace when trained.
m_heal     = build_ours("outputs/none_seed123_v9/best_model.pth", "none", 4)
# (g) SO3UFormer — placeholder: C2 GE-RPE ablation. Replace when trained.
m_so3      = build_ours("outputs/c2_seed123_v9/best_model.pth", "equivariant", 2)
m_c4       = build_ours("outputs/rpe_ablation_c4_v2/best_model.pth", "equivariant", 4)

try:
    # Scaled down for 8-col layout (psz=240) so panel titles like
    # "(h) EquiSSL (Ours)" fit within their cells.
    font_bold = ImageFont.truetype("${HOME_DIR}/.fonts/TimesNewRoman-Bold.ttf", 26)
    font_sub  = ImageFont.truetype("${HOME_DIR}/.fonts/TimesNewRoman.ttf", 19)
except Exception:
    font_bold = ImageFont.load_default(); font_sub = font_bold

for split, idx, suf in CANDIDATES:
    print(f"\n--- {split} idx={idx} ({suf}) ---")
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
    p_tangent  = run_ours(m_tangent, rgb)
    p_pano     = run_ours(m_pano, rgb)
    p_heal     = run_ours(m_heal, rgb)
    p_so3      = run_ours(m_so3, rgb)
    p_c4       = run_ours(m_c4, rgb)

    gt_r         = up_proj_to_render(gt_proj)
    p_sphereu_r  = up_proj_to_render(p_sphereu)
    p_standard_r = up_proj_to_render(p_standard)
    p_tangent_r  = up_proj_to_render(p_tangent)
    p_pano_r     = up_proj_to_render(p_pano)
    p_heal_r     = up_proj_to_render(p_heal)
    p_so3_r      = up_proj_to_render(p_so3)
    p_c4_r       = up_proj_to_render(p_c4)
    # Paint prediction panels grey in gt=0 regions (same as GT panel). These
    # pixels are ignored in the metric, so forcing the class to 0 for the
    # visualization only affects the rendered color.
    mask_ignore = (gt_r == 0)
    p_sphereu_r[mask_ignore]  = 0
    p_standard_r[mask_ignore] = 0
    p_tangent_r[mask_ignore]  = 0
    p_pano_r[mask_ignore]     = 0
    p_heal_r[mask_ignore]     = 0
    p_so3_r[mask_ignore]      = 0
    p_c4_r[mask_ignore]       = 0

    # Row 1: class-colored predictions
    row1 = [
        ("(a)  Ground truth",      class_face_colors(gt_r)),
        ("(b)  SphereUFormer",     class_face_colors(p_sphereu_r)),
        ("(c)  Standard RPE",      class_face_colors(p_standard_r)),
        ("(d)  Tangent-Img",       class_face_colors(p_tangent_r)),
        ("(e)  PanoFormer",        class_face_colors(p_pano_r)),
        ("(f)  HEAL-SWIN",         class_face_colors(p_heal_r)),
        ("(g)  SO3UFormer",        class_face_colors(p_so3_r)),
        ("(h)  EquiSSL (Ours)",    class_face_colors(p_c4_r)),
    ]
    # Row 2: col1 is input RGB on the sphere; cols 2-8 are error maps
    row2 = [
        ("Input RGB on sphere",    rgb_face_colors_from_erp(erp)),
        ("SphereUFormer error",    error_face_colors(p_sphereu_r,  gt_r)),
        ("Standard RPE error",     error_face_colors(p_standard_r, gt_r)),
        ("Tangent-Img error",      error_face_colors(p_tangent_r,  gt_r)),
        ("PanoFormer error",       error_face_colors(p_pano_r,     gt_r)),
        ("HEAL-SWIN error",        error_face_colors(p_heal_r,     gt_r)),
        ("SO3UFormer error",       error_face_colors(p_so3_r,      gt_r)),
        ("EquiSSL (Ours) error",   error_face_colors(p_c4_r,       gt_r)),
    ]

    panels = []
    for (t, fc) in row1 + row2:
        if fc is None:
            panels.append((t, None))
        else:
            panels.append((t, render_and_crop(fc)))

    # Compose canvas: 2 rows × 8 cols
    n_cols = 8
    psz = 240
    gap = 14
    title_h = 60
    subtitle_h = 44
    legend_h = 120  # room for wrapped legend row
    W = n_cols * psz + (n_cols - 1) * gap
    H = title_h + psz + subtitle_h + psz + legend_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Row 1 titles
    for i in range(n_cols):
        x = i * (psz + gap)
        t, img = panels[i]
        draw.text((x + psz // 2, title_h // 2), t, fill=(20, 20, 20), font=font_bold, anchor="mm")
        if img is not None:
            resized = img.resize((psz, psz), Image.LANCZOS)
            bg = Image.new("RGB", (psz, psz), (255, 255, 255))
            bg.paste(resized, (0, 0), resized)
            canvas.paste(bg, (x, title_h))

    # Row 2 subtitles + panels
    y_sub = title_h + psz
    y_row2 = y_sub + subtitle_h
    for i in range(n_cols):
        x = i * (psz + gap)
        t, img = panels[n_cols + i]
        if img is None:
            continue
        draw.text((x + psz // 2, y_sub + subtitle_h // 2), t,
                  fill=(60, 60, 60), font=font_sub, anchor="mm")
        resized = img.resize((psz, psz), Image.LANCZOS) if img.size != (psz, psz) else img
        bg = Image.new("RGB", (psz, psz), (255, 255, 255))
        bg.paste(resized, (0, 0), resized if resized.mode == "RGBA" else None)
        canvas.paste(bg, (x, y_row2))

    # Legend: only classes occupying meaningful front-facing area (avoid clutter
    # from classes that exist in GT but aren't visually present in this view).
    vis_ids = visible_classes(gt_r, min_frac=0.003)
    has_unlabeled = 0 in vis_ids
    present = [c for c in vis_ids if c != 0]
    swatch = 24
    row_stride = swatch + 14
    left_margin = 20
    right_margin = 20

    # Build class entries; reserve right-side space for correct/wrong legend
    err_legend_w = 260
    max_lx_first_row = W - right_margin - err_legend_w

    entries = []
    for c in present:
        entries.append((CLASSES[c], tuple((S2D3D_COLORS[c] * 255).astype(int))))
    if has_unlabeled:
        entries.append(("unlabeled", (153, 153, 153)))

    ly = y_row2 + psz + 22
    lx = left_margin
    current_cap = max_lx_first_row  # first row shares width with err legend
    for name, col in entries:
        text_w = draw.textlength(name, font=font_sub)
        entry_w = swatch + 8 + text_w + 22
        if lx + entry_w > current_cap:
            lx = left_margin
            ly += row_stride
            current_cap = W - right_margin  # second row has full width
        draw.rectangle([lx, ly, lx + swatch, ly + swatch], fill=col, outline=(60, 60, 60))
        draw.text((lx + swatch + 6, ly + swatch // 2), name,
                  fill=(30, 30, 30), font=font_sub, anchor="lm")
        lx += entry_w

    # Error-map legend always pinned to the first row, flush right
    err_lx = W - right_margin - err_legend_w
    err_ly = y_row2 + psz + 22
    for lbl, col in [("correct", (115, 204, 115)), ("wrong", (230, 77, 77))]:
        draw.rectangle([err_lx, err_ly, err_lx + swatch, err_ly + swatch], fill=col, outline=(60, 60, 60))
        draw.text((err_lx + swatch + 6, err_ly + swatch // 2), lbl,
                  fill=(30, 30, 30), font=font_sub, anchor="lm")
        err_lx += swatch + 8 + draw.textlength(lbl, font=font_sub) + 16

    out_path = f"{OUT}/seg_comparison{('_' + suf) if suf else ''}.png"
    canvas.save(out_path, optimize=True)
    canvas.save(out_path.replace(".png", ".pdf"), "PDF", resolution=300)
    print(f"  Saved {out_path}  {canvas.size}")

print("\nDone.")
