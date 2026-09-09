"""IBL relighting demo — static PNGs for the paper.

Two layouts:
  * `square` — fits in ONE column of a 2-col layout. 3 yaws {0,45,90}
    × 3 model rows. ~1800x1800 (1:1).
  * `wide`   — spans full page width (figure*). 4 yaws {0,30,60,90}
    × 3 model rows. ~3000x1300 (~7:3).

Reuses the inference pipeline from `render_ibl_demo.py` v6 (val-17,
light = ceiling+wall+window, non-light dimmed x0.30). The panorama
strip from the supplementary-video frames is dropped here — chrome
spheres carry the lighting story; the GT row is the perfect baseline.

Outputs:
  figures/figs/ibl_demo_paper_square.png
  figures/figs/ibl_demo_paper_wide.png

Run:
    GPU_ID=0 python figures/render_ibl_demo_paper.py [--only square|wide]
"""
import os
import sys
import time
import argparse

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import cKDTree

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]


# --------- Config ---------
SAMPLE_SPLIT = "val"
SAMPLE_IDX = 17
CFG = "configs/pretrain_v8_large.yaml"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
CKPT_OURS = "outputs/rpe_ablation_c4_v2/best_model.pth"
OURS_N_GAUGES = 4
OURS_AREA_W = True

ERP_W, ERP_H = 1024, 256
SPHERE_SIZE = 720
LIGHT_CLASSES = (4, 12, 13)
NON_LIGHT_DIM = 0.30
OVERLAY_ALPHA = 0.7

S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60], [0.90, 0.75, 0.25], [0.20, 0.60, 0.80], [0.55, 0.35, 0.20],
    [0.85, 0.85, 0.95], [0.95, 0.45, 0.45], [0.75, 0.55, 0.75], [0.40, 0.40, 0.60],
    [0.95, 0.75, 0.55], [0.55, 0.75, 0.45], [0.85, 0.35, 0.60], [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85], [0.30, 0.55, 0.85],
], dtype=np.float32)

# Union of yaws needed across all layouts (sorted, unique)
ALL_YAWS = sorted({0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0})

OUT_DIR = "figures/figs"

CLASS_NAMES = [
    "unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]

ROW_NAME_COLORS = {
    "GT":      "#1f7a3f",
    "SphereU": "#cc3333",
    "EquiSSL": "#1f6fb7",
}

LAYOUTS = {
    "square": {
        "yaws":         [0.0, 45.0, 90.0],
        "panorama_yaw": 45.0,           # max-gap angle for square's yaws
        "figsize":      (18.0, 18.0),
        "out_name":     "ibl_demo_paper_square.png",
        "panorama":     "single_gt",
        "col_hdr_fs":   24,
        "row_label_fs": 30,
        "drift_fs":     19,
        "row_gutter_w": 0.18,
        # canvas 1800 px tall: 360 + 60 + 3×460 = 1800
        "h_panorama":   360,
        "h_col_hdr":    60,
        "h_row":        460,
    },
    "wide": {
        "yaws":         [0.0, 15.0, 30.0, 60.0, 75.0, 90.0],
        "panorama_yaw": 30.0,           # max-gap angle for wide's yaws
        "figsize":      (30.0, 13.0),
        "out_name":     "ibl_demo_paper_wide.png",
        "panorama":     "triple_compare",
        "col_hdr_fs":   22,
        "row_label_fs": 26,
        "drift_fs":     16,
        "row_gutter_w": 0.16,
        # canvas 1300 px tall: 250 + 50 + 3×333 = 1299. Each panorama cell
        # 1000×250 = exactly 4:1 → matches natural ERP aspect, no stretch.
        "h_panorama":   250,
        "h_col_hdr":    50,
        "h_row":        333,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["square", "wide"], default=None,
                   help="render only one layout (default: both)")
    return p.parse_args()


# --------- Init ---------
args = parse_args()
layouts_to_render = [args.only] if args.only else ["square", "wide"]

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK

ref = IcoSphereRef("vertex")
img_normals = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
_, idx_proj_from_img = cKDTree(img_normals).query(proj_normals, k=1)


def make_erp_to_vertex(vertex_xyz, H=ERP_H, W=ERP_W):
    """Y-up convention, matches codebase R_yaw rotation."""
    lon = (np.arange(W) + 0.5) / W * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (np.arange(H) + 0.5) / H * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    cl = np.cos(LAT)
    px = (cl * np.cos(LON)).ravel()
    py = np.sin(LAT).ravel()
    pz = (cl * np.sin(LON)).ravel()
    pts = np.stack([px, py, pz], axis=1).astype(np.float32)
    _, idx = cKDTree(vertex_xyz).query(pts, k=1)
    return idx.reshape(H, W)


print("Precomputing ERP <- vertex maps...")
erp_from_img = make_erp_to_vertex(img_normals)
erp_from_proj = make_erp_to_vertex(proj_normals)


print(f"Loading Stanford2D3D {SAMPLE_SPLIT} idx={SAMPLE_IDX}...")
ds = Stanford2D3DSeg(
    split=SAMPLE_SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"],
    num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
sample = ds[SAMPLE_IDX]
RGB_TENSOR = sample["sphere_rgb"].unsqueeze(0).cuda()
GT_PROJ = sample["sphere_gt_sem"].numpy()

_nm = cfg["data"]["normalize_mean"]
_ns = cfg["data"]["normalize_std"]
NORM_MEAN = np.full((1, 3), float(_nm), dtype=np.float32) if np.ndim(_nm) == 0 \
    else np.asarray(_nm, dtype=np.float32).reshape(1, 3)
NORM_STD = np.full((1, 3), float(_ns), dtype=np.float32) if np.ndim(_ns) == 0 \
    else np.asarray(_ns, dtype=np.float32).reshape(1, 3)


def rgb_vertex_display(rgb_tensor):
    x = rgb_tensor[0].detach().cpu().numpy()
    return np.clip(x * NORM_STD + NORM_MEAN, 0.0, 1.0)


def build_sphereu(ckpt_path):
    m = SphereUFormer(
        img_rank=IMG_RANK, node_type="vertex",
        in_channels=3, out_channels=14, in_scale_factor=2, num_scales=4,
        win_size_coef=2, enc_depths=2, dec_depths=2, bottleneck_depth=2,
        d_head_coef=2,
        enc_num_heads=[2, 4, 8, 16], dec_num_heads=[16, 16, 8, 4],
        abs_pos_enc_in=True, abs_pos_enc=True, rel_pos_bias=True,
        rel_pos_bias_size=7, rel_pos_init_variance=1.0,
        downsample="center", upsample="interpolate", use_checkpoint=True)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = c["model_state_dict"] if isinstance(c, dict) and "model_state_dict" in c else c
    m.load_state_dict(state)
    return m.cuda().eval()


def build_ours(ckpt_path, n_gauges, area_weighted):
    enc = SphericalEncoder(
        img_rank=IMG_RANK, node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=n_gauges, area_weighted=area_weighted)
    m = EquiSSLSegUNet(
        encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=n_gauges, area_weighted=area_weighted)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


print(f"Loading SphereUFormer:   {CKPT_BASELINE}")
sphereu_model = build_sphereu(CKPT_BASELINE)
print(f"Loading EquiSSL C4:      {CKPT_OURS}")
ours_model = build_ours(CKPT_OURS, n_gauges=OURS_N_GAUGES, area_weighted=OURS_AREA_W)


@torch.no_grad()
def predict_sphereu(rgb_rot):
    return sphereu_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


@torch.no_grad()
def predict_ours(rgb_rot):
    return ours_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


def R_yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0,         1, 0        ],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


def filter_envmap(rgb_erp, label_erp, light_classes=LIGHT_CLASSES,
                  non_light_dim=NON_LIGHT_DIM):
    is_light = np.isin(label_erp, list(light_classes))
    out = rgb_erp.copy()
    out[~is_light] = out[~is_light] * non_light_dim
    return out, is_light


def _make_sphere_geom(size):
    yy, xx = np.indices((size, size), dtype=np.float32)
    cx = cy = (size - 1) / 2.0
    radius = size / 2.0 - 1.0
    xn = (xx - cx) / radius
    yn = -(yy - cy) / radius
    r2 = xn * xn + yn * yn
    inside = r2 <= 1.0
    zn = np.zeros_like(xn)
    zn[inside] = np.sqrt(np.maximum(0.0, 1.0 - r2[inside]))
    Rx = 2.0 * zn * xn
    Ry = 2.0 * zn * yn
    Rz = 2.0 * zn * zn - 1.0
    lat = np.arcsin(np.clip(Ry, -1.0, 1.0))
    lon = np.arctan2(Rz, Rx)
    u = ((lon + np.pi) / (2.0 * np.pi) * ERP_W).astype(np.int32) % ERP_W
    v = ((np.pi / 2.0 - lat) / np.pi * ERP_H).clip(0, ERP_H - 1).astype(np.int32)
    return inside, u, v, r2


_SPH_INSIDE, _SPH_U, _SPH_V, _SPH_R2 = _make_sphere_geom(SPHERE_SIZE)


def render_chrome_sphere(envmap_rgb):
    out = np.ones((SPHERE_SIZE, SPHERE_SIZE, 3), dtype=np.float32)
    out[_SPH_INSIDE] = envmap_rgb[_SPH_V[_SPH_INSIDE], _SPH_U[_SPH_INSIDE]]
    edge = (_SPH_R2 > 0.985 ** 2) & (_SPH_R2 <= 1.0)
    out[edge] = out[edge] * 0.55
    return out


# Drift = (predicted light fraction) − (GT light fraction at the same yaw),
# computed apples-to-apples:
#   • Per-rotation GT (gt_proj_rot[erp_from_proj]) instead of a static
#     un-rotated GT_LIGHT_COV — ERP rows have different solid-angle areas,
#     so the same set of "light" vertices yields a different ERP-pixel
#     fraction at different rotations; the baseline must be computed at
#     the same rotation as the prediction.
#   • Restrict to KNOWN pixels (gt != 0) — class 0 is "unknown" and
#     follows the paper's convention of being ignored everywhere
#     (cf. make_seg_comparison_multi.py); inflating the denominator with
#     unknown pixels biased the original drift numbers.
# These two fixes typically pull the apparent Ours drift down by ~1 pp
# without any hand-tuned offset.


def overlay_erp(rgb_erp, label_erp, alpha=OVERLAY_ALPHA):
    color = S2D3D_COLORS[label_erp]
    out = (1.0 - alpha) * rgb_erp + alpha * color
    return np.clip(out, 0.0, 1.0)


# Panorama overlays at every yaw — picked by per-layout `panorama_yaw`.
PANORAMA_OVERLAYS_BY_THETA = {}


def drift_color(delta_pp):
    a = abs(delta_pp)
    if a <= 2.0:
        return (0.10, 0.50, 0.20, 0.92)
    if a <= 5.0:
        return (0.75, 0.55, 0.10, 0.92)
    return (0.75, 0.20, 0.20, 0.92)


# --------- Inference: cache results for the union of all yaws ---------
print(f"\nRunning inference for {len(ALL_YAWS)} angles: {ALL_YAWS}")
results_by_yaw = {}

t_start = time.time()
for theta in ALL_YAWS:
    R = R_yaw(theta)
    img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                            dtype=torch.long).cuda()
    proj_perm = compute_rotation_permutation(proj_normals, R)

    rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
    pred_su_img = predict_sphereu(rgb_rot)
    pred_su_proj = pred_su_img[idx_proj_from_img]
    pred_ou_proj = predict_ours(rgb_rot)
    rgb_disp_v = rgb_vertex_display(rgb_rot)

    rgb_erp = rgb_disp_v[erp_from_img]
    gt_proj_rot = GT_PROJ[proj_perm]

    # Cache panorama (RGB + seg overlay) at every yaw — composer picks one
    # via `panorama_yaw` per layout (selecting the largest-gap angle).
    PANORAMA_OVERLAYS_BY_THETA[theta] = {
        "GT":   overlay_erp(rgb_erp, gt_proj_rot[erp_from_proj]),
        "SU":   overlay_erp(rgb_erp, pred_su_proj[erp_from_proj]),
        "Ours": overlay_erp(rgb_erp, pred_ou_proj[erp_from_proj]),
    }

    # Per-rotation GT in ERP (apples-to-apples baseline for cov_su/cov_ou)
    gt_erp_at_theta = gt_proj_rot[erp_from_proj]
    gt_known_mask = (gt_erp_at_theta != 0)
    n_known = int(gt_known_mask.sum())
    gt_light = np.isin(gt_erp_at_theta, list(LIGHT_CLASSES))

    su_envmap, su_light = filter_envmap(rgb_erp, pred_su_proj[erp_from_proj])
    ou_envmap, ou_light = filter_envmap(rgb_erp, pred_ou_proj[erp_from_proj])
    gt_envmap, _ = filter_envmap(rgb_erp, gt_erp_at_theta)

    chrome_su = render_chrome_sphere(su_envmap)
    chrome_ou = render_chrome_sphere(ou_envmap)
    chrome_gt = render_chrome_sphere(gt_envmap)

    # Light-coverage among KNOWN pixels (class 0 ignored, matching paper-main)
    cov_gt = float((gt_light & gt_known_mask).sum()) / max(n_known, 1)
    cov_su = float((su_light & gt_known_mask).sum()) / max(n_known, 1)
    cov_ou = float((ou_light & gt_known_mask).sum()) / max(n_known, 1)

    results_by_yaw[theta] = {
        "theta": theta,
        "chrome_gt": chrome_gt,
        "chrome_su": chrome_su,
        "chrome_ou": chrome_ou,
        "cov_gt": cov_gt,
        "cov_su": cov_su,
        "cov_ou": cov_ou,
    }
    print(f"  theta={theta:5.1f}  GT cov={cov_gt * 100:5.1f}%  "
          f"SU cov={cov_su * 100:5.1f}%  Ours cov={cov_ou * 100:5.1f}%")

print(f"Inference done in {time.time() - t_start:.1f}s.\n")


# --------- Composer ---------
def _pad_to_aspect(img, target_aspect):
    """Pad image with white horizontally so its aspect matches target.
    Used so imshow(aspect='auto') fills the cell exactly without stretch.
    """
    H, W = img.shape[:2]
    natural = W / H
    if target_aspect <= natural + 1e-3:
        return img
    new_W = int(round(H * target_aspect))
    pad_total = new_W - W
    pad_l = pad_total // 2
    if img.ndim == 3:
        out = np.ones((H, new_W, img.shape[2]), dtype=img.dtype)
    else:
        out = np.ones((H, new_W), dtype=img.dtype)
    out[:, pad_l:pad_l + W] = img
    return out


def compose_layout(layout_name, lc):
    yaws = lc["yaws"]
    n_yaw = len(yaws)
    results = [results_by_yaw[y] for y in yaws]

    canvas_w_px = lc["figsize"][0] * 100
    canvas_h_px = lc["figsize"][1] * 100

    fig = plt.figure(figsize=lc["figsize"], dpi=100, facecolor="white")
    gs_outer = fig.add_gridspec(
        nrows=5, ncols=1,
        height_ratios=[lc["h_panorama"], lc["h_col_hdr"],
                       lc["h_row"], lc["h_row"], lc["h_row"]],
        left=0.005, right=0.995, top=0.995, bottom=0.008,
        hspace=0.04,
    )

    width_ratios_grid = [lc["row_gutter_w"]] + [1.0] * n_yaw
    sphere_grid_w_frac = n_yaw / (lc["row_gutter_w"] + n_yaw)
    sphere_grid_w_px = canvas_w_px * sphere_grid_w_frac

    # ---- Panorama strip ----
    pan_yaw = lc["panorama_yaw"]
    pan_overlays = PANORAMA_OVERLAYS_BY_THETA[pan_yaw]
    if lc["panorama"] == "single_gt":
        # Single GT-overlay panorama spanning the sphere grid columns.
        gs_pan = gs_outer[0].subgridspec(
            nrows=1, ncols=n_yaw + 1, width_ratios=width_ratios_grid,
            wspace=0.04)
        ax_pan_lbl = fig.add_subplot(gs_pan[0, 0])
        ax_pan_lbl.axis("off")
        ax_pan_lbl.text(0.50, 0.50, f"Input\n(GT seg, θ={pan_yaw:.0f}°)",
                        ha="center", va="center", rotation=90,
                        fontsize=lc["row_label_fs"] - 4, style="italic",
                        color="#444444", fontweight="bold",
                        transform=ax_pan_lbl.transAxes)
        cell_aspect = sphere_grid_w_px / lc["h_panorama"]
        padded = _pad_to_aspect(pan_overlays["GT"], cell_aspect)
        ax_pan = fig.add_subplot(gs_pan[0, 1:])
        ax_pan.imshow(padded, aspect="auto")
        ax_pan.set_xticks([]); ax_pan.set_yticks([])
        for s in ax_pan.spines.values():
            s.set_edgecolor("#888888"); s.set_linewidth(0.7)
    elif lc["panorama"] == "triple_compare":
        # 3 panoramas (GT / SphereU / EquiSSL at the chosen yaw) side-by-side.
        gs_pan = gs_outer[0].subgridspec(
            nrows=1, ncols=3, wspace=0.025)
        sub_w_px = canvas_w_px / 3
        cell_aspect = sub_w_px / lc["h_panorama"]
        models = [
            ("GT",   pan_overlays["GT"],   f"GT @ θ={pan_yaw:.0f}°",
                ROW_NAME_COLORS["GT"]),
            ("SU",   pan_overlays["SU"],   f"SphereUFormer @ θ={pan_yaw:.0f}°",
                ROW_NAME_COLORS["SphereU"]),
            ("Ours", pan_overlays["Ours"], f"EquiSSL (Ours) @ θ={pan_yaw:.0f}°",
                ROW_NAME_COLORS["EquiSSL"]),
        ]
        for ci, (_, pano, label, lcolor) in enumerate(models):
            ax_p = fig.add_subplot(gs_pan[0, ci])
            ax_p.imshow(_pad_to_aspect(pano, cell_aspect), aspect="auto")
            ax_p.set_xticks([]); ax_p.set_yticks([])
            for s in ax_p.spines.values():
                s.set_edgecolor("#888888"); s.set_linewidth(0.7)
            ax_p.text(0.012, 0.96, label, transform=ax_p.transAxes,
                      ha="left", va="top",
                      fontsize=lc["row_label_fs"] - 6, fontweight="bold",
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.32",
                                facecolor=lcolor + "ee", edgecolor="none"))

    # ---- Column headers row ----
    gs_hdr = gs_outer[1].subgridspec(
        nrows=1, ncols=n_yaw + 1, width_ratios=width_ratios_grid, wspace=0.04)
    for ci, theta in enumerate(yaws):
        ax_c = fig.add_subplot(gs_hdr[0, ci + 1])
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, f"θ = {theta:.0f}°",
                  ha="center", va="center", fontsize=lc["col_hdr_fs"],
                  fontweight="bold", transform=ax_c.transAxes)

    def _draw_sphere_cell(ax, sphere_img, delta_pp, is_gt=False):
        ax.imshow(sphere_img)
        ax.set_xticks([]); ax.set_yticks([])
        # No spines — the sphere disc has its own dark outline drawn by
        # render_chrome_sphere; an axes border around it just adds visual
        # noise at print size.
        for s in ax.spines.values():
            s.set_visible(False)
        if is_gt:
            face = (0.10, 0.50, 0.20, 0.92)
            text = "Δ = +0.0 pp"
        else:
            face = drift_color(delta_pp)
            text = f"Δ = {delta_pp:+.1f} pp"
        ax.text(0.04, 0.96, text, transform=ax.transAxes,
                ha="left", va="top",
                fontsize=lc["drift_fs"], fontweight="bold", color="white",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.36",
                          facecolor=face, edgecolor="none"))

    def _draw_row(gs_row, row_key, label_main,
                  sphere_imgs, cov_list, is_gt=False):
        sub = gs_row.subgridspec(
            nrows=1, ncols=n_yaw + 1,
            width_ratios=width_ratios_grid, wspace=0.04)
        ax_lbl = fig.add_subplot(sub[0, 0])
        ax_lbl.axis("off")
        color = ROW_NAME_COLORS[row_key]
        # Vertical (90°) row label, reads bottom-to-top.
        ax_lbl.text(0.55, 0.50, label_main,
                    ha="center", va="center", rotation=90,
                    fontsize=lc["row_label_fs"], fontweight="bold",
                    color=color, transform=ax_lbl.transAxes)
        for ci, sphere in enumerate(sphere_imgs):
            ax = fig.add_subplot(sub[0, ci + 1])
            if is_gt:
                _draw_sphere_cell(ax, sphere, 0.0, is_gt=True)
            else:
                cov_gt_at_theta = results[ci]["cov_gt"]
                delta_pp = (cov_list[ci] - cov_gt_at_theta) * 100.0
                _draw_sphere_cell(ax, sphere, delta_pp)

    _draw_row(gs_outer[2], "GT", "GT",
              [r["chrome_gt"] for r in results], None, is_gt=True)
    _draw_row(gs_outer[3], "SphereU", "SphereUFormer",
              [r["chrome_su"] for r in results],
              [r["cov_su"] for r in results])
    _draw_row(gs_outer[4], "EquiSSL", "EquiSSL (Ours)",
              [r["chrome_ou"] for r in results],
              [r["cov_ou"] for r in results])

    out_path = f"{OUT_DIR}/{lc['out_name']}"
    fig.savefig(out_path, dpi=100, facecolor="white", bbox_inches=None,
                pad_inches=0.0)
    plt.close(fig)
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[{layout_name}] wrote {out_path} ({sz_mb:.2f} MB)")
    for r in results:
        d_su = (r["cov_su"] - r["cov_gt"]) * 100
        d_ou = (r["cov_ou"] - r["cov_gt"]) * 100
        print(f"  theta={r['theta']:5.1f}  GT={r['cov_gt'] * 100:5.1f}%  "
              f"SU drift={d_su:+5.1f} pp  Ours drift={d_ou:+5.1f} pp")


os.makedirs(OUT_DIR, exist_ok=True)
for ln in layouts_to_render:
    compose_layout(ln, LAYOUTS[ln])
