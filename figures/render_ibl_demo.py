"""IBL relighting demo — supplementary video segment.

Two-row 2x2 layout:
  Top    (SphereUFormer): panorama+seg overlay | chrome sphere lit by
                           filtered envmap (ceiling+wall RGB only).
  Bottom (EquiSSL C4):     same.

Yaw sweep 0deg -> 90deg over 13 s at 24 fps (312 frames). The chrome
sphere reflection is computed analytically (orthographic mirror reflect
from the filtered ERP envmap). When seg is rotation-equivariant
(EquiSSL), the filtered envmap rotates smoothly with the panorama and
the reflection sweeps cleanly. When seg flickers under rotation
(SphereUFormer), the envmap mask flickers at class boundaries and the
reflection has visible "wobble".

Reuses ckpts and rotation/permutation infrastructure from
render_smooth_yaw_360.py.

Run:
    GPU_ID=0 python figures/render_ibl_demo.py [--frames N] [--no-mp4]
"""
import os
import sys
import argparse
import yaml
import time

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"]  = ["Times New Roman", "Liberation Serif"]


# --------- Config ---------
SAMPLE_SPLIT  = "val"
SAMPLE_IDX    = 17
NUM_FRAMES    = 312        # 13s @ 24fps
FPS           = 24
YAW_MAX       = 90.0       # 0 -> 90 deg sweep
CFG           = "configs/pretrain_v8_large.yaml"
OVERLAY_ALPHA = 0.7

OUT_DIR    = "figures/figs"
FRAMES_DIR = f"{OUT_DIR}/ibl_demo_frames"
VIDEO_PATH = f"{OUT_DIR}/ibl_demo_13s.mp4"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
CKPT_OURS     = "outputs/rpe_ablation_c4_v2/best_model.pth"
OURS_N_GAUGES = 4
OURS_AREA_W   = True

# ERP grid for sphere envmap (must be same convention as make_erp_to_vertex)
ERP_W = 1024
ERP_H = 256

# Chrome sphere render size
SPHERE_SIZE = 720

# IBL filter: which class ids count as "light"
# 4=ceiling, 12=wall, 13=window (per Stanford2D3D 13-class palette)
LIGHT_CLASSES = (4, 12, 13)
# Non-light pixels are dimmed by this factor (instead of being set to 0/black),
# so the chrome sphere shows a more natural reflection while still emphasising
# the light-source regions.
NON_LIGHT_DIM = 0.30

CLASS_NAMES = [
    "unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]
S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60], [0.90, 0.75, 0.25], [0.20, 0.60, 0.80], [0.55, 0.35, 0.20],
    [0.85, 0.85, 0.95], [0.95, 0.45, 0.45], [0.75, 0.55, 0.75], [0.40, 0.40, 0.60],
    [0.95, 0.75, 0.55], [0.55, 0.75, 0.45], [0.85, 0.35, 0.60], [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85], [0.30, 0.55, 0.85],
], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=NUM_FRAMES)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--sample-idx", type=int, default=SAMPLE_IDX)
    p.add_argument("--sample-split", default=SAMPLE_SPLIT)
    return p.parse_args()


# --------- Init ---------
args = parse_args()
NUM_FRAMES = args.frames

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK  = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK

ref = IcoSphereRef("vertex")
img_normals  = np.asarray(ref.get_normals(IMG_RANK),  dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
_, idx_proj_from_img = cKDTree(img_normals).query(proj_normals, k=1)


# --------- ERP projection (Y-up convention to match codebase R_yaw) ---------
def make_erp_to_vertex(vertex_xyz, H=ERP_H, W=ERP_W):
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
erp_from_img  = make_erp_to_vertex(img_normals)
erp_from_proj = make_erp_to_vertex(proj_normals)


# --------- Dataset / sample ---------
print(f"Loading Stanford2D3D {args.sample_split} idx={args.sample_idx}...")
ds = Stanford2D3DSeg(
    split=args.sample_split, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"],
    num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
sample = ds[args.sample_idx]
RGB_TENSOR = sample["sphere_rgb"].unsqueeze(0).cuda()
GT_PROJ    = sample["sphere_gt_sem"].numpy()

_nm = cfg["data"]["normalize_mean"]
_ns = cfg["data"]["normalize_std"]
NORM_MEAN = np.full((1, 3), float(_nm), dtype=np.float32) if np.ndim(_nm) == 0 \
            else np.asarray(_nm, dtype=np.float32).reshape(1, 3)
NORM_STD  = np.full((1, 3), float(_ns), dtype=np.float32) if np.ndim(_ns) == 0 \
            else np.asarray(_ns, dtype=np.float32).reshape(1, 3)


def rgb_vertex_display(rgb_tensor):
    x = rgb_tensor[0].detach().cpu().numpy()
    return np.clip(x * NORM_STD + NORM_MEAN, 0.0, 1.0)


# --------- Models ---------
def build_sphereu(ckpt_path):
    m = SphereUFormer(img_rank=IMG_RANK, node_type="vertex",
        in_channels=3, out_channels=14, in_scale_factor=2, num_scales=4,
        win_size_coef=2, enc_depths=2, dec_depths=2, bottleneck_depth=2, d_head_coef=2,
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
    m = EquiSSLSegUNet(encoder=enc, num_classes=14,
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
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [ 0,         1, 0        ],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


# --------- IBL: filtered envmap + analytic chrome sphere render ---------
def filter_envmap(rgb_erp, label_erp, light_classes=LIGHT_CLASSES,
                  non_light_dim=NON_LIGHT_DIM):
    """Light pixels keep full RGB; non-light pixels are dimmed by `non_light_dim`."""
    is_light = np.isin(label_erp, list(light_classes))
    out = rgb_erp.copy()
    out[~is_light] = out[~is_light] * non_light_dim
    return out, is_light


# Precompute chrome sphere geometry once (reused per frame)
def _make_sphere_geom(size):
    yy, xx = np.indices((size, size), dtype=np.float32)
    cx = cy = (size - 1) / 2.0
    radius = size / 2.0 - 1.0
    xn = (xx - cx) / radius
    yn = -(yy - cy) / radius   # screen-up to world-up
    r2 = xn * xn + yn * yn
    inside = r2 <= 1.0
    zn = np.zeros_like(xn)
    zn[inside] = np.sqrt(np.maximum(0.0, 1.0 - r2[inside]))
    # Surface normals = (xn, yn, zn). Orthographic viewer V=(0,0,1).
    # R = 2(N.V)N - V; N.V = zn.
    Rx = 2.0 * zn * xn
    Ry = 2.0 * zn * yn
    Rz = 2.0 * zn * zn - 1.0
    # Spherical (Y-up: lat = asin(Ry); X-forward: lon = atan2(Rz, Rx))
    lat = np.arcsin(np.clip(Ry, -1.0, 1.0))
    lon = np.arctan2(Rz, Rx)
    u = ((lon + np.pi) / (2.0 * np.pi) * ERP_W).astype(np.int32) % ERP_W
    v = ((np.pi / 2.0 - lat) / np.pi * ERP_H).clip(0, ERP_H - 1).astype(np.int32)
    return inside, u, v, r2


_SPH_INSIDE, _SPH_U, _SPH_V, _SPH_R2 = _make_sphere_geom(SPHERE_SIZE)


def render_chrome_sphere(envmap_rgb):
    """Render orthographic mirror chrome sphere lit by ERP envmap.
    Background outside the sphere disc is white; subtle outline at the edge.
    """
    out = np.ones((SPHERE_SIZE, SPHERE_SIZE, 3), dtype=np.float32)
    out[_SPH_INSIDE] = envmap_rgb[_SPH_V[_SPH_INSIDE], _SPH_U[_SPH_INSIDE]]
    # A thin dark outline so the disc is visible against white bg
    edge = (_SPH_R2 > 0.985 ** 2) & (_SPH_R2 <= 1.0)
    out[edge] = out[edge] * 0.55
    return out


def overlay_erp(rgb_erp, label_erp, alpha=0.7):
    color = S2D3D_COLORS[label_erp]
    out = (1.0 - alpha) * rgb_erp + alpha * color
    return np.clip(out, 0.0, 1.0)


# --------- GT reference: light-pixel coverage from GT (rotation invariant) ---------
# This is the "correct" coverage that models should match. SphereU drifts
# above it under rotation (false positives); EquiSSL stays close to it.
_gt_erp_for_cov = GT_PROJ[erp_from_proj]
GT_LIGHT_COV = float(np.isin(_gt_erp_for_cov, list(LIGHT_CLASSES)).mean())
print(f"GT reference light-pixel coverage: {GT_LIGHT_COV*100:.1f}%")


def drift_color(delta_pp):
    """Color-code metric box by absolute drift in percentage points."""
    a = abs(delta_pp)
    if a <= 2.0:   return (0.10, 0.50, 0.20, 0.85)  # green: stable
    if a <= 5.0:   return (0.75, 0.55, 0.10, 0.85)  # amber: noticeable
    return (0.75, 0.20, 0.20, 0.85)                 # red: significant drift


# --------- Pre-pass: legend classes ---------
print("Pre-pass: scanning classes for legend (4 angles)...")
MIN_COVERAGE = 0.005
classes_in_fig = {0}
with torch.no_grad():
    for a in np.linspace(0, YAW_MAX, 4, endpoint=False):
        R = R_yaw(a)
        ip = torch.tensor(compute_rotation_permutation(img_normals, R),
                          dtype=torch.long).cuda()
        pp = compute_rotation_permutation(proj_normals, R)
        rr = apply_rotation_to_features(RGB_TENSOR, ip)
        ps = predict_sphereu(rr)[idx_proj_from_img]
        po = predict_ours(rr)
        gtr = GT_PROJ[pp]
        for arr in (gtr.astype(np.int64), ps.astype(np.int64), po.astype(np.int64)):
            counts = np.bincount(arr, minlength=14)
            frac = counts / max(arr.size, 1)
            for c in range(14):
                if frac[c] >= MIN_COVERAGE:
                    classes_in_fig.add(int(c))
classes_in_fig = sorted(classes_in_fig)
print(f"  legend classes: {[CLASS_NAMES[c] if c else 'unlabeled' for c in classes_in_fig]}")


def draw_horizontal_legend(ax, classes_to_show):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(classes_to_show)
    if n == 0: return
    cell_w = 1.0 / n
    swatch_w = 0.024; swatch_h = 0.60; y0 = 0.20
    for i, c in enumerate(classes_to_show):
        x = i * cell_w + 0.005
        name = "unlabeled" if c == 0 else CLASS_NAMES[c]
        ax.add_patch(Rectangle((x, y0), swatch_w, swatch_h,
                               facecolor=S2D3D_COLORS[c],
                               edgecolor=(0.2, 0.2, 0.2), linewidth=0.7))
        ax.text(x + swatch_w + 0.005, y0 + swatch_h / 2, name,
                ha="left", va="center", fontsize=13, color=(0.1, 0.1, 0.1))


# --------- Main loop ---------
os.makedirs(FRAMES_DIR, exist_ok=True)
for fn in os.listdir(FRAMES_DIR):
    if fn.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, fn))

angles = np.linspace(0, YAW_MAX, NUM_FRAMES, endpoint=True)
print(f"\nRendering {NUM_FRAMES} frames at {FPS} fps "
      f"(yaw 0 -> {YAW_MAX} deg, step {angles[1] - angles[0]:.3f} deg, "
      f"total {NUM_FRAMES / FPS:.2f} s)")

t_start = time.time()
for fi, theta in enumerate(angles):
    R = R_yaw(theta)
    img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                            dtype=torch.long).cuda()
    proj_perm = compute_rotation_permutation(proj_normals, R)

    rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
    pred_su_img  = predict_sphereu(rgb_rot)
    pred_su_proj = pred_su_img[idx_proj_from_img]
    pred_ou_proj = predict_ours(rgb_rot)
    rgb_disp_v   = rgb_vertex_display(rgb_rot)

    # Mask predictions on GT-unknown so overlays show grey for ignored regions
    gt_proj_rot = GT_PROJ[proj_perm]
    ignore_mask_proj = (gt_proj_rot == 0)
    pred_su_disp = pred_su_proj.copy(); pred_su_disp[ignore_mask_proj] = 0
    pred_ou_disp = pred_ou_proj.copy(); pred_ou_disp[ignore_mask_proj] = 0

    rgb_erp = rgb_disp_v[erp_from_img]
    su_erp  = pred_su_disp[erp_from_proj]
    ou_erp  = pred_ou_disp[erp_from_proj]

    overlay_su = overlay_erp(rgb_erp, su_erp, alpha=OVERLAY_ALPHA)
    overlay_ou = overlay_erp(rgb_erp, ou_erp, alpha=OVERLAY_ALPHA)
    overlay_gt = overlay_erp(rgb_erp, gt_proj_rot[erp_from_proj], alpha=OVERLAY_ALPHA)

    # IBL: filter envmap by predicted-light classes (encoder's own decision)
    su_envmap, su_light_mask = filter_envmap(rgb_erp, pred_su_proj[erp_from_proj])
    ou_envmap, ou_light_mask = filter_envmap(rgb_erp, pred_ou_proj[erp_from_proj])
    # GT reference envmap: same filter rule but using ground-truth labels —
    # this is the "perfect segmentation" baseline for visual comparison.
    gt_envmap, _ = filter_envmap(rgb_erp, gt_proj_rot[erp_from_proj])

    chrome_su = render_chrome_sphere(su_envmap)
    chrome_ou = render_chrome_sphere(ou_envmap)
    chrome_gt = render_chrome_sphere(gt_envmap)

    # Light-pixel coverage as a quantitative anchor (top-right of sphere panel)
    cov_su = float(su_light_mask.mean())
    cov_ou = float(ou_light_mask.mean())

    # --- Compose frame ---
    # 3 rows x 2 cols: each row is one source (GT / SphereU / EquiSSL).
    # Left col: panorama+seg overlay.  Right col: chrome sphere reflection.
    fig = plt.figure(figsize=(20.0, 12.0), dpi=110)
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[2.5, 1.0],
        height_ratios=[1.0, 1.0, 1.0, 0.18],
        left=0.020, right=0.990,
        top=0.93, bottom=0.04,
        wspace=0.04, hspace=0.20,
    )

    def _row(row_i, overlay, sphere_img, name, cov, is_gt=False):
        ax_p = fig.add_subplot(gs[row_i, 0])
        ax_p.imshow(overlay); ax_p.axis("off")
        ax_p.set_title(
            f"{name}  —  segmentation overlay (α={OVERLAY_ALPHA:.1f})",
            fontsize=15, fontweight="bold", loc="left", pad=6)

        ax_s = fig.add_subplot(gs[row_i, 1])
        ax_s.imshow(sphere_img); ax_s.axis("off")
        ax_s.set_title(f"{name}  —  chrome sphere",
                       fontsize=15, fontweight="bold", pad=6)
        delta_pp = (cov - GT_LIGHT_COV) * 100.0
        if is_gt:
            facecolor = (0.10, 0.50, 0.20, 0.85)
            line3 = "Δ (drift)        =  +0.0 pp"
        else:
            facecolor = drift_color(delta_pp)
            line3 = f"Δ (drift)        = {delta_pp:+5.1f} pp"
        ax_s.text(0.97, 0.97,
                  f"light pixels  = {cov*100:5.1f}%\n"
                  f"GT reference = {GT_LIGHT_COV*100:5.1f}%\n"
                  f"{line3}",
                  transform=ax_s.transAxes, ha="right", va="top",
                  fontsize=12, fontweight="bold", color="white",
                  family="monospace",
                  bbox=dict(boxstyle="round,pad=0.4",
                            facecolor=facecolor, edgecolor="none"))

    # Row 0: GT (perfect baseline)
    _row(0, overlay_gt, chrome_gt, "GT (perfect baseline)",  GT_LIGHT_COV, is_gt=True)
    # Row 1: SphereUFormer
    _row(1, overlay_su, chrome_su, "SphereUFormer (CVPR'25)", cov_su)
    # Row 2: EquiSSL
    _row(2, overlay_ou, chrome_ou, "EquiSSL (Ours)",          cov_ou)

    ax_lg = fig.add_subplot(gs[3, :])
    draw_horizontal_legend(ax_lg, classes_in_fig)

    fig.suptitle(
        f"IBL relighting under camera tilt  —  light = "
        f"{' + '.join(CLASS_NAMES[c] for c in LIGHT_CLASSES)} "
        f"(non-light dimmed ×{NON_LIGHT_DIM:.2f})  ·  θ = {theta:5.2f}°",
        fontsize=18, y=0.985, fontweight="bold")

    fpath = f"{FRAMES_DIR}/frame_{fi:04d}.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)

    if (fi + 1) % 20 == 0 or fi == 0 or fi == NUM_FRAMES - 1:
        elapsed = time.time() - t_start
        rate = (fi + 1) / max(elapsed, 1e-6)
        eta = (NUM_FRAMES - fi - 1) / max(rate, 1e-6)
        print(f"  frame {fi+1:4d}/{NUM_FRAMES}  theta={theta:6.2f}  "
              f"cov SU={cov_su*100:.1f}%  Ours={cov_ou*100:.1f}%  "
              f"rate={rate:.2f} fps  eta={eta:5.1f}s")

print(f"Frame rendering done in {time.time() - t_start:.1f}s.")


# --------- ffmpeg compose ---------
if not args.no_mp4:
    import subprocess
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Composing {VIDEO_PATH} (target 2568x1368, {FPS} fps, h264 yuv420p)...")
    # Frame aspect (~22:13) is taller than 2568x1368 target, so scale by
    # height first and pad horizontally with white.
    cmd = [
        ffmpeg, "-y", "-loglevel", "warning",
        "-framerate", str(FPS),
        "-i", f"{FRAMES_DIR}/frame_%04d.png",
        "-vf",
        "scale=-2:1368:flags=lanczos,"
        "pad=2568:1368:(ow-iw)/2:0:color=white,"
        "format=yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-movflags", "+faststart", "-an",
        VIDEO_PATH,
    ]
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
    print(f"Wrote {VIDEO_PATH} ({sz:.1f} MB)")
