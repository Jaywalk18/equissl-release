"""Smooth continuous yaw (0->360deg) two-panel ERP video — v2.

Top:    SphereUFormer (CVPR'25, published baseline checkpoint).
Bottom: EquiSSL random-init GE-RPE C_4 (rpe_ablation_c4_v2).

v2 changes vs first cut:
- Switch EquiSSL ckpt from C6-noarea -> C4 (paper main variant; consistent
  with seg_comparison.py and render_rotation_video_multi.py).
- Switch sample from val idx=28 to val idx=22 (chair/bookcase/clutter
  rich panorama, 48.2% target-class coverage; differences read more
  clearly).
- alpha 0.5 -> 0.7 (seg color dominates over panorama texture).
- Add per-frame PA + mIoU numeric overlay per row.
- Enlarge GT panel; shrink class legend.

Output: figures/figs/smooth_yaw_360_10s.mp4 (10s, 36fps, 360 frames).

Run:
    GPU_ID=2 python figures/render_smooth_yaw_360.py [--frames N] [--no-mp4]
        [--sample-idx N] [--sample-split val|test]
"""
import os
import sys
import argparse
import yaml
import time

GPU_ID = os.environ.get("GPU_ID", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
import imageio.v2 as iio

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"]  = ["Times New Roman", "Liberation Serif"]

# --------- Config ---------
SAMPLE_SPLIT = "val"
SAMPLE_IDX   = 17   # max EquiSSL-vs-SphereU PA gap on val (+0.561 over 5 yaw angles)
NUM_FRAMES   = 360  # 1deg per frame
FPS          = 36   # 360 / 36 = 10s
CFG          = "configs/pretrain_v8_large.yaml"
OVERLAY_ALPHA = 0.7

OUT_DIR    = "figures/figs"
FRAMES_DIR = f"{OUT_DIR}/smooth_yaw_360_frames"
VIDEO_PATH = f"{OUT_DIR}/smooth_yaw_360_10s.mp4"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
CKPT_OURS     = "outputs/rpe_ablation_c4_v2/best_model.pth"   # EquiSSL C4 (paper main)
OURS_N_GAUGES = 4
OURS_AREA_W   = True

# ERP grid for overlay rendering
ERP_W = 1024
ERP_H = 256

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
    p.add_argument("--frames", type=int, default=NUM_FRAMES,
                   help="Number of frames (use small N for dry-run)")
    p.add_argument("--no-mp4", action="store_true",
                   help="Skip ffmpeg compose at the end")
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


# --------- Build ERP-grid -> nearest-vertex maps ---------
def make_erp_to_vertex(vertex_xyz, H=ERP_H, W=ERP_W):
    """For each ERP pixel (h, w), compute the nearest vertex index.

    Y-up convention to match this codebase's R_yaw (rotation around Y axis):
    - latitude controls Y (north pole at +Y)
    - longitude controls (X, Z) plane
    """
    lon = (np.arange(W) + 0.5) / W * 2.0 * np.pi - np.pi   # [-pi, pi)
    lat = np.pi / 2.0 - (np.arange(H) + 0.5) / H * np.pi   # [pi/2, -pi/2)
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
RGB_TENSOR = sample["sphere_rgb"].unsqueeze(0).cuda()  # [1, N_img, 3]
GT_PROJ    = sample["sphere_gt_sem"].numpy()           # [N_proj]

# RGB at IMG_RANK in display range (un-normalized) for ERP background
_nm = cfg["data"]["normalize_mean"]
_ns = cfg["data"]["normalize_std"]
NORM_MEAN = np.full((1, 3), float(_nm), dtype=np.float32) if np.ndim(_nm) == 0 \
            else np.asarray(_nm, dtype=np.float32).reshape(1, 3)
NORM_STD  = np.full((1, 3), float(_ns), dtype=np.float32) if np.ndim(_ns) == 0 \
            else np.asarray(_ns, dtype=np.float32).reshape(1, 3)


def rgb_vertex_display(rgb_tensor):
    """Un-normalize the rotated RGB tensor back to [0,1] per-vertex array."""
    x = rgb_tensor[0].detach().cpu().numpy()
    return np.clip(x * NORM_STD + NORM_MEAN, 0.0, 1.0)


# Static GT panel (in original orientation, never rotates — reference)
gt_static_erp = S2D3D_COLORS[GT_PROJ[erp_from_proj]]


# --------- Build models ---------
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


print(f"Loading SphereUFormer:        {CKPT_BASELINE}")
sphereu_model = build_sphereu(CKPT_BASELINE)
print(f"Loading EquiSSL C{OURS_N_GAUGES} (area_weighted={OURS_AREA_W}): {CKPT_OURS}")
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


# --------- Pre-pass: classes present (for legend) ---------
print("Pre-pass: scanning classes (4 angles)...")
MIN_COVERAGE = 0.005
classes_in_fig = {0}
with torch.no_grad():
    for a in np.linspace(0, 360, 4, endpoint=False):
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


# --------- Frame rendering ---------
def overlay_erp(rgb_erp, label_erp, alpha=0.5):
    """Alpha-blend per-class color onto rgb background."""
    color = S2D3D_COLORS[label_erp]
    out = (1.0 - alpha) * rgb_erp + alpha * color
    return np.clip(out, 0.0, 1.0)


def frame_pa(pred, gt, ignore=0):
    valid = gt != ignore
    return float((pred[valid] == gt[valid]).mean()) if valid.any() else 0.0


def frame_miou(pred, gt, num_classes=14, ignore=0):
    valid = gt != ignore
    if not valid.any(): return 0.0
    p, g = pred[valid], gt[valid]
    ious = []
    for c in range(num_classes):
        if c == ignore: continue
        pm, gm = (p == c), (g == c)
        if not gm.any(): continue
        inter = (pm & gm).sum()
        union = (pm | gm).sum()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else 0.0


def draw_horizontal_legend(ax, classes_to_show):
    """Single horizontal legend strip, shared by both rows."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(classes_to_show)
    if n == 0:
        return
    cell_w = 1.0 / n
    swatch_w = 0.024
    swatch_h = 0.60
    y0 = 0.20
    for i, c in enumerate(classes_to_show):
        x = i * cell_w + 0.005
        name = "unlabeled" if c == 0 else CLASS_NAMES[c]
        ax.add_patch(Rectangle((x, y0), swatch_w, swatch_h,
                               facecolor=S2D3D_COLORS[c],
                               edgecolor=(0.2, 0.2, 0.2), linewidth=0.7))
        ax.text(x + swatch_w + 0.005, y0 + swatch_h / 2, name,
                ha="left", va="center",
                fontsize=13, color=(0.1, 0.1, 0.1))


os.makedirs(FRAMES_DIR, exist_ok=True)
for fn in os.listdir(FRAMES_DIR):
    if fn.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, fn))

angles = np.linspace(0, 360, NUM_FRAMES, endpoint=False)
print(f"\nRendering {NUM_FRAMES} frames at {FPS} fps "
      f"(step {angles[1]-angles[0]:.2f} deg, total {NUM_FRAMES/FPS:.2f}s)")

t_start = time.time()
for fi, theta in enumerate(angles):
    R = R_yaw(theta)
    img_perm = torch.tensor(
        compute_rotation_permutation(img_normals, R),
        dtype=torch.long).cuda()
    proj_perm = compute_rotation_permutation(proj_normals, R)

    rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
    pred_su_img  = predict_sphereu(rgb_rot)
    pred_su_proj = pred_su_img[idx_proj_from_img]
    pred_ou_proj = predict_ours(rgb_rot)
    rgb_disp_v   = rgb_vertex_display(rgb_rot)

    # Per-frame metrics vs rotated GT (in rotated frame, both rotated together)
    gt_proj_rot = GT_PROJ[proj_perm]
    pa_su   = frame_pa(pred_su_proj, gt_proj_rot)
    pa_ours = frame_pa(pred_ou_proj, gt_proj_rot)
    miou_su   = frame_miou(pred_su_proj, gt_proj_rot)
    miou_ours = frame_miou(pred_ou_proj, gt_proj_rot)

    # Mask predictions on GT-unknown regions to class 0 (unlabeled / grey),
    # so the overlay matches the GT panel on these pixels — consistent with
    # how figures/make_seg_comparison_multi.py handles ignored regions.
    # Metrics above are computed on the un-masked predictions (frame_pa /
    # frame_miou already exclude GT==0).
    ignore_mask = (gt_proj_rot == 0)
    pred_su_proj_disp = pred_su_proj.copy(); pred_su_proj_disp[ignore_mask] = 0
    pred_ou_proj_disp = pred_ou_proj.copy(); pred_ou_proj_disp[ignore_mask] = 0

    # Project to ERP grid (rotated frame — both rgb and labels are rotated together)
    rgb_erp = rgb_disp_v[erp_from_img]
    su_erp  = pred_su_proj_disp[erp_from_proj]
    ou_erp  = pred_ou_proj_disp[erp_from_proj]

    overlay_su = overlay_erp(rgb_erp, su_erp, alpha=OVERLAY_ALPHA)
    overlay_ou = overlay_erp(rgb_erp, ou_erp, alpha=OVERLAY_ALPHA)

    # --- Compose frame ---
    fig = plt.figure(figsize=(20.0, 9.0), dpi=120)
    # 2 ERP cols (rotated overlay | GT ref), shared horizontal legend at bottom.
    # Wider top/bottom margins for breathing room around title and theta.
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 0.15],
        left=0.020, right=0.990,
        top=0.86, bottom=0.10,
        wspace=0.05, hspace=0.30,
    )

    def _row(row_i, overlay, name, pa, miou):
        ax = fig.add_subplot(gs[row_i, 0])
        ax.imshow(overlay); ax.axis("off")
        ax.set_title(f"{name}  —  segmentation overlay (α={OVERLAY_ALPHA:.1f})",
                     fontsize=17, fontweight="bold", loc="left", pad=8)
        ax.text(0.99, 0.96,
                f"PA = {pa:.3f}\nmIoU = {miou:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=15, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=(0.15, 0.15, 0.15, 0.75),
                          edgecolor="none"))
        ax_gt = fig.add_subplot(gs[row_i, 1]); ax_gt.imshow(gt_static_erp); ax_gt.axis("off")
        ax_gt.set_title("GT (ref, static, original orientation)",
                        fontsize=14, pad=8)

    _row(0, overlay_su, "SphereUFormer (CVPR'25)", pa_su,   miou_su)
    _row(1, overlay_ou, "EquiSSL (Ours)",          pa_ours, miou_ours)

    # Shared horizontal legend spanning both columns
    ax_lg = fig.add_subplot(gs[2, :])
    draw_horizontal_legend(ax_lg, classes_in_fig)

    fig.suptitle("Continuous yaw rotation (360°)  —  same input scene",
                 fontsize=20, y=0.955, fontweight="bold")
    fig.text(0.5, 0.025, rf"$\theta$ = {theta:6.1f}°",
             ha="center", va="bottom", fontsize=18, fontweight="bold",
             color=(0.15, 0.15, 0.15))

    fpath = f"{FRAMES_DIR}/frame_{fi:04d}.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)

    if (fi + 1) % 20 == 0 or fi == 0 or fi == NUM_FRAMES - 1:
        elapsed = time.time() - t_start
        rate = (fi + 1) / max(elapsed, 1e-6)
        eta = (NUM_FRAMES - fi - 1) / max(rate, 1e-6)
        print(f"  frame {fi+1:4d}/{NUM_FRAMES}  theta={theta:6.1f}  "
              f"rate={rate:.2f} fps  eta={eta:5.1f}s")

print(f"Frame rendering done in {time.time() - t_start:.1f}s.")


# --------- ffmpeg compose (with pad to 2568x1368) ---------
if not args.no_mp4:
    import subprocess
    print(f"Composing {VIDEO_PATH} (target 2568x1368, {FPS} fps, h264 yuv420p)...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", f"{FRAMES_DIR}/frame_%04d.png",
        "-vf",
        "scale=2568:-2:flags=lanczos,"
        "pad=2568:1368:0:(oh-ih)/2:color=white,"
        "format=yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-movflags", "+faststart",
        "-an",  # no audio
        VIDEO_PATH,
    ]
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
    print(f"Wrote {VIDEO_PATH} ({sz:.1f} MB)")
