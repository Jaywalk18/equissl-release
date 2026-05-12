"""Rotation equivariance video — 3x6 grid, all samples rotate in parallel.

Unlike render_rotation_video.py (which cycles through samples serially over
3x120=360 frames), this variant shows all three Stanford2D3D samples side-by-
side rotating in sync — layout mirrors seg_comparison_multi.py so the rotation
story reads as a multi-sample panel, not a cherry-picked clip.

Grid (3 rows x 6 cols):
  row r [Input RGB | Ground truth | SphereUFormer | HEAL-SWIN | SO3UFormer | EquiSSL]

Single 360-deg rotation in 120 frames @ 24 fps -> 5 s seamless loop.

NOTE: HEAL-SWIN (col 4) and SO3UFormer (col 5) are LAYOUT PLACEHOLDERS sourced
from internal ablation checkpoints (No-RPE and C2 GE-RPE). Replace
CKPT_HEAL / CKPT_SO3 below once real baselines are trained.

Output: figures/figs/rotation_video_multi.mp4 (prior single-row version at
        rotation_video.mp4 is kept for comparison).
"""
import os, sys, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
import imageio.v2 as iio

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from equissl.utils.sphere import compute_rotation_permutation, apply_rotation_to_features
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

# --------- Config ---------
SAMPLES = [
    ("val",  28),
    ("test", 225),
    ("test", 59),
]
NUM_FRAMES = 120                  # one 360-deg rotation, seamless loop
FPS        = 24                   # 5 s total
CFG        = "configs/pretrain_v8_large.yaml"
OUT_DIR    = "figures/figs"
FRAMES_DIR = f"{OUT_DIR}/rotation_video_multi_frames"
VIDEO_PATH = f"{OUT_DIR}/rotation_video_multi.mp4"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
CKPT_HEAL     = "outputs/none_seed123_v9/best_model.pth"   # placeholder for HEAL-SWIN
CKPT_SO3      = "outputs/c2_seed123_v9/best_model.pth"     # placeholder for SO3UFormer
CKPT_OURS     = "outputs/rpe_ablation_c4_v2/best_model.pth"

SEG_RANK = 5
RGB_RANK = 6

os.makedirs(FRAMES_DIR, exist_ok=True)
for fn in os.listdir(FRAMES_DIR):
    if fn.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, fn))

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK  = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK

# --------- View / lighting ---------
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW  = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])

# --------- Icosphere references ---------
ref = IcoSphereRef("vertex")
img_normals  = np.asarray(ref.get_normals(IMG_RANK),  dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
_, idx_proj_from_img = cKDTree(img_normals).query(proj_normals, k=1)


def precompute_context(rank):
    verts = np.asarray(ref.get_normals(rank), dtype=np.float32)
    faces = np.asarray(ref.get_icosphere(rank, False).faces, dtype=np.int64)
    vf    = verts[faces]
    centroids = vf.mean(axis=1)
    nrm = np.cross(vf[:, 1]-vf[:, 0], vf[:, 2]-vf[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    flip = (centroids * nrm).sum(axis=1) < 0
    nrm[flip] = -nrm[flip]
    visible = (nrm @ VIEW) > -0.05
    vf_vis    = vf[visible]
    shade_vis = 0.75 + 0.25 * np.clip(nrm[visible] @ LIGHT, 0, 1)
    _, idx_from_img  = cKDTree(img_normals).query(verts, k=1)
    _, idx_from_proj = cKDTree(proj_normals).query(verts, k=1)
    return dict(
        verts=verts, faces=faces, vf_vis=vf_vis, shade_vis=shade_vis,
        visible=visible, idx_from_img=idx_from_img, idx_from_proj=idx_from_proj,
    )


ctx_seg = precompute_context(SEG_RANK)
ctx_rgb = precompute_context(RGB_RANK)

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


# --------- Load samples ---------
_nm = cfg["data"]["normalize_mean"]
_ns = cfg["data"]["normalize_std"]
NORM_MEAN = np.full((1, 3), float(_nm), dtype=np.float32) if np.ndim(_nm) == 0 \
            else np.asarray(_nm, dtype=np.float32).reshape(1, 3)
NORM_STD  = np.full((1, 3), float(_ns), dtype=np.float32) if np.ndim(_ns) == 0 \
            else np.asarray(_ns, dtype=np.float32).reshape(1, 3)

_datasets = {}
def _get_ds(split):
    if split not in _datasets:
        _datasets[split] = Stanford2D3DSeg(
            split=split, data_dir="${STANFORD2D3D_PATH}",
            img_rank=IMG_RANK, node_type=mc["node_type"],
            num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
            normalize_mean=cfg["data"]["normalize_mean"],
            normalize_std=cfg["data"]["normalize_std"])
    return _datasets[split]

print(f"Loading {len(SAMPLES)} samples...")
sample_data = []
for (split, idx) in SAMPLES:
    ds = _get_ds(split)
    s = ds[idx]
    sample_data.append(dict(
        rgb=s["sphere_rgb"].unsqueeze(0).cuda(),
        gt=s["sphere_gt_sem"].numpy(),
        label=f"{split} idx={idx}",
    ))
print(f"  loaded: {[sd['label'] for sd in sample_data]}")


# --------- Build models ---------
def build_ours(ckpt_path, rpe_mode="equivariant", n_gauges=4):
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
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=True)
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


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


print(f"Loading baseline: {CKPT_BASELINE}")
sphereu_model = build_sphereu(CKPT_BASELINE)
print(f"Loading HEAL-SWIN placeholder (No-RPE):   {CKPT_HEAL}")
heal_model    = build_ours(CKPT_HEAL, rpe_mode="none", n_gauges=4)
print(f"Loading SO3UFormer placeholder (C2 GE-RPE): {CKPT_SO3}")
so3_model     = build_ours(CKPT_SO3, rpe_mode="equivariant", n_gauges=2)
print(f"Loading ours:     {CKPT_OURS}")
ours_model    = build_ours(CKPT_OURS, rpe_mode="equivariant", n_gauges=4)


@torch.no_grad()
def predict_sphereu(rgb_rot):
    return sphereu_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


@torch.no_grad()
def predict_heal(rgb_rot):
    return heal_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


@torch.no_grad()
def predict_so3(rgb_rot):
    return so3_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


@torch.no_grad()
def predict_ours(rgb_rot):
    return ours_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


def R_yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0,         1, 0],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


# --------- Render helpers ---------
def render_sphere(ax, face_rgb, ctx, title=None, subtitle=None,
                  linewidths=0.25, antialiased=False, title_fontsize=13):
    shaded = (face_rgb[ctx["visible"]] * ctx["shade_vis"][:, None]).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(
        ctx["vf_vis"], facecolors=shaded, edgecolors=shaded,
        linewidths=linewidths, antialiased=antialiased))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    # zoom>1 tightens the cell crop around the unit sphere so the rendered
    # geometry occupies more of the subplot area (each panel feels larger).
    try:
        ax.set_box_aspect([1, 1, 1], zoom=1.4)
    except TypeError:  # older matplotlib without zoom kwarg
        ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()
    if title is not None:
        header = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(header, fontsize=title_fontsize, pad=6, linespacing=1.25)


def seg_face_colors(labels_render, ctx):
    face_labels = labels_render[ctx["faces"]]
    face_vote = np.array([np.bincount(fl, minlength=14).argmax() for fl in face_labels])
    return S2D3D_COLORS[face_vote]


def rgb_face_colors(rgb_per_vertex_render, ctx):
    return rgb_per_vertex_render[ctx["faces"]].mean(axis=1).clip(0, 1)


def rgb_to_display_per_vertex(rgb_tensor_rot, ctx):
    x = rgb_tensor_rot[0].detach().cpu().numpy()
    x = x * NORM_STD + NORM_MEAN
    x = np.clip(x, 0.0, 1.0)
    return x[ctx["idx_from_img"]]


def draw_legend(fig, classes_to_show, ncols_target=7, bottom=0.05, height=0.05):
    ax = fig.add_axes([0.02, bottom, 0.96, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(classes_to_show)
    ncols = min(ncols_target, n)
    nrows = (n + ncols - 1) // ncols
    col_w = 1.0 / ncols
    for i, c in enumerate(classes_to_show):
        r = i // ncols
        col = i % ncols
        x = col * col_w + 0.005
        y_top = 1.0 - r / nrows
        y_h   = 0.75 / nrows
        y     = y_top - y_h - 0.05
        name = "unlabeled" if c == 0 else CLASS_NAMES[c]
        ax.add_patch(Rectangle((x, y), 0.03, y_h, facecolor=S2D3D_COLORS[c],
                               edgecolor=(0.2, 0.2, 0.2), linewidth=0.6))
        ax.text(x + 0.04, y + y_h / 2, name, ha="left", va="center",
                fontsize=11, color=(0.1, 0.1, 0.1))


def frame_pa(pred, gt, ignore=0):
    valid = gt != ignore
    if not valid.any():
        return 0.0
    return float((pred[valid] == gt[valid]).mean())


def frame_miou(pred, gt, num_classes=14, ignore=0):
    valid = gt != ignore
    if not valid.any():
        return 0.0
    p, g = pred[valid], gt[valid]
    ious = []
    for c in range(num_classes):
        if c == ignore:
            continue
        pm, gm = (p == c), (g == c)
        if not gm.any():
            continue
        inter = (pm & gm).sum()
        union = (pm | gm).sum()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else 0.0


# --------- Pre-pass: classes present at >= 0.5% coverage ---------
print("\nPre-pass: scanning classes (4 angles x 3 samples)...")
MIN_COVERAGE = 0.005
classes_in_fig = {0}


def _add_present(arr):
    counts = np.bincount(arr, minlength=14)
    frac = counts / max(arr.size, 1)
    for c in range(14):
        if frac[c] >= MIN_COVERAGE:
            classes_in_fig.add(int(c))


with torch.no_grad():
    for sd in sample_data:
        for a in np.linspace(0, 360, 4, endpoint=False):
            R = R_yaw(a)
            ip = torch.tensor(compute_rotation_permutation(img_normals, R),
                              dtype=torch.long).cuda()
            pp = compute_rotation_permutation(proj_normals, R)
            rr = apply_rotation_to_features(sd["rgb"], ip)
            ps = predict_sphereu(rr)[idx_proj_from_img]
            ph = predict_heal(rr)
            ps3 = predict_so3(rr)
            po = predict_ours(rr)
            gtr = sd["gt"][pp]
            _add_present(gtr.astype(np.int64))
            _add_present(ps.astype(np.int64))
            _add_present(ph.astype(np.int64))
            _add_present(ps3.astype(np.int64))
            _add_present(po.astype(np.int64))
classes_in_fig = sorted(classes_in_fig)
_legend_names = ["unlabeled" if c == 0 else CLASS_NAMES[c] for c in classes_in_fig]
print(f"  {len(classes_in_fig)} classes >= {MIN_COVERAGE*100:.1f}% coverage: {_legend_names}")


# --------- Main loop: 3x4 grid, all samples rotate together ---------
angles = np.linspace(0, 360, NUM_FRAMES, endpoint=False)
print(f"\nRendering {NUM_FRAMES} frames (3x4 grid, all samples rotate in sync, "
      f"step {angles[1]:.2f}deg, duration {NUM_FRAMES/FPS:.1f}s @ {FPS} fps)")

COL_TITLES = ["Input RGB", "Ground truth", "SphereUFormer (CVPR'25)",
              "HEAL-SWIN", "SO3UFormer", "EquiSSL (ours)"]
N_COLS = 6

frame_paths = []
for frame_i, angle_deg in enumerate(angles):
    R = R_yaw(angle_deg)
    img_perm = torch.tensor(
        compute_rotation_permutation(img_normals, R),
        dtype=torch.long).cuda()
    proj_perm = compute_rotation_permutation(proj_normals, R)

    # Compute per-sample rotated content + predictions
    per_row = []
    for sd in sample_data:
        rgb_rot = apply_rotation_to_features(sd["rgb"], img_perm)
        pred_sphereu_img  = predict_sphereu(rgb_rot)
        pred_heal_proj    = predict_heal(rgb_rot)
        pred_so3_proj     = predict_so3(rgb_rot)
        pred_ours_proj    = predict_ours(rgb_rot)
        pred_sphereu_proj = pred_sphereu_img[idx_proj_from_img]
        gt_proj_rot       = sd["gt"][proj_perm]
        ignore_mask       = (gt_proj_rot == 0)

        pa_sphereu = frame_pa(pred_sphereu_proj, gt_proj_rot)
        pa_heal    = frame_pa(pred_heal_proj,    gt_proj_rot)
        pa_so3     = frame_pa(pred_so3_proj,     gt_proj_rot)
        pa_ours    = frame_pa(pred_ours_proj,    gt_proj_rot)

        pred_sphereu_proj = pred_sphereu_proj.copy(); pred_sphereu_proj[ignore_mask] = 0
        pred_heal_proj    = pred_heal_proj.copy();    pred_heal_proj[ignore_mask]    = 0
        pred_so3_proj     = pred_so3_proj.copy();     pred_so3_proj[ignore_mask]     = 0
        pred_ours_proj    = pred_ours_proj.copy();    pred_ours_proj[ignore_mask]    = 0

        gt_render      = gt_proj_rot[ctx_seg["idx_from_proj"]]
        sphereu_render = pred_sphereu_proj[ctx_seg["idx_from_proj"]]
        heal_render    = pred_heal_proj[ctx_seg["idx_from_proj"]]
        so3_render     = pred_so3_proj[ctx_seg["idx_from_proj"]]
        ours_render    = pred_ours_proj[ctx_seg["idx_from_proj"]]
        rgb_vertices_rank6 = rgb_to_display_per_vertex(rgb_rot, ctx_rgb)

        per_row.append(dict(
            label=sd["label"],
            rgb=rgb_vertices_rank6, gt=gt_render,
            sphereu=sphereu_render, heal=heal_render,
            so3=so3_render, ours=ours_render,
            pa_sphereu=pa_sphereu, pa_heal=pa_heal,
            pa_so3=pa_so3, pa_ours=pa_ours,
        ))

    # Wider canvas (6 cols) but keep height; tighter wspace/hspace + zoomed-in
    # spheres recover per-panel size relative to the old 4-col layout.
    fig = plt.figure(figsize=(22.0, 12.0), dpi=110)
    gs = fig.add_gridspec(3, N_COLS, left=0.025, right=0.995,
                          top=0.90, bottom=0.13,
                          wspace=-0.04, hspace=-0.02)

    def _seg_panel(ax, render, title, pa_str):
        render_sphere(ax, seg_face_colors(render, ctx_seg), ctx_seg,
                      title=title if title is not None else pa_str,
                      subtitle=pa_str if title is not None else None,
                      title_fontsize=12)

    for r, row in enumerate(per_row):
        pa_strs = {
            "sphereu": f"pixel acc. = {row['pa_sphereu']:.3f}",
            "heal":    f"pixel acc. = {row['pa_heal']:.3f}",
            "so3":     f"pixel acc. = {row['pa_so3']:.3f}",
            "ours":    f"pixel acc. = {row['pa_ours']:.3f}",
        }

        # Col 0: RGB (with row label on the left)
        ax = fig.add_subplot(gs[r, 0], projection="3d")
        render_sphere(ax, rgb_face_colors(row["rgb"], ctx_rgb), ctx_rgb,
                      title=COL_TITLES[0] if r == 0 else None,
                      linewidths=0.0, antialiased=False, title_fontsize=12)
        ax.text2D(-0.04, 0.5, row["label"],
                  transform=ax.transAxes,
                  fontsize=11, rotation=90,
                  ha="center", va="center",
                  fontweight="bold", color=(0.25, 0.25, 0.25))

        # Col 1: GT
        ax = fig.add_subplot(gs[r, 1], projection="3d")
        render_sphere(ax, seg_face_colors(row["gt"], ctx_seg), ctx_seg,
                      title=COL_TITLES[1] if r == 0 else None,
                      title_fontsize=12)

        # Cols 2..5: model predictions
        for ci, (key, ttl) in enumerate(
                [("sphereu", COL_TITLES[2]),
                 ("heal",    COL_TITLES[3]),
                 ("so3",     COL_TITLES[4]),
                 ("ours",    COL_TITLES[5])], start=2):
            ax = fig.add_subplot(gs[r, ci], projection="3d")
            _seg_panel(ax, row[key],
                       ttl if r == 0 else None,
                       pa_strs[key])

    fig.suptitle(
        "Rotation equivariance under continuous yaw rotation  —  "
        r"$\theta = $" + f"{angle_deg:5.1f}" + r"$\degree$",
        fontsize=15, y=0.975, fontweight="bold")

    draw_legend(fig, classes_in_fig, ncols_target=7, bottom=0.045, height=0.07)

    fpath = f"{FRAMES_DIR}/frame_{frame_i:03d}.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    frame_paths.append(fpath)
    if (frame_i + 1) % 10 == 0 or frame_i == 0:
        row_summary = "  ".join(
            f"[{row['label']}: SU={row['pa_sphereu']:.3f} Ours={row['pa_ours']:.3f}]"
            for row in per_row)
        print(f"  frame {frame_i+1:3d}/{NUM_FRAMES}  theta={angle_deg:5.1f}  "
              f"{row_summary}")


# --------- Stitch ---------
print(f"\nStitching {len(frame_paths)} frames @ {FPS} fps -> {VIDEO_PATH}")
writer = iio.get_writer(VIDEO_PATH, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=8)
for fp in frame_paths:
    writer.append_data(iio.imread(fp))
writer.close()
print(f"Saved {VIDEO_PATH}")
print(f"  duration: {NUM_FRAMES / FPS:.1f}s  ({NUM_FRAMES} frames @ {FPS} fps, seamless loop)")
