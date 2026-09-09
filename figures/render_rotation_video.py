"""Rotation equivariance video — 4-panel, multi-sample seamless segments.

Panels (1x4):
  (a) Input RGB (rotated)
  (b) Ground truth (rotated)
  (c) SphereUFormer (CVPR'25)
  (d) EquiSSL (ours)

Cycles through 3 Stanford2D3D samples (val idx=28 + test idx=225 +
test idx=59) to demonstrate the rotation-robustness pattern holds across
samples — mirrors the 3-sample breadth of seg_comparison_multi.py so the
rotation story cannot be dismissed as cherry-picked. Each sample rotates
360 deg in 120 frames (3 deg/frame, 5 s per sample).

Metric shown: pixel accuracy (per-sample mIoU is inherently low due to
rare-class averaging; pa matches sample_ranking_all.json values).

RGB panel uses antialiased polygons + thicker edge stroke (linewidths=0.6)
to suppress the sub-pixel moire that appeared on sparse rank-5 meshes.

Output: figures/figs/rotation_video.mp4
        figures/figs/rotation_video_frames/frame_<NNN>.png  (intermediate)
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
    ("val",  28),    # main — same as seg_comparison.py / fig1
    ("test", 225),   # seg_comparison_multi sample A
    ("test", 59),    # seg_comparison_multi sample B
]
FRAMES_PER_SAMPLE = 120       # 3 deg/frame -> one full 360 deg rotation per sample
FPS               = 24        # 120 frames / sample @ 24 fps = 5 s per sample
NUM_FRAMES        = FRAMES_PER_SAMPLE * len(SAMPLES)  # 360 total
CFG               = "configs/pretrain_v8_large.yaml"
OUT_DIR           = "figures/figs"
FRAMES_DIR        = f"{OUT_DIR}/rotation_video_frames"
VIDEO_PATH        = f"{OUT_DIR}/rotation_video.mp4"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
CKPT_OURS     = "outputs/rpe_ablation_c4_v2/best_model.pth"

SEG_RANK = 5   # 10,242 verts — low-frequency solid-color regions, no moire risk
RGB_RANK = 6   # 40,962 verts — dense enough that face size < pixel; kills moire

os.makedirs(FRAMES_DIR, exist_ok=True)
for fn in os.listdir(FRAMES_DIR):
    if fn.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, fn))

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK  = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK

# --------- View / lighting (shared by all panels) ---------
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW  = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])

# --------- Icosphere references + precomputed render geometry ---------
ref = IcoSphereRef("vertex")
img_normals    = np.asarray(ref.get_normals(IMG_RANK),  dtype=np.float32)
proj_normals   = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)

_, idx_proj_from_img = cKDTree(img_normals).query(proj_normals, k=1)


def precompute_context(rank):
    """Precompute render geometry (static — content rotates, verts don't)."""
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


ctx_seg = precompute_context(SEG_RANK)  # seg panels (GT / SphereU / EquiSSL)
ctx_rgb = precompute_context(RGB_RANK)  # RGB panel (denser, no moire)

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


# --------- Load all samples ---------
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
        rgb=s["sphere_rgb"].unsqueeze(0).cuda(),  # (1, N_img, 3)
        gt=s["sphere_gt_sem"].numpy(),             # (N_proj,)
        label=f"{split} idx={idx}",
    ))
print(f"  loaded: {[sd['label'] for sd in sample_data]}")


# --------- Build both models ---------
def build_ours(ckpt_path):
    enc = SphericalEncoder(
        img_rank=IMG_RANK, node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=4, area_weighted=True)
    m = EquiSSLSegUNet(encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths", [2, 2, 2, 2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16, 16, 8, 4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=True, n_gauges=4, area_weighted=True)
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
print(f"Loading ours:     {CKPT_OURS}")
ours_model    = build_ours(CKPT_OURS)


@torch.no_grad()
def predict_sphereu(rgb_rot):
    return sphereu_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]  # (N_img,)


@torch.no_grad()
def predict_ours(rgb_rot):
    return ours_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]     # (N_proj,)


def R_yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0,         1, 0],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


# --------- Render helpers ---------
def render_sphere(ax, face_rgb, ctx, title, subtitle=None,
                  linewidths=0.25, antialiased=False):
    shaded = (face_rgb[ctx["visible"]] * ctx["shade_vis"][:, None]).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(
        ctx["vf_vis"], facecolors=shaded, edgecolors=shaded,
        linewidths=linewidths, antialiased=antialiased))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()
    header = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(header, fontsize=14, pad=8, linespacing=1.3)


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


def draw_legend(fig, classes_to_show, ncols_target=7, bottom=0.10, height=0.08):
    """Horizontal legend strip — positioned higher than v1 (bottom 0.015 -> 0.10)."""
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


# --------- Pre-pass: classes present at >= 0.5% coverage across all samples ---------
print("\nPre-pass: scanning classes across samples (4 angles per sample)...")
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
            po = predict_ours(rr)
            gtr = sd["gt"][pp]
            _add_present(gtr.astype(np.int64))
            _add_present(ps.astype(np.int64))
            _add_present(po.astype(np.int64))
classes_in_fig = sorted(classes_in_fig)
_legend_names = ["unlabeled" if c == 0 else CLASS_NAMES[c] for c in classes_in_fig]
print(f"  {len(classes_in_fig)} classes >= {MIN_COVERAGE*100:.1f}% coverage: {_legend_names}")


# --------- Main frame loop ---------
angles_per_sample = np.linspace(0, 360, FRAMES_PER_SAMPLE, endpoint=False)
print(f"\nRendering {NUM_FRAMES} frames ({len(SAMPLES)} samples x "
      f"{FRAMES_PER_SAMPLE} frames, step {angles_per_sample[1]:.2f}deg, "
      f"duration {NUM_FRAMES/FPS:.1f}s @ {FPS} fps)")

frame_paths = []
frame_i = 0
for sd_i, sd in enumerate(sample_data):
    rgb_base = sd["rgb"]
    gt_proj  = sd["gt"]
    for angle_deg in angles_per_sample:
        R = R_yaw(angle_deg)
        img_perm = torch.tensor(
            compute_rotation_permutation(img_normals, R),
            dtype=torch.long).cuda()
        proj_perm = compute_rotation_permutation(proj_normals, R)

        rgb_rot = apply_rotation_to_features(rgb_base, img_perm)

        pred_sphereu_img  = predict_sphereu(rgb_rot)
        pred_ours_proj    = predict_ours(rgb_rot)
        pred_sphereu_proj = pred_sphereu_img[idx_proj_from_img]

        gt_proj_rot = gt_proj[proj_perm]
        ignore_mask = (gt_proj_rot == 0)

        pa_sphereu   = frame_pa(pred_sphereu_proj, gt_proj_rot)
        pa_ours      = frame_pa(pred_ours_proj,    gt_proj_rot)
        miou_sphereu = frame_miou(pred_sphereu_proj, gt_proj_rot)
        miou_ours    = frame_miou(pred_ours_proj,    gt_proj_rot)

        pred_sphereu_proj = pred_sphereu_proj.copy(); pred_sphereu_proj[ignore_mask] = 0
        pred_ours_proj    = pred_ours_proj.copy();    pred_ours_proj[ignore_mask]    = 0

        # Seg upsampling at rank 5
        gt_render      = gt_proj_rot[ctx_seg["idx_from_proj"]]
        sphereu_render = pred_sphereu_proj[ctx_seg["idx_from_proj"]]
        ours_render    = pred_ours_proj[ctx_seg["idx_from_proj"]]
        # RGB upsampling at rank 6 (denser mesh, no moire)
        rgb_vertices_rank6 = rgb_to_display_per_vertex(rgb_rot, ctx_rgb)

        fig = plt.figure(figsize=(16.0, 7.2), dpi=110)
        gs = fig.add_gridspec(1, 4, left=0.01, right=0.99,
                              top=0.89, bottom=0.26, wspace=0.04)

        ax_rgb = fig.add_subplot(gs[0, 0], projection="3d")
        render_sphere(ax_rgb, rgb_face_colors(rgb_vertices_rank6, ctx_rgb), ctx_rgb,
                      "Input RGB",
                      r"$\theta = $" + f"{angle_deg:5.1f}" + r"$\degree$",
                      linewidths=0.0, antialiased=False)  # rank-6 dense mesh -> no moire

        ax_gt = fig.add_subplot(gs[0, 1], projection="3d")
        render_sphere(ax_gt, seg_face_colors(gt_render, ctx_seg), ctx_seg,
                      "Ground truth", "(reference)")

        ax_bl = fig.add_subplot(gs[0, 2], projection="3d")
        render_sphere(ax_bl, seg_face_colors(sphereu_render, ctx_seg), ctx_seg,
                      "SphereUFormer (CVPR'25)",
                      f"pixel acc. = {pa_sphereu:.3f}")

        ax_ours = fig.add_subplot(gs[0, 3], projection="3d")
        render_sphere(ax_ours, seg_face_colors(ours_render, ctx_seg), ctx_seg,
                      "EquiSSL (ours)",
                      f"pixel acc. = {pa_ours:.3f}")

        fig.suptitle(
            f"Rotation equivariance under continuous yaw rotation  —  "
            f"Sample {sd_i+1}/{len(SAMPLES)} ({sd['label']})",
            fontsize=14, y=0.96, fontweight="bold")

        draw_legend(fig, classes_in_fig, ncols_target=7, bottom=0.14, height=0.09)

        fpath = f"{FRAMES_DIR}/frame_{frame_i:03d}.png"
        plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(fpath)
        if (frame_i + 1) % 10 == 0 or frame_i == 0:
            print(f"  frame {frame_i+1:3d}/{NUM_FRAMES}  "
                  f"sample {sd_i+1}/{len(SAMPLES)} theta={angle_deg:5.1f}  "
                  f"pa(SU)={pa_sphereu:.3f} pa(Ours)={pa_ours:.3f}  "
                  f"miou(SU)={miou_sphereu:.3f} miou(Ours)={miou_ours:.3f}")
        frame_i += 1


# --------- Stitch into MP4 ---------
print(f"\nStitching {len(frame_paths)} frames @ {FPS} fps -> {VIDEO_PATH}")
writer = iio.get_writer(VIDEO_PATH, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=8)
for fp in frame_paths:
    writer.append_data(iio.imread(fp))
writer.close()
print(f"Saved {VIDEO_PATH}")
print(f"  duration: {NUM_FRAMES / FPS:.1f}s  ({NUM_FRAMES} frames @ {FPS} fps)")
print(f"  per-sample: {FRAMES_PER_SAMPLE/FPS:.1f}s each, "
      f"{FRAMES_PER_SAMPLE}-frame rotation from 0 to {angles_per_sample[-1]:.1f} deg")
