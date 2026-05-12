"""Clean per-method icosphere PNGs for PPT redraw (Fig 1 / Fig 2).

Per `paper/prompts/server_prompt_clean_spheres.md`: dump single-cell
PNGs the author can drop into PowerPoint without cropping the existing
multi-method video frames (which have neighbouring spheres bleeding in
and a `pixel acc. = X.XXX` annotation overlay).

Re-uses the rendering infrastructure from
`render_rotation_video_multi.py` (precompute_context, render_sphere,
seg_face_colors). Strips: no titles, no axis decorations, no pixel-
accuracy text, no grid, no legend. White background. Each sphere disc
fills ~85% of the 1024x1024 canvas via figure margins + box_aspect
zoom=1.4 (same view geometry as the multi-method video so visual style
is consistent across the paper).

Core deliverable: 3 methods x 3 angles x 1 scene = 9 PNGs.
Optional extras: rotated panorama (val_17 @ 45°, 2048x1024), and a
6-angle EquiSSL sweep strip {0,18,36,54,72,90}° as separate PNGs.

EquiSSL canonical substitution: prompt asks for the iBOT+MAE × C₆
no-area, seed 42, val 68.30 checkpoint. That checkpoint does not exist
as actually trained (68.30 is a paper-side derived target). Closest
available is `outputs/rpe_ablation_c6_noarea/best_model.pth` —
random-init C₆ no-area at val 67.82. Same architecture, real measured
performance. Audit note is shipped alongside the PNGs.

Run:
    GPU_ID=0 python figures/render_clean_spheres.py [--scope core|all]
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
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
SAMPLE_IDX   = 17
SCENE_TAG    = "val17"
CFG          = "configs/pretrain_v8_large.yaml"

CKPT_BASELINE = "outputs/sphereuformer_baseline/best_model.pth"
# EquiSSL canonical for the paper-main figure. The prompt requests the
# 68.30 iBOT+MAE × C₆ no-area run; that does not exist as a trained ckpt.
# rpe_ablation_c6_noarea (random-init C₆ no-area, val 67.82) is the
# closest actually-measured proxy; see audit note for full provenance.
CKPT_EQUISSL  = "outputs/rpe_ablation_c6_noarea/best_model.pth"
EQUISSL_N_GAUGES   = 6
EQUISSL_AREA_W     = False    # "no-area"

SEG_RANK = 6   # was 5; bumped for finer triangle resolution → smoother
               # class boundaries (4× more faces).
RGB_RANK = 6
SUPERSAMPLE = 2   # render at 2× then PIL-LANCZOS downsample for clean AA

# Canvas
CANVAS_PX = 1024              # square output
PANORAMA_W = 2048
PANORAMA_H = 1024

CORE_YAWS  = [0, 45, 90]
SWEEP_YAWS = [0, 18, 36, 54, 72, 90]

OUT_DIR = "figures/ppt_assets/server_delivered"

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
    p.add_argument("--scope", choices=["core", "all"], default="all",
                   help="core = 9 PNGs only; all = core + panorama + sweep")
    return p.parse_args()


# --------- View / lighting (matches render_rotation_video_multi.py) ---------
AZ, EL = np.deg2rad(30), np.deg2rad(20)
VIEW  = np.array([np.cos(EL)*np.cos(AZ), np.cos(EL)*np.sin(AZ), np.sin(EL)])
LIGHT = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])


# --------- Init ---------
args = parse_args()

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK  = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1 if mc.get("in_scale_factor", 2) == 2 else IMG_RANK

ref = IcoSphereRef("vertex")
img_normals  = np.asarray(ref.get_normals(IMG_RANK),  dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)
_, idx_proj_from_img = cKDTree(img_normals).query(proj_normals, k=1)


def precompute_context(rank):
    verts = np.asarray(ref.get_normals(rank), dtype=np.float32)
    faces = np.asarray(ref.get_icosphere(rank, False).faces, dtype=np.int64)
    vf    = verts[faces]
    centroids = vf.mean(axis=1)
    nrm = np.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    flip = (centroids * nrm).sum(axis=1) < 0
    nrm[flip] = -nrm[flip]
    visible   = (nrm @ VIEW) > -0.05
    vf_vis    = vf[visible]
    shade_vis = 0.75 + 0.25 * np.clip(nrm[visible] @ LIGHT, 0, 1)
    _, idx_from_img  = cKDTree(img_normals).query(verts, k=1)
    _, idx_from_proj = cKDTree(proj_normals).query(verts, k=1)
    return dict(
        verts=verts, faces=faces, vf_vis=vf_vis, shade_vis=shade_vis,
        visible=visible, idx_from_img=idx_from_img, idx_from_proj=idx_from_proj,
    )


print("Precomputing render contexts...")
ctx_seg = precompute_context(SEG_RANK)
ctx_rgb = precompute_context(RGB_RANK)

# ERP -> vertex map for the optional rotated-panorama deliverable.
def make_erp_to_vertex(vertex_xyz, H=PANORAMA_H, W=PANORAMA_W):
    """Y-up convention to match codebase R_yaw."""
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


erp_from_img = make_erp_to_vertex(img_normals)


# --------- Load sample ---------
print(f"Loading Stanford2D3D {SAMPLE_SPLIT} idx={SAMPLE_IDX}...")
ds = Stanford2D3DSeg(
    split=SAMPLE_SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=IMG_RANK, node_type=mc["node_type"],
    num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"],
    normalize_std=cfg["data"]["normalize_std"])
sample = ds[SAMPLE_IDX]
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


print(f"Loading SphereUFormer: {CKPT_BASELINE}")
sphereu_model = build_sphereu(CKPT_BASELINE)
print(f"Loading EquiSSL (C{EQUISSL_N_GAUGES} "
      f"{'area' if EQUISSL_AREA_W else 'no-area'}): {CKPT_EQUISSL}")
equissl_model = build_ours(CKPT_EQUISSL, n_gauges=EQUISSL_N_GAUGES,
                           area_weighted=EQUISSL_AREA_W)


@torch.no_grad()
def predict_sphereu(rgb_rot):
    return sphereu_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


@torch.no_grad()
def predict_equissl(rgb_rot):
    return equissl_model(rgb_rot).argmax(dim=-1).cpu().numpy()[0]


def R_yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [ 0,         1, 0        ],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


# --------- Sphere render (no titles, no overlays, no axes) ---------
def seg_face_colors(labels_render, ctx):
    face_labels = labels_render[ctx["faces"]]
    face_vote = np.array([np.bincount(fl, minlength=14).argmax()
                          for fl in face_labels])
    return S2D3D_COLORS[face_vote]


def render_clean_sphere_png(face_rgb, ctx, out_path,
                            canvas_px=CANVAS_PX, zoom=1.55,
                            supersample=SUPERSAMPLE):
    """Render a single icosphere coloured by `face_rgb` and save as PNG.

    Anti-aliasing pipeline (3 layers):
      1. SEG_RANK=6 — 4× more triangles than the video renderer, so
         class boundaries are made of finer faces.
      2. matplotlib antialiased=True — smooths edge rasterization.
      3. Super-sample at `supersample`× canvas and PIL-LANCZOS downsample
         — kills residual pixel-grid aliasing.

    `zoom` > 1 tightens crop around the unit sphere — 1.55 gives a disc
    that fills ≈ 85% of the canvas.
    """
    shaded = (face_rgb[ctx["visible"]] * ctx["shade_vis"][:, None]).clip(0, 1)

    super_px = canvas_px * supersample
    figsize = (super_px / 100.0, super_px / 100.0)
    fig = plt.figure(figsize=figsize, dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.add_collection3d(Poly3DCollection(
        ctx["vf_vis"], facecolors=shaded, edgecolors=shaded,
        linewidths=0.4, antialiased=True))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    try:
        ax.set_box_aspect([1, 1, 1], zoom=zoom)
    except TypeError:
        ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.set_axis_off()

    # Save to in-memory PNG buffer, then PIL-LANCZOS downsample to target.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != (canvas_px, canvas_px):
        img = img.resize((canvas_px, canvas_px), Image.LANCZOS)
    img.save(out_path, format="PNG", optimize=True)


# --------- Main render loop ---------
os.makedirs(OUT_DIR, exist_ok=True)
manifest = []
t_start = time.time()

for yaw_deg in CORE_YAWS:
    print(f"\n=== yaw = {yaw_deg}° ===")
    R = R_yaw(yaw_deg)
    img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                            dtype=torch.long).cuda()
    proj_perm = compute_rotation_permutation(proj_normals, R)

    rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
    pred_su_img  = predict_sphereu(rgb_rot)
    pred_su_proj = pred_su_img[idx_proj_from_img]
    pred_eq_proj = predict_equissl(rgb_rot)
    gt_proj_rot  = GT_PROJ[proj_perm]

    # Apply class-0 (unknown) mask to predictions for visual consistency
    # with paper-main figures (make_seg_comparison_multi.py convention).
    ignore_mask = (gt_proj_rot == 0)
    pred_su_disp = pred_su_proj.copy(); pred_su_disp[ignore_mask] = 0
    pred_eq_disp = pred_eq_proj.copy(); pred_eq_disp[ignore_mask] = 0

    # Lift proj-rank labels onto seg-rank vertices for rendering
    labels_gt = gt_proj_rot[ctx_seg["idx_from_proj"]]
    labels_su = pred_su_disp[ctx_seg["idx_from_proj"]]
    labels_eq = pred_eq_disp[ctx_seg["idx_from_proj"]]

    method_labels = [
        ("gt",            labels_gt),
        ("sphereuformer", labels_su),
        ("equissl",       labels_eq),
    ]
    for method_name, labels in method_labels:
        face_rgb = seg_face_colors(labels, ctx_seg)
        fname = f"{method_name}_{SCENE_TAG}_yaw{yaw_deg:02d}.png"
        path = os.path.join(OUT_DIR, fname)
        render_clean_sphere_png(face_rgb, ctx_seg, path)
        sz_kb = os.path.getsize(path) / 1024
        manifest.append((fname, sz_kb))
        print(f"  wrote {fname}  ({sz_kb:.1f} KB)")


# --------- Optional: rotated panorama at 45° ---------
if args.scope == "all":
    print("\n=== optional: rotated panorama val17 yaw 45° ===")
    R = R_yaw(45.0)
    img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                            dtype=torch.long).cuda()
    rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
    rgb_disp_v = rgb_vertex_display(rgb_rot)
    panorama = rgb_disp_v[erp_from_img]   # (H, W, 3)
    panorama = (panorama * 255).clip(0, 255).astype(np.uint8)

    from PIL import Image
    pan_path = os.path.join(OUT_DIR, f"rotated_panorama_{SCENE_TAG}_yaw45.png")
    Image.fromarray(panorama).save(pan_path)
    sz_kb = os.path.getsize(pan_path) / 1024
    manifest.append((os.path.basename(pan_path), sz_kb))
    print(f"  wrote {os.path.basename(pan_path)} ({PANORAMA_W}x{PANORAMA_H}, "
          f"{sz_kb:.1f} KB)")

    # --------- Optional: 6-angle EquiSSL sweep ---------
    print("\n=== optional: EquiSSL 6-angle sweep ===")
    for yaw_deg in SWEEP_YAWS:
        if yaw_deg in CORE_YAWS:
            # Skip duplicates already in core (0, 90)
            continue
        R = R_yaw(yaw_deg)
        img_perm = torch.tensor(compute_rotation_permutation(img_normals, R),
                                dtype=torch.long).cuda()
        proj_perm = compute_rotation_permutation(proj_normals, R)
        rgb_rot = apply_rotation_to_features(RGB_TENSOR, img_perm)
        pred_eq_proj = predict_equissl(rgb_rot)
        gt_proj_rot = GT_PROJ[proj_perm]
        ignore_mask = (gt_proj_rot == 0)
        pred_eq_disp = pred_eq_proj.copy(); pred_eq_disp[ignore_mask] = 0
        labels_eq = pred_eq_disp[ctx_seg["idx_from_proj"]]
        face_rgb = seg_face_colors(labels_eq, ctx_seg)
        fname = f"equissl_{SCENE_TAG}_yaw{yaw_deg:02d}.png"
        path = os.path.join(OUT_DIR, fname)
        render_clean_sphere_png(face_rgb, ctx_seg, path)
        sz_kb = os.path.getsize(path) / 1024
        manifest.append((fname, sz_kb))
        print(f"  wrote {fname}  ({sz_kb:.1f} KB)")


# --------- Manifest ---------
print(f"\nAll renders done in {time.time() - t_start:.1f}s.")
print(f"Output dir: {OUT_DIR}")
print(f"Manifest ({len(manifest)} files):")
for f, kb in manifest:
    print(f"  {f}  ({kb:.1f} KB)")
