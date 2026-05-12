"""Single-pass render: segmentation-colored rank-7 sphere + rank-5 wireframe overlay.
Produces transparent-background PNG matching real_icosphere_rgb_wire.png in style,
for pipeline Figure 1 Stage 5."""
import os, sys, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from trimesh_utils import IcoSphereRef

SAMPLE_IDX = 28
SPLIT = "val"
CKPT = "outputs/rpe_ablation_c4_v2/best_model.pth"
CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs/real_icosphere_seg_wire.png"
TEX_RANK = 7
WIRE_RANK = 5
AZIM, ELEV = 30, 20
FINAL_PX = 1500
SS = 2.4
DPI = 300

S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60], [0.90, 0.75, 0.25], [0.20, 0.60, 0.80], [0.55, 0.35, 0.20],
    [0.85, 0.85, 0.95], [0.95, 0.45, 0.45], [0.75, 0.55, 0.75], [0.40, 0.40, 0.60],
    [0.95, 0.75, 0.55], [0.55, 0.75, 0.45], [0.85, 0.35, 0.60], [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85], [0.30, 0.55, 0.85],
], dtype=np.float32)

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

ref = IcoSphereRef("vertex")
V = np.asarray(ref.get_normals(TEX_RANK), dtype=np.float32)
F = np.asarray(ref.get_icosphere(TEX_RANK, False).faces, dtype=np.int64)
proj_normals = np.asarray(ref.get_normals(mc["img_rank"] - 1), dtype=np.float32)

# Load sample + run EquiSSL inference
ds = Stanford2D3DSeg(
    split=SPLIT, data_dir="${STANFORD2D3D_PATH}",
    img_rank=mc["img_rank"], node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"],
)
sample = ds[SAMPLE_IDX]
rgb_sphere = sample["sphere_rgb"].unsqueeze(0).cuda()
gt_proj = sample["sphere_gt_sem"].numpy()

encoder = SphericalEncoder(
    img_rank=mc["img_rank"], node_type=mc["node_type"],
    embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
    bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
    abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
    rel_pos_bias=True, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True,
)
model = EquiSSLSegUNet(
    encoder=encoder, num_classes=14,
    dec_depths=tuple(mc.get("dec_depths", [2,2,2,2])),
    dec_num_heads=tuple(mc.get("dec_num_heads", [16,16,8,4])),
    d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
    mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
    abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=True,
    rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
    equivariant_rpe=True, n_gauges=4, area_weighted=True,
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda().eval()
with torch.no_grad():
    pred = model(rgb_sphere).argmax(dim=-1).cpu().numpy()[0]

# Upsample prediction from proj rank to render rank 7
tree = cKDTree(proj_normals)
_, nn_idx = tree.query(V, k=1)
pred_render = pred[nn_idx]
gt_render = gt_proj[nn_idx]
# Mask unlabeled regions grey to match the other figures
pred_render[gt_render == 0] = 0

vert_seg_rgb = S2D3D_COLORS[pred_render]
face_seg = vert_seg_rgb[F].mean(axis=1)

# View cull + shade
centroids = V[F].mean(axis=1)
cnorm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
az, el = np.deg2rad(AZIM), np.deg2rad(ELEV)
view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
visible = (cnorm @ view) > 0.01
vf = V[F][visible]
nf = np.cross(vf[:,1]-vf[:,0], vf[:,2]-vf[:,0])
nf /= np.linalg.norm(nf, axis=1, keepdims=True) + 1e-9
flip = (nf * centroids[visible] / np.linalg.norm(centroids[visible], axis=1, keepdims=True)).sum(axis=1) < 0
nf[flip] = -nf[flip]
light = np.array([np.cos(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225))*np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])
shade = 0.55 + 0.45 * np.clip(nf @ light, 0, 1)
shaded = (face_seg[visible] * shade[:, None]).clip(0, 1)

# Wireframe rank-5 (only front-facing edges)
Vw = np.asarray(ref.get_normals(WIRE_RANK), dtype=np.float32)
Fw = np.asarray(ref.get_icosphere(WIRE_RANK, False).faces, dtype=np.int64)
cw = Vw[Fw].mean(axis=1)
cw_u = cw / (np.linalg.norm(cw, axis=1, keepdims=True) + 1e-9)
fw_vis = Fw[(cw_u @ view) > 0.01]
edges = set()
for fc in fw_vis:
    for a, b in [(fc[0], fc[1]), (fc[1], fc[2]), (fc[2], fc[0])]:
        edges.add((a, b) if a < b else (b, a))
seg_pts = Vw[np.array(list(edges))] * 1.001  # avoid z-fight

# Render
size_px = int(FINAL_PX * SS)
fig = plt.figure(figsize=(size_px/DPI, size_px/DPI), dpi=DPI)
ax = fig.add_subplot(111, projection="3d")

coll = Poly3DCollection(vf, facecolors=shaded, edgecolors=shaded,
                        linewidths=0.5, antialiased=False)
ax.add_collection3d(coll)
lc = Line3DCollection(seg_pts, colors=[(0, 0, 0, 0.35)] * len(seg_pts),
                      linewidths=0.35, antialiased=True)
ax.add_collection3d(lc)

ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_axis_off()
fig.patch.set_alpha(0.0); ax.set_facecolor("none")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig.savefig(OUT, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# Post: crop + square + Lanczos
from PIL import Image
im = Image.open(OUT).convert("RGBA")
bbox = im.getbbox()
if bbox is not None:
    w, h = im.size
    mx = int((bbox[2]-bbox[0]) * 0.02); my = int((bbox[3]-bbox[1]) * 0.02)
    bbox = (max(0, bbox[0]-mx), max(0, bbox[1]-my),
            min(w, bbox[2]+mx), min(h, bbox[3]+my))
    im = im.crop(bbox)
    W = max(im.size)
    sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    sq.paste(im, ((W-im.size[0])//2, (W-im.size[1])//2))
    sq.resize((FINAL_PX, FINAL_PX), Image.LANCZOS).save(OUT, optimize=True)
print(f"Saved {OUT}  {os.path.getsize(OUT)//1024} KB")
