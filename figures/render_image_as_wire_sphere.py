"""Render ${EQUISSL_ROOT}/image.png as an icosphere with
rank-5 wireframe overlay.

Cross-correlation against val/test/train splits of Stanford2D3D maxed
at 0.37 (train_957 WC_1), well below a "this is the same image"
threshold — image.png is *not* in our dataset, probably a custom
panorama the user sourced separately. We therefore treat it as a
generic ERP and bilinear-sample directly onto the icosphere instead
of looking up a Stanford2D3D `samples[idx]` entry.

Camera azim is bumped 30° → 60° (elev=20° unchanged) to give the
"转30度" view the user asked for. The same wireframe style as
`render_rgb_with_wireframe.py` is preserved (rank-7 textured sphere
+ rank-5 wire overlay, push wires by ×1.001 to avoid z-fighting,
black α=0.35 line color).

Outputs:
  figures/figs/image_icosphere_rgb_wire.png
  figures/ppt_assets/server_delivered/image_icosphere_rgb_wire.png

Run:
    GPU_ID=0 python figures/render_image_as_wire_sphere.py
"""
import os
import sys

GPU_ID = os.environ.get("GPU_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from trimesh_utils import IcoSphereRef, asSpherical

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]


# --------- Config ---------
ERP_PATH    = "${EQUISSL_ROOT}/image.png"
OUT_LOCAL   = "figures/figs/image_icosphere_rgb_wire.png"
OUT_PPT     = "figures/ppt_assets/server_delivered/image_icosphere_rgb_wire.png"

TEX_RANK    = 7
WIRE_RANK   = 5
AZIM        = 60     # 30° rotated from the original render_rgb_with_wireframe.py
ELEV        = 20
FINAL_PX    = 1500
SS          = 2.4    # supersample factor for AA
DPI         = 300


# --------- Load ERP ---------
erp = cv2.imread(ERP_PATH)
if erp is None:
    sys.exit(f"Cannot read {ERP_PATH}")
erp = cv2.cvtColor(erp, cv2.COLOR_BGR2RGB)
print(f"Loaded {ERP_PATH}  shape={erp.shape}")

# Crop any pure-black polar caps (image.png is ~2.31:1, mostly already
# cropped but be safe — bilinear sampling near poles drags black into
# vertices if any black rows remain).
row_mean = erp.mean(axis=(1, 2))
valid = np.where(row_mean > 5)[0]
if len(valid) > 0 and (valid[0] > 0 or valid[-1] < erp.shape[0] - 1):
    top = max(0, valid[0] - 2)
    bot = min(erp.shape[0] - 1, valid[-1] + 2)
    erp = erp[top:bot + 1]
    print(f"  cropped polar caps → {erp.shape}")
H_, W_ = erp.shape[:2]


# --------- Textured rank-7 sphere ---------
ref = IcoSphereRef("vertex")
V = np.asarray(ref.get_normals(TEX_RANK), dtype=np.float32)
F = np.asarray(ref.get_icosphere(TEX_RANK, False).faces, dtype=np.int64)
print(f"Texture rank {TEX_RANK}: {len(V)} verts, {len(F)} faces")

sph = asSpherical(V)
phi_deg, theta_deg = sph[:, 1], sph[:, 2]
v_coord = np.clip((phi_deg / 180.0) * (H_ - 1), 0, H_ - 1)
u_coord = np.clip(((theta_deg + 180.0) / 360.0) * (W_ - 1), 0, W_ - 1)
u0 = np.floor(u_coord).astype(np.int32); v0 = np.floor(v_coord).astype(np.int32)
u1 = np.minimum(u0 + 1, W_ - 1);         v1 = np.minimum(v0 + 1, H_ - 1)
du = (u_coord - u0)[:, None];            dv = (v_coord - v0)[:, None]
ef = erp.astype(np.float32)
vert_rgb = ((1 - du) * (1 - dv) * ef[v0, u0] + du * (1 - dv) * ef[v0, u1]
            + (1 - du) * dv * ef[v1, u0] + du * dv * ef[v1, u1]) / 255.0
face_rgb = vert_rgb[F].mean(axis=1)


# --------- View cull + shade (rotated camera azim=60) ---------
centroids = V[F].mean(axis=1)
cnorm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
az, el = np.deg2rad(AZIM), np.deg2rad(ELEV)
view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
visible = (cnorm @ view) > 0.01
vf = V[F][visible]
nf = np.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
nf /= np.linalg.norm(nf, axis=1, keepdims=True) + 1e-9
c_vis_unit = centroids[visible] / np.linalg.norm(centroids[visible], axis=1, keepdims=True)
flip = (nf * c_vis_unit).sum(axis=1) < 0
nf[flip] = -nf[flip]
light = np.array([np.cos(np.deg2rad(225)) * np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(225)) * np.cos(np.deg2rad(55)),
                  np.sin(np.deg2rad(55))])
shade = 0.55 + 0.45 * np.clip(nf @ light, 0, 1)
shaded = (face_rgb[visible] * shade[:, None]).clip(0, 1)
print(f"  visible faces: {visible.sum()}/{len(F)}")


# --------- Wireframe rank-5 (only front-facing edges) ---------
Vw = np.asarray(ref.get_normals(WIRE_RANK), dtype=np.float32)
Fw = np.asarray(ref.get_icosphere(WIRE_RANK, False).faces, dtype=np.int64)
cw = Vw[Fw].mean(axis=1)
cw_u = cw / (np.linalg.norm(cw, axis=1, keepdims=True) + 1e-9)
fw_vis = Fw[(cw_u @ view) > 0.01]
edges = set()
for fc in fw_vis:
    for a, b in [(int(fc[0]), int(fc[1])),
                 (int(fc[1]), int(fc[2])),
                 (int(fc[2]), int(fc[0]))]:
        edges.add((a, b) if a < b else (b, a))
seg_pts = Vw[np.array(list(edges))] * 1.001  # push slightly outside to avoid z-fight
print(f"  wireframe edges (rank {WIRE_RANK}, front): {len(edges)}")


# --------- Render ---------
size_px = int(FINAL_PX * SS)
fig = plt.figure(figsize=(size_px / DPI, size_px / DPI), dpi=DPI)
ax = fig.add_subplot(111, projection="3d")

ax.add_collection3d(Poly3DCollection(
    vf, facecolors=shaded, edgecolors=shaded,
    linewidths=0.5, antialiased=True))
ax.add_collection3d(Line3DCollection(
    seg_pts, colors=[(0, 0, 0, 0.35)] * len(seg_pts),
    linewidths=0.35, antialiased=True))

ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_axis_off()
fig.patch.set_alpha(0.0); ax.set_facecolor("none")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

os.makedirs(os.path.dirname(OUT_LOCAL), exist_ok=True)
fig.savefig(OUT_LOCAL, dpi=DPI, transparent=True,
            bbox_inches="tight", pad_inches=0)
plt.close(fig)


# --------- Post: tight crop + square pad + LANCZOS downsample ---------
im = Image.open(OUT_LOCAL).convert("RGBA")
bbox = im.getbbox()
if bbox is not None:
    w, h = im.size
    mx = int((bbox[2] - bbox[0]) * 0.02)
    my = int((bbox[3] - bbox[1]) * 0.02)
    bbox = (max(0, bbox[0] - mx), max(0, bbox[1] - my),
            min(w, bbox[2] + mx), min(h, bbox[3] + my))
    im = im.crop(bbox)
    W = max(im.size)
    sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    sq.paste(im, ((W - im.size[0]) // 2, (W - im.size[1]) // 2))
    sq.resize((FINAL_PX, FINAL_PX), Image.LANCZOS).save(
        OUT_LOCAL, optimize=True)

# Mirror to ppt_assets path
os.makedirs(os.path.dirname(OUT_PPT), exist_ok=True)
import shutil
shutil.copy2(OUT_LOCAL, OUT_PPT)

print(f"\nSaved {OUT_LOCAL}  ({os.path.getsize(OUT_LOCAL) // 1024} KB)")
print(f"Mirrored to {OUT_PPT}")
print(f"Camera: azim={AZIM}° elev={ELEV}° (azim rotated +30° from the val_28 baseline)")
