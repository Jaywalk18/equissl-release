"""Single-pass render: textured rank-7 sphere + rank-5 wireframe overlay.
Produces transparent-background PNG for pipeline Figure 1 Stage 2."""
import os, sys, yaml, cv2
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from trimesh_utils import IcoSphereRef

SAMPLE_IDX = 28
SPLIT = "val"
CFG = "configs/pretrain_v8_large.yaml"
OUT = "figures/figs/real_icosphere_rgb_wire.png"
TEX_RANK = 7
WIRE_RANK = 5
AZIM, ELEV = 30, 20
FINAL_PX = 1500
SS = 2.4
DPI = 300

with open(CFG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

# Load ERP
ds = Stanford2D3DSeg(
    split=SPLIT,
    data_dir="${STANFORD2D3D_PATH}", img_rank=mc["img_rank"],
    node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"],
)
rgb_path, _ = ds.samples[SAMPLE_IDX]
erp = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
H_, W_ = erp.shape[:2]

# Textured rank-7 sphere
ref = IcoSphereRef("vertex")
V = np.asarray(ref.get_normals(TEX_RANK), dtype=np.float32)
F = np.asarray(ref.get_icosphere(TEX_RANK, False).faces, dtype=np.int64)

from trimesh_utils import asSpherical
sph = asSpherical(V)
phi_deg, theta_deg = sph[:, 1], sph[:, 2]
v_coord = np.clip((phi_deg / 180.0) * (H_ - 1), 0, H_ - 1)
u_coord = np.clip(((theta_deg + 180.0) / 360.0) * (W_ - 1), 0, W_ - 1)
u0 = np.floor(u_coord).astype(np.int32); v0 = np.floor(v_coord).astype(np.int32)
u1 = np.minimum(u0 + 1, W_ - 1); v1 = np.minimum(v0 + 1, H_ - 1)
du = (u_coord - u0)[:, None]; dv = (v_coord - v0)[:, None]
ef = erp.astype(np.float32)
vert_rgb = ((1-du)*(1-dv)*ef[v0,u0] + du*(1-dv)*ef[v0,u1]
            + (1-du)*dv*ef[v1,u0] + du*dv*ef[v1,u1]) / 255.0
face_rgb = vert_rgb[F].mean(axis=1)

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
shaded = (face_rgb[visible] * shade[:, None]).clip(0, 1)

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
seg_pts = Vw[np.array(list(edges))] * 1.001  # push slightly outside to avoid z-fight

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
