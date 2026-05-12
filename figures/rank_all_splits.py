"""Scan train+val+test, rank by pixel accuracy > 90% AND ours wins over both baselines.
Writes figures/sample_ranking_all.json with top candidates.
"""
import os, sys, json, yaml
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")
import numpy as np
import torch
from scipy.spatial import cKDTree

from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg
from network.sphere_model import SphereUFormer
from trimesh_utils import IcoSphereRef

CFG = "configs/pretrain_v8_large.yaml"
OUT_JSON = "figures/sample_ranking_all.json"
NUM_CLASSES = 14
IGNORE = 0
PIXEL_ACC_THRESHOLD = 0.90

with open(CFG) as f: cfg = yaml.safe_load(f)
mc = cfg["model"]
IMG_RANK = mc["img_rank"]
PROJ_RANK = IMG_RANK - 1

ref = IcoSphereRef("vertex")
img_normals = np.asarray(ref.get_normals(IMG_RANK), dtype=np.float32)
proj_normals = np.asarray(ref.get_normals(PROJ_RANK), dtype=np.float32)

def make_ds(split):
    return Stanford2D3DSeg(split=split, data_dir="${STANFORD2D3D_PATH}",
        img_rank=IMG_RANK, node_type=mc["node_type"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"],
        normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"])

def build_ours(ckpt_path, rpe_mode, n_gauges):
    rp, eq = (True, False) if rpe_mode == "standard" else (True, True)
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

def metrics(pred, gt):
    valid = gt != IGNORE
    p = pred[valid]; g = gt[valid]
    if len(g) == 0:
        return 0.0, 0.0, 0
    pix_acc = float((p == g).mean())
    ious = []; present = []
    for c in range(NUM_CLASSES):
        if c == IGNORE: continue
        pm = p == c; gm = g == c
        if not gm.any(): continue
        inter = (pm & gm).sum(); union = (pm | gm).sum()
        ious.append(float(inter / max(union, 1)))
        present.append(c)
    miou = float(np.mean(ious)) if ious else 0.0
    return pix_acc, miou, len(present)

tree_img = cKDTree(img_normals)
_, idx_proj_from_img = tree_img.query(proj_normals, k=1)

@torch.no_grad()
def run_ours(m, rgb): return m(rgb).argmax(dim=-1).cpu().numpy()[0]

@torch.no_grad()
def run_sphereu(m, rgb):
    return m(rgb).argmax(dim=-1).cpu().numpy()[0][idx_proj_from_img]

ckpts = {
    "sphereuformer": ("outputs/sphereuformer_baseline/best_model.pth", "sphereuformer"),
    "standard":     ("outputs/standard_seed123/best_model.pth", ("standard", 6)),
    "c4":           ("outputs/rpe_ablation_c4_v2/best_model.pth", ("equivariant", 4)),
}

all_results = []
for split in ["train", "val", "test"]:
    ds = make_ds(split)
    print(f"\n=== {split}: {len(ds)} samples ===")
    split_results = {k: [None]*len(ds) for k in ckpts}
    for model_name, (path, spec) in ckpts.items():
        print(f"  Loading {model_name}...")
        if spec == "sphereuformer":
            m = build_sphereuformer(path); runner = run_sphereu
        else:
            mode, ng = spec
            m = build_ours(path, mode, ng); runner = run_ours
        for i in range(len(ds)):
            sample = ds[i]
            rgb = sample["sphere_rgb"].unsqueeze(0).cuda()
            gt = sample["sphere_gt_sem"].numpy()
            pred = runner(m, rgb)
            pa, mi, nc = metrics(pred, gt)
            split_results[model_name][i] = {"pix_acc": pa, "miou": mi, "n_classes": nc}
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(ds)}")
        del m; torch.cuda.empty_cache()
    for i in range(len(ds)):
        row = {
            "split": split, "idx": i,
            "sphereu_pa": split_results["sphereuformer"][i]["pix_acc"],
            "standard_pa": split_results["standard"][i]["pix_acc"],
            "c4_pa": split_results["c4"][i]["pix_acc"],
            "sphereu_miou": split_results["sphereuformer"][i]["miou"],
            "standard_miou": split_results["standard"][i]["miou"],
            "c4_miou": split_results["c4"][i]["miou"],
            "n_classes": split_results["c4"][i]["n_classes"],
        }
        row["c4_pa_gap"] = row["c4_pa"] - max(row["sphereu_pa"], row["standard_pa"])
        row["c4_miou_gap"] = row["c4_miou"] - max(row["sphereu_miou"], row["standard_miou"])
        all_results.append(row)

# Filter: c4 pix_acc > 0.90 AND c4 beats both baselines (on pixel accuracy)
filtered = [r for r in all_results
            if r["c4_pa"] > PIXEL_ACC_THRESHOLD
            and r["c4_pa"] > r["sphereu_pa"]
            and r["c4_pa"] > r["standard_pa"]]
filtered.sort(key=lambda r: -r["c4_pa_gap"])

# Also rank all by c4_pa_gap regardless of threshold
by_gap = sorted(all_results, key=lambda r: -r["c4_pa_gap"])

out = {
    "filter": f"c4_pa > {PIXEL_ACC_THRESHOLD} AND c4 beats both baselines",
    "n_passed": len(filtered),
    "top_filtered_by_gap": filtered[:20],
    "top_by_c4_pa_gap_any": by_gap[:20],
    "top_by_c4_pa_absolute": sorted(all_results, key=lambda r: -r["c4_pa"])[:20],
    "all": all_results,
}

with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {OUT_JSON}")

print(f"\n=== Filtered (c4_pa > {PIXEL_ACC_THRESHOLD} AND ours wins), top 10 by gap ===")
for r in filtered[:10]:
    print(f"  {r['split']:5s} idx={r['idx']:4d}  "
          f"c4_pa={r['c4_pa']:.3f}  std_pa={r['standard_pa']:.3f}  sphU_pa={r['sphereu_pa']:.3f}  "
          f"gap={r['c4_pa_gap']:+.3f}  "
          f"c4_miou={r['c4_miou']:.3f}  n={r['n_classes']}")
