"""Table 6 runtime/memory benchmark — single-sample forward + build time.

Outputs JSON at figures/runtime_benchmark.json with per-variant:
  build_s, forward_ms, peak_mem_mb, params_M
"""
import os, sys, yaml, time, json
sys.path.insert(0, "${EQUISSL_ROOT}")
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

import numpy as np
import torch
from equissl.models.encoder import SphericalEncoder, EquiSSLSegUNet
from equissl.data.stanford2d3d_seg import Stanford2D3DSeg

CFG = "configs/pretrain_v8_large.yaml"
OUT_JSON = os.environ.get("RT_BENCH_OUT", "figures/runtime_benchmark.json")
BATCH = int(os.environ.get("RT_BENCH_BATCH", "4"))
WARMUP, MEASURE = int(os.environ.get("RT_BENCH_WARMUP", "10")), int(os.environ.get("RT_BENCH_ITERS", "50"))

with open(CFG) as f: cfg = yaml.safe_load(f)
mc = cfg["model"]

ds = Stanford2D3DSeg(split="val", data_dir="${STANFORD2D3D_PATH}",
    img_rank=mc["img_rank"], node_type=mc["node_type"], num_scales=mc["num_scales"],
    in_scale_factor=mc["in_scale_factor"],
    normalize_mean=cfg["data"]["normalize_mean"], normalize_std=cfg["data"]["normalize_std"])
rgb = torch.stack([ds[i]["sphere_rgb"] for i in range(BATCH)]).cuda()
print(f"Input shape: {rgb.shape}")


def build(rpe_mode, n_gauges, area_weighted=True):
    rp, eq = (False, False) if rpe_mode == "none" else (
        (True, False) if rpe_mode == "standard" else (True, True))
    enc = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"], enc_depths=mc["enc_depths"],
        bottleneck_depth=mc["bottleneck_depth"], enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"], drop_path_rate=0.0,
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=rp, rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=area_weighted)
    m = EquiSSLSegUNet(encoder=enc, num_classes=14,
        dec_depths=tuple(mc.get("dec_depths", [2,2,2,2])),
        dec_num_heads=tuple(mc.get("dec_num_heads", [16,16,8,4])),
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        abs_pos_enc=mc["abs_pos_enc"], rel_pos_bias=rp,
        rel_pos_bias_size=mc.get("rel_pos_bias_size", 7),
        equivariant_rpe=eq, n_gauges=n_gauges, area_weighted=area_weighted)
    return m


# Six variants collected in one GPU-hot run (Table 6 fresh).
# aw = area_weighted flag passed to the RPE module.
variants = [
    ("None",                           "none",        0, True),
    ("Standard RPE",                   "standard",    0, True),
    ("EquiSSL-C_2",                    "equivariant", 2, True),
    ("EquiSSL-C_4",                    "equivariant", 4, True),
    ("EquiSSL-C_6",                    "equivariant", 6, True),
    ("EquiSSL-C_6 no-area (canonical)","equivariant", 6, False),
]

results = []
for label, rpe, ng, aw in variants:
    print(f"\n=== {label} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    m = build(rpe, ng, aw).cuda().eval()
    build_s = time.time() - t0
    params_M = sum(p.numel() for p in m.parameters()) / 1e6

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = m(rgb)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(MEASURE):
            _ = m(rgb)
        torch.cuda.synchronize()
        forward_ms = (time.time() - t0) / MEASURE * 1000

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    entry = {
        "variant": label, "rpe_mode": rpe, "n_gauges": ng,
        "area_weighted": aw,
        "params_M": round(params_M, 1),
        "build_s": round(build_s, 1),
        "forward_ms": round(forward_ms, 1),
        "peak_mem_mb": round(peak_mem_mb, 0),
    }
    results.append(entry)
    print(f"  params={params_M:.1f}M  build={build_s:.1f}s  "
          f"forward={forward_ms:.1f}ms  mem={peak_mem_mb:.0f}MB")

    del m; torch.cuda.empty_cache()

with open(OUT_JSON, "w") as f:
    json.dump({"batch": BATCH, "img_rank": mc["img_rank"], "results": results}, f, indent=2)
print(f"\nSaved {OUT_JSON}")

# Print Markdown table for easy copy-paste
print("\n=== Markdown table ===")
print("| Variant      | Params | Build (s) | Forward (ms) | Peak mem (MB) |")
print("|--------------|-------:|----------:|-------------:|--------------:|")
for r in results:
    print(f"| {r['variant']:12s} | {r['params_M']:.1f} M | {r['build_s']:.1f} | "
          f"{r['forward_ms']:.1f} | {int(r['peak_mem_mb']):<4d} |")
