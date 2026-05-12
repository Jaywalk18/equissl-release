"""Dump `.bias_grid` parameters from 4 seed-42 checkpoints into NPZ files
for paper-side theoretical visualization (Proposal A + B).

For each variant, creates data/bias_dumps/<variant>.npz containing:
  - one array per bias_grid tensor, keyed by its full state_dict name,
    shape (1, H_layer, S, S)
  - "_meta_variant":         shape (1,) str
  - "_meta_n_gauges":         shape (1,) int     (1 for Standard RPE)
  - "_meta_grid_size":        shape (1,) int     (=7)
  - "_meta_num_layers":       shape (1,) int
  - "_meta_heads_per_layer":  shape (L,) int
  - "_meta_key_list":         shape (L,) str
  - "_meta_ckpt_path":        shape (1,) str
  - "_meta_best_val_miou":    shape (1,) float

Usage:
  python figures/dump_bias_grids.py
Output: ${WORKSPACE}/siggraph-asia-2026-gerpe/data/bias_dumps/
"""
import os, json
import numpy as np
import torch


VARIANTS = [
    ("standard",  1, "outputs/finetune_v8_random_s/best_model.pth"),
    ("c2",        2, "outputs/rpe_ablation_c2_v2/best_model.pth"),
    ("c4",        4, "outputs/rpe_ablation_c4_v2/best_model.pth"),
    ("c6_noarea", 6, "outputs/rpe_ablation_c6_noarea/best_model.pth"),
]

OUT_DIR = "${WORKSPACE}/siggraph-asia-2026-gerpe/data/bias_dumps"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_bias(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = c["model_state_dict"]
    bias_keys = sorted([k for k in state.keys() if k.endswith(".bias_grid")])
    biases = {k: state[k].detach().cpu().numpy() for k in bias_keys}

    best_val = None
    for k in ("best_metric", "best_val_miou"):
        if k in c:
            v = c[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            best_val = float(v)
            break
    return biases, bias_keys, best_val


summary = []
for name, n, ckpt_path in VARIANTS:
    print(f"\n[{name}]  n_gauges={n}  ckpt={ckpt_path}")
    biases, bias_keys, best_val = extract_bias(ckpt_path)
    heads_per_layer = [b.shape[1] for b in [biases[k] for k in bias_keys]]

    out = dict(biases)
    out["_meta_variant"]         = np.array([name])
    out["_meta_n_gauges"]        = np.array([n], dtype=np.int32)
    out["_meta_grid_size"]       = np.array([biases[bias_keys[0]].shape[-1]],
                                            dtype=np.int32)
    out["_meta_num_layers"]      = np.array([len(bias_keys)], dtype=np.int32)
    out["_meta_heads_per_layer"] = np.array(heads_per_layer, dtype=np.int32)
    out["_meta_key_list"]        = np.array(bias_keys)
    out["_meta_ckpt_path"]       = np.array([ckpt_path])
    out["_meta_best_val_miou"]   = np.array([best_val if best_val is not None else np.nan],
                                            dtype=np.float32)

    npz_path = f"{OUT_DIR}/{name}.npz"
    np.savez(npz_path, **out)
    sz_mb = os.path.getsize(npz_path) / 1e6
    print(f"  -> {npz_path}  ({sz_mb:.2f} MB, {len(bias_keys)} layers, "
          f"H per layer = {heads_per_layer})")
    if best_val is not None:
        print(f"     best_val_miou = {best_val:.4f}")

    summary.append({
        "variant": name,
        "n_gauges": n,
        "num_layers": len(bias_keys),
        "heads_per_layer": heads_per_layer,
        "grid_size": int(biases[bias_keys[0]].shape[-1]),
        "best_val_miou": best_val,
        "ckpt": ckpt_path,
    })

with open(f"{OUT_DIR}/_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary → {OUT_DIR}/_summary.json")
print("\nAll dumps complete.")
