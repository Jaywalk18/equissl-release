"""Extract GE-RPE `bias_grid` parameters from a trained checkpoint.

Per `paper/prompts/server_prompt_mechanism_bias.md`: pull the real
trained 7x7 bilinear bias values out of the canonical EquiSSL
checkpoint so the mechanism animation can visualise the actual learned
function, not a random seed=2026 proxy.

Output: `paper/data/trained_bias.npz` (delivered to Win paper repo).

Per-layer parameter shape: `(1, H, 7, 7)`. We squeeze the singleton
batch dim and permute to `(7, 7, H)` to match the spec's per-layer
convention. Head count varies across the U-Net hierarchy
(3/6/12/24/48), so we NaN-pad to max H and expose `layer_head_counts`
so consumers know where the real data ends.

Run:
    PYTHONPATH=${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC} \
        python figures/extract_trained_bias.py
"""
import argparse
import os
import sys

sys.path.insert(0, "${EQUISSL_ROOT}")

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/rpe_ablation_c4_v2/best_model.pth",
                   help="path to the .pth checkpoint")
    p.add_argument("--cn", type=int, default=4,
                   help="gauge count C_n used at training time")
    p.add_argument("--out", default="figures/data/trained_bias.npz",
                   help="output npz path")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # GaugeEquivariantRPE.bias_grid is the learnable 7x7xH per-head bias
    # (see equissl/models/equivariant_rpe.py:118).
    pairs = []
    for k, v in sd.items():
        if not k.endswith("bias_grid"):
            continue
        if v.ndim != 4 or v.shape[0] != 1 or v.shape[2:] != (7, 7):
            print(f"  skip (unexpected shape) {k}: {tuple(v.shape)}")
            continue
        pairs.append((k, v))

    if not pairs:
        sys.exit("No GE-RPE bias_grid parameters found in checkpoint.")

    L = len(pairs)
    head_counts = np.array([v.shape[1] for _, v in pairs], dtype=np.int32)
    max_H = int(head_counts.max())
    print(f"Found {L} GE-RPE layers, head counts in {sorted(set(head_counts))}, "
          f"max H = {max_H}")

    # (L, 7, 7, max_H) NaN-padded; real heads occupy [..., :H_layer]
    stacked = np.full((L, 7, 7, max_H), np.nan, dtype=np.float32)
    names = []
    for i, (k, v) in enumerate(pairs):
        # (1, H, 7, 7) → (7, 7, H)
        arr = v.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        H = arr.shape[-1]
        stacked[i, :, :, :H] = arr
        names.append(k)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        bias=stacked,
        cn=np.int32(args.cn),
        layer_names=np.array(names, dtype=object),
        layer_head_counts=head_counts,
        head_count=np.int32(max_H),
        checkpoint_path=args.ckpt,
    )
    sz_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out} ({sz_kb:.1f} KB)")
    print(f"  bias               : {stacked.shape}  float32, NaN-padded")
    print(f"  cn                 : {args.cn}")
    print(f"  layer_names        : ({L},) object")
    print(f"  layer_head_counts  : {tuple(head_counts.tolist())}")
    print(f"  head_count         : {max_H}")
    print(f"  checkpoint_path    : {args.ckpt}")
    print()
    print("Sample layer 0:", names[0],
          "real shape (7, 7,", head_counts[0], ")")
    print("Bias value range across all layers:")
    finite = stacked[~np.isnan(stacked)]
    print(f"  min={finite.min():+.3f}  mean={finite.mean():+.3f}  "
          f"max={finite.max():+.3f}  std={finite.std():.3f}")


if __name__ == "__main__":
    main()
