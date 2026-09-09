#!/usr/bin/env python3
"""Anti-collapse forensic check for pretrain_v9 checkpoint.

Pass criteria (vs collapsed v8 baseline of eff_rank=15, cos=0.997):
  bottleneck eff_rank >= 60   (v8 was 15; random init ~95)
  bottleneck pairwise |cos| <= 0.65  (v8 was 0.997; random ~0.51)

Exit code 0 = PASS (proceed to finetune)
Exit code 1 = FAIL (collapse persisted, do not finetune)
"""
import sys, os, json, argparse
sys.path.insert(0, '${EQUISSL_ROOT}')
sys.path.insert(0, '${SPHERE_UFORMER_SRC}')
import torch
import torch.nn.functional as F
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out_json", default=None)
    p.add_argument("--min_eff_rank", type=float, default=60.0)
    p.add_argument("--max_avg_cos", type=float, default=0.65)
    args = p.parse_args()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEV}")
    from equissl.models.encoder import SphericalEncoder
    from equissl.data.stanford2d3d_seg import Stanford2D3DSeg

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc, dc = cfg["model"], cfg["data"]

    enc = SphericalEncoder(
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        embed_dim=mc["embed_dim"], num_scales=mc["num_scales"],
        in_scale_factor=mc["in_scale_factor"],
        enc_depths=mc["enc_depths"], bottleneck_depth=mc["bottleneck_depth"],
        enc_num_heads=mc["enc_num_heads"],
        d_head_coef=mc["d_head_coef"], win_size_coef=mc["win_size_coef"],
        mlp_ratio=mc["mlp_ratio"], qkv_bias=mc["qkv_bias"],
        drop_path_rate=mc["drop_path_rate"],
        abs_pos_enc_in=mc["abs_pos_enc_in"], abs_pos_enc=mc["abs_pos_enc"],
        rel_pos_bias=mc["rel_pos_bias"], rel_pos_bias_size=mc["rel_pos_bias_size"],
        equivariant_rpe=mc["equivariant_rpe"],
        n_gauges=mc["n_gauges"], area_weighted=mc["area_weighted"],
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    enc_state = {k.replace("student.", "", 1): v for k, v in state.items() if k.startswith("student.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    print(f"Encoder load: {len(enc_state)} src, missing={len(missing)}, unexpected={len(unexpected)}")
    enc.to(DEV).eval()

    ds = Stanford2D3DSeg(
        data_dir="${STANFORD2D3D_PATH}/", split="val",
        img_rank=mc["img_rank"], node_type=mc["node_type"],
        num_scales=mc["num_scales"], in_scale_factor=mc["in_scale_factor"],
        normalize_mean=dc["normalize_mean"], normalize_std=dc["normalize_std"],
    )

    # Average forensic over 5 val samples
    eff_ranks, cos_means = [], []
    with torch.no_grad():
        for i in range(min(5, len(ds))):
            x = ds[i]["sphere_rgb"].unsqueeze(0).to(DEV)
            out = enc(x)
            feats = out["patch"].reshape(-1, out["patch"].shape[-1])
            xc = feats - feats.mean(0)
            S = torch.linalg.svdvals(xc)
            Sn = S / S.sum().clamp(min=1e-6)
            eff_rank = (-(Sn * (Sn + 1e-9).log()).sum()).exp().item()
            xn = F.normalize(feats, dim=-1)
            sim = (xn @ xn.T).abs()
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            cos_mean = sim[mask].mean().item()
            eff_ranks.append(eff_rank); cos_means.append(cos_mean)

    eff_rank_avg = sum(eff_ranks) / len(eff_ranks)
    cos_avg = sum(cos_means) / len(cos_means)

    bottleneck_dim = enc.bottleneck_dim
    print(f"\n=== FORENSIC RESULT ===")
    print(f"  bottleneck dim: {bottleneck_dim}")
    print(f"  eff_rank avg over 5 samples: {eff_rank_avg:.1f} (need >= {args.min_eff_rank})")
    print(f"  pairwise |cos| avg: {cos_avg:.4f} (need <= {args.max_avg_cos})")
    print(f"  v8 baseline (collapsed): eff_rank=15.0, cos=0.997")
    print(f"  random init reference:  eff_rank=94.7, cos=0.513")

    passed = eff_rank_avg >= args.min_eff_rank and cos_avg <= args.max_avg_cos
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    result = dict(
        ckpt=args.ckpt, eff_rank_avg=eff_rank_avg, cos_avg=cos_avg,
        eff_ranks=eff_ranks, cos_means=cos_means,
        min_eff_rank=args.min_eff_rank, max_avg_cos=args.max_avg_cos,
        passed=passed, bottleneck_dim=bottleneck_dim,
    )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote {args.out_json}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
