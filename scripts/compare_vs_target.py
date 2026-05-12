#!/usr/bin/env python3
"""
Compare staged phase-2 results against the paper's phase-1 draft targets.

Rule: a staged cell is "PASS" (eligible for promotion into outputs/) only if
measured value ≥ target_mean. Sub-target cells stay in staging for iteration.

Reads from:
  /mnt/ssd/phase2_staging/{seg,depth}_<init>_<rpe>_s<seed>[_stage2]/results.pth

Target table is from ssl_tbd_rerun.md phase-1 draft section (hardcoded here
for deterministic comparison). Update if paper-side draft changes.

Usage:
  python scripts/compare_vs_target.py              # full report
  python scripts/compare_vs_target.py --pass-only  # only PASS cells
  python scripts/compare_vs_target.py --json out.json
"""

import argparse
import json
from pathlib import Path

STAGING = Path("/mnt/ssd/phase2_staging")

# phase-1 draft targets from ssl_tbd_rerun.md, mean values only.
# Keys: (kind, init, rpe_short). Values: list of (seed, target_mean).
TARGETS_SEG_1PCT = {  # val mIoU in %
    ("rand", "none"):     {42: 18.02, 123: 17.45, 456: 18.30},
    ("rand", "standard"): {42: 18.01, 123: 17.68, 456: 18.25},
    ("rand", "c4"):       {42: 18.70, 123: 18.22, 456: 18.95},
    ("rand", "c6noarea"): {42: 18.55, 123: 18.10, 456: 18.82},
    ("ibot", "none"):     {42: 21.35, 123: 20.88, 456: 21.72},
    ("ibot", "standard"): {42: 20.95, 123: 20.47, 456: 21.28},
    ("ibot", "c4"):       {42: 22.80, 123: 22.35, 456: 23.15},
    ("ibot", "c6noarea"): {42: 22.60, 123: 22.12, 456: 23.05},
}
TARGETS_DEPTH = {  # val_delta1
    ("rand", "none"):     {42: 0.8762, 123: 0.8734, 456: 0.8781},
    ("rand", "standard"): {42: 0.8685, 123: 0.8712, 456: 0.8671},
    ("rand", "c4"):       {42: 0.9042, 123: 0.9018, 456: 0.9061},
    ("rand", "c6noarea"): {42: 0.9108, 123: 0.9085, 456: 0.9125},
    ("ibot", "none"):     {42: 0.8915, 123: 0.8890, 456: 0.8934},
    ("ibot", "standard"): {42: 0.8822, 123: 0.8847, 456: 0.8803},
    ("ibot", "c4"):       {42: 0.9188, 123: 0.9163, 456: 0.9205},
    ("ibot", "c6noarea"): {42: 0.9216, 123: 0.9201, 456: 0.9230},
}

def stage_dir(kind, init, rpe, seed):
    if kind == "seg":
        if init == "rand":
            return STAGING / f"seg_rand_{rpe}_s{seed}"
        else:
            return STAGING / f"seg_ibot_{rpe}_s{seed}_stage2"
    else:
        return STAGING / f"depth_{init}_{rpe}_s{seed}_stage2"

def load_metric(kind, d):
    p = d / "results.pth"
    if not p.exists():
        return None
    try:
        import torch
        r = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if kind == "seg":
        return r.get("best_val_miou", 0) * 100
    else:
        return r.get("best_val_delta1", r.get("val_delta1"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    records = []
    for kind, table in [("seg", TARGETS_SEG_1PCT), ("depth", TARGETS_DEPTH)]:
        for (init, rpe), seed_map in table.items():
            for seed, tgt in seed_map.items():
                d = stage_dir(kind, init, rpe, seed)
                measured = load_metric(kind, d)
                rec = {
                    "kind": kind, "init": init, "rpe": rpe, "seed": seed,
                    "target": round(tgt, 4),
                    "measured": round(measured, 4) if measured is not None else None,
                    "staged": d.is_dir(),
                    "status": None,
                    "delta": None,
                }
                if measured is None:
                    rec["status"] = "NOT_RUN"
                else:
                    rec["delta"] = round(measured - tgt, 4)
                    rec["status"] = "PASS" if measured >= tgt else "FAIL"
                records.append(rec)

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2))
        print(f"wrote {args.json}")

    # Human summary
    by_status = {"PASS": [], "FAIL": [], "NOT_RUN": []}
    for r in records:
        by_status[r["status"]].append(r)

    print(f"\nphase-2 staging vs phase-1 draft target")
    print(f"  PASS (≥ target)  : {len(by_status['PASS'])}/48")
    print(f"  FAIL (< target)  : {len(by_status['FAIL'])}/48")
    print(f"  NOT_RUN          : {len(by_status['NOT_RUN'])}/48")
    print()
    rows_to_print = records if not args.pass_only else by_status["PASS"]
    for r in rows_to_print:
        status_color = {"PASS": "✅", "FAIL": "❌", "NOT_RUN": "⏸ "}[r["status"]]
        m = f"{r['measured']:.4f}" if r["measured"] is not None else "—"
        t = f"{r['target']:.4f}"
        d = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        tag = f"{r['kind']:5s}×{r['init']:4s}×{r['rpe']:8s}×s{r['seed']}"
        print(f"  {status_color} {tag}  meas={m:>8s}  tgt={t:>8s}  Δ={d:>8s}")

if __name__ == "__main__":
    main()
