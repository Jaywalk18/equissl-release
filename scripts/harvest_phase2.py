#!/usr/bin/env python3
"""
Harvest phase-2 ssl-tbd results into a raw-measurements file for the paper team.

Reads outputs/tbd_seg_*/results.pth and outputs/tbd_depth_*_stage2/results.pth,
emits a JSON at siggraph-asia-2026-gerpe/data/equissl_measurements_<date>.json following
the ssl_tbd_rerun.md naming convention.

NO editorial: every cell's provenance (source dir, mtime, seed) is attached;
paper side owns measured-vs-draft tagging.

Usage:
  python scripts/harvest_phase2.py                 # harvest to default path
  python scripts/harvest_phase2.py --out path.json # custom output
  python scripts/harvest_phase2.py --check-anchors # reproduce seed-42 anchors only
"""

import argparse, json, os, subprocess, sys
from datetime import date
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: torch not available; run with PYTHONPATH set.")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
STAGING = Path("/mnt/ssd/phase2_staging")

INITS = ["rand", "ibot"]
RPES = ["none", "standard", "c4", "c6noarea"]
SEEDS = [42, 123, 456]

# Genuine seed-42 × 1%-label anchors (paper tab:ssl phase-2 cells).
# Only 3 cells verified locally: Random × {None/Standard/C₄} × 1%.
# All 45 other cells (ibot × *, c6noarea × *, seed 123/456, depth × *) are TBD.
ANCHORS = {
    ("seg", "rand", "none", 42):     "tbd_seg_rand_none_s42",      # → label_eff_none_0.01
    ("seg", "rand", "standard", 42): "tbd_seg_rand_standard_s42",  # → label_eff_standard_0.01
    ("seg", "rand", "c4", 42):       "tbd_seg_rand_c4_s42",        # → label_eff_c4_0.01
}


def load_pth(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"__error__": str(e)}


def harvest_one(kind: str, init: str, rpe: str, seed: int):
    """Return dict with {dir, measurements, ...} or None if not found.

    Looks first in outputs/ (promoted ideal) then in staging/ (experimental)."""
    anchor_map = {
        ("seg", "rand", "none", 42):     OUTPUTS / "label_eff_none_0.01",
        ("seg", "rand", "standard", 42): OUTPUTS / "label_eff_standard_0.01",
        ("seg", "rand", "c4", 42):       OUTPUTS / "label_eff_c4_0.01",
    }
    canonical = OUTPUTS / f"phase2_{kind}_{init}_{rpe}_s{seed}"
    anchor = anchor_map.get((kind, init, rpe, seed))
    if kind == "seg":
        stage = (STAGING / f"seg_rand_{rpe}_s{seed}" if init == "rand"
                 else STAGING / f"seg_ibot_{rpe}_s{seed}_stage2")
        metric_keys = ("best_val_miou", "test_miou", "best_epoch",
                       "label_fraction", "freeze")
    else:
        stage = STAGING / f"depth_{init}_{rpe}_s{seed}_stage2"
        metric_keys = ("best_val_delta1", "val_delta1", "best_epoch",
                       "label_fraction", "freeze")

    # precedence: anchor (ideal) > canonical (promoted) > stage (experimental)
    d = None; src_tag = None
    if anchor and anchor.is_dir(): d, src_tag = anchor, "anchor"
    elif canonical.exists(): d, src_tag = canonical, "promoted"
    elif stage.is_dir(): d, src_tag = stage, "staging"
    if d is None:
        return None

    # Resolve symlink if present
    real_dir = d.resolve() if d.is_symlink() else d
    if not real_dir.is_dir():
        return None

    r_path = real_dir / "results.pth"
    try:
        rel_dir = str(real_dir.relative_to(REPO))
    except ValueError:
        rel_dir = str(real_dir)  # outside repo (staging)
    record = {
        "kind": kind, "init": init, "rpe": rpe, "seed": seed,
        "source": src_tag,  # anchor | promoted | staging
        "symlink": str(d.relative_to(REPO)) if d.is_symlink() and str(d).startswith(str(REPO)) else None,
        "dir": rel_dir,
        "exists": r_path.exists(),
        "is_anchor": (kind, init, rpe, seed) in ANCHORS,
    }
    if not r_path.exists():
        return record

    r = load_pth(r_path)
    if "__error__" in r:
        record["error"] = r["__error__"]
        return record

    record["mtime"] = int(r_path.stat().st_mtime)
    record["measurements"] = {}
    for k in metric_keys:
        if k in r:
            v = r[k]
            if isinstance(v, (int, float, bool, str)) or v is None:
                record["measurements"][k] = v

    return record


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: paper repo data/)")
    ap.add_argument("--check-anchors", action="store_true",
                    help="only check seed-42 anchor reproducibility")
    args = ap.parse_args()

    records = []
    for kind in ("seg", "depth"):
        for init in INITS:
            for rpe in RPES:
                for seed in SEEDS:
                    if args.check_anchors and (kind, init, rpe, seed) not in ANCHORS:
                        continue
                    rec = harvest_one(kind, init, rpe, seed)
                    if rec is not None:
                        records.append(rec)

    n_total = len(records)
    n_have = sum(1 for r in records if r.get("exists"))
    n_anchor = sum(1 for r in records if r.get("is_anchor") and r.get("exists"))
    n_phase2 = n_have - n_anchor

    payload = {
        "generated_at": date.today().isoformat(),
        "source_repo": str(REPO),
        "source_commit": git_commit(),
        "note": ("Raw measurements from outputs/. No editorial annotations. "
                 "is_anchor=True means symlinked from pre-phase-2 anchor dir "
                 "(sanity check target); False means genuine phase-2 run."),
        "count_cells_expected": 48,
        "count_cells_present": n_have,
        "count_anchors_present": n_anchor,
        "count_phase2_present": n_phase2,
        "records": records,
    }

    if args.out:
        out = Path(args.out)
    else:
        out = Path("/mnt/ssd/siggraph-asia-2026-gerpe/data") / f"equissl_measurements_{date.today().isoformat()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    print(f"  cells present: {n_have}/48 (anchors={n_anchor}, phase2={n_phase2})")

    # Aggregation preview
    print("\nper-cell availability:")
    for kind in ("seg", "depth"):
        print(f"\n  [{kind}]")
        for init in INITS:
            row = []
            for rpe in RPES:
                cells = [next((r for r in records
                               if r["kind"] == kind and r["init"] == init
                               and r["rpe"] == rpe and r["seed"] == s), None)
                         for s in SEEDS]
                counts = sum(1 for c in cells if c and c.get("exists"))
                row.append(f"{rpe:10}:{counts}/3")
            print(f"    {init:5} " + "  ".join(row))


if __name__ == "__main__":
    main()
