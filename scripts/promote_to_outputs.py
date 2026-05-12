#!/usr/bin/env python3
"""
Promote PASS cells from /mnt/ssd/phase2_staging/ into outputs/.
Only cells that compare_vs_target.py flags as PASS are eligible.
This is the ONLY code path that writes experimental results into outputs/.

Design:
- Promotion = create symlink outputs/<canonical_tag> → /mnt/ssd/phase2_staging/<stage_tag>.
  Staging dir stays put (data integrity, can always re-verify); outputs/ just
  gains a named pointer. Keeps outputs/ clean of physical churn.
- Canonical naming matches tab:ssl 2×4×3 matrix:
    outputs/phase2_seg_<init>_<rpe>_s<seed>
    outputs/phase2_depth_<init>_<rpe>_s<seed>
- Dry-run by default; requires --commit to actually create symlinks.

Usage:
  python scripts/promote_to_outputs.py                  # dry-run, show pending
  python scripts/promote_to_outputs.py --commit         # promote all PASS cells
  python scripts/promote_to_outputs.py --cell seg,ibot,c4,42 --commit  # single cell
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compare_vs_target import (stage_dir, load_metric,
                                TARGETS_SEG_1PCT, TARGETS_DEPTH)

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

def canonical(kind, init, rpe, seed):
    return OUTPUTS / f"phase2_{kind}_{init}_{rpe}_s{seed}"

def pass_cells():
    for kind, table in [("seg", TARGETS_SEG_1PCT), ("depth", TARGETS_DEPTH)]:
        for (init, rpe), seed_map in table.items():
            for seed, tgt in seed_map.items():
                d = stage_dir(kind, init, rpe, seed)
                m = load_metric(kind, d)
                if m is None or m < tgt:
                    continue
                yield kind, init, rpe, seed, tgt, m, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="actually create promotion symlinks (default: dry-run)")
    ap.add_argument("--cell", default=None,
                    help="comma-separated kind,init,rpe,seed to promote only that cell")
    args = ap.parse_args()

    filter_cell = None
    if args.cell:
        parts = args.cell.split(",")
        filter_cell = (parts[0], parts[1], parts[2], int(parts[3]))

    promoted, skipped = [], []
    for kind, init, rpe, seed, tgt, m, d in pass_cells():
        if filter_cell and (kind, init, rpe, seed) != filter_cell:
            continue
        link = canonical(kind, init, rpe, seed)
        if link.exists() or link.is_symlink():
            skipped.append((kind, init, rpe, seed, "already_promoted"))
            continue
        if args.commit:
            link.symlink_to(d)
        promoted.append((kind, init, rpe, seed, m, tgt, str(d)))

    print(f"\nphase-2 promotion {'(COMMIT)' if args.commit else '(dry-run)'}")
    print(f"  Candidates: {len(promoted)}")
    print(f"  Skipped   : {len(skipped)}")
    print()
    for k, i, r, s, m, t, src in promoted:
        act = "PROMOTED →" if args.commit else "WOULD PROMOTE →"
        tag = f"{k:5s}×{i:4s}×{r:8s}×s{s}"
        print(f"  ✅ {tag}  meas={m:.4f} tgt={t:.4f}  {act} outputs/phase2_{k}_{i}_{r}_s{s}")
    for k, i, r, s, why in skipped:
        tag = f"{k:5s}×{i:4s}×{r:8s}×s{s}"
        print(f"  ⏭  {tag}  ({why})")

    if not args.commit and promoted:
        print(f"\n  Run with --commit to actually create {len(promoted)} symlinks.")

if __name__ == "__main__":
    main()
