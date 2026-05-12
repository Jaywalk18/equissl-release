#!/usr/bin/env python3
"""
Split phase-2 48-job manifest into 3 round-robin per-GPU queue files.

Design goals:
- load balance: seg (slow) and depth (fast) interleaved across GPUs
- determinism: same input => same output; no randomness
- resumability: queue files stable across reruns, run_phase2_queue.sh
  auto-skips completed cells by checking results.pth

Usage:
    # 1. Generate the full job manifest
    bash scripts/launch_phase2_ssl.sh all > .ops/all_48_jobs.txt
    # 2. Split into 3 queues
    python scripts/split_phase2_jobs.py .ops/all_48_jobs.txt
    # 3. Kick off 3 tmux sessions (one per GPU)
    for G in 0 1 2; do
      tmux new-session -d -s ph2g$G \
        "GPU=$G bash scripts/run_phase2_queue.sh .ops/q_gpu${G}.sh"
    done
    # 4. Tail logs
    tail -f .ops/q_gpu{0,1,2}.log
"""

import argparse
import sys
from pathlib import Path


def parse_jobs(manifest_path: Path):
    """Parse launch_phase2_ssl.sh output into list of (label, command) pairs."""
    jobs = []
    cur_label = None
    cur_cmd_lines = []
    for raw in manifest_path.read_text().splitlines():
        if raw.startswith("# --- "):
            if cur_cmd_lines:
                jobs.append((cur_label, " ".join(cur_cmd_lines).strip()))
            cur_label = raw.strip("# -").strip()
            cur_cmd_lines = []
        elif raw.strip() == "" or raw.startswith("#"):
            if cur_cmd_lines:
                jobs.append((cur_label, " ".join(cur_cmd_lines).strip()))
                cur_label = None
                cur_cmd_lines = []
        else:
            cur_cmd_lines.append(raw)
    if cur_cmd_lines:
        jobs.append((cur_label, " ".join(cur_cmd_lines).strip()))
    return [(l, c) for (l, c) in jobs if c]


def interleave(jobs):
    """Reorder so seg and depth alternate — load-balances GPU time."""
    seg = [(l, c) for l, c in jobs if "seg" in (l or "")]
    dep = [(l, c) for l, c in jobs if "depth" in (l or "")]
    out = []
    for i in range(max(len(seg), len(dep))):
        if i < len(seg):
            out.append(seg[i])
        if i < len(dep):
            out.append(dep[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path, help="output of launch_phase2_ssl.sh")
    ap.add_argument("--ngpu", type=int, default=3)
    ap.add_argument("--outdir", type=Path, default=Path(".ops"))
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest {args.manifest} not found", file=sys.stderr)
        sys.exit(1)

    jobs = parse_jobs(args.manifest)
    print(f"parsed {len(jobs)} jobs from {args.manifest}")
    jobs = interleave(jobs)
    args.outdir.mkdir(parents=True, exist_ok=True)

    queues = [[] for _ in range(args.ngpu)]
    for i, (label, cmd) in enumerate(jobs):
        queues[i % args.ngpu].append((label, cmd))

    for g, q in enumerate(queues):
        path = args.outdir / f"q_gpu{g}.sh"
        with path.open("w") as f:
            f.write(f"# phase-2 queue for GPU {g} — {len(q)} jobs\n")
            f.write(f"# generated from {args.manifest}\n\n")
            for label, cmd in q:
                f.write(f"# --- {label} ---\n")
                f.write(cmd + "\n\n")
        print(f"wrote {path}  ({len(q)} jobs)")

    total = sum(len(q) for q in queues)
    print(f"\n  Σ = {total} jobs across {args.ngpu} GPUs "
          f"(expect ~{total//args.ngpu}-{total//args.ngpu+1}/GPU)")
    print("\n  Kick off (tmux):")
    for g in range(args.ngpu):
        print(f"    tmux new-session -d -s ph2g{g} "
              f"'GPU={g} bash scripts/run_phase2_queue.sh "
              f"{args.outdir}/q_gpu{g}.sh'")


if __name__ == "__main__":
    main()
