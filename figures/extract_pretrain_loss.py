"""Parse iBOT+MAE per-step pretraining log → per-epoch mean CSV.

Per `paper/prompts/server_prompt_loss_curves.md`: review R9 W3 wants a
loss-curve plot in App.I.1 to support the textual claim that iBOT
patch-distillation cross-entropy plateaus past ep 60 and MAE
reconstruction loss flattens by ep 70.

The "canonical iBOT+MAE run" that the prompt references (val 68.30
after finetune) does not exist as an actually-trained checkpoint —
68.30 is a paper-derived target. The
best ACTUALLY-TRAINED iBOT+MAE run is v9 KoLeo (downstream val 61.42).
That's what we ship here, with an explicit canonical-mismatch note in
the delivery.

Source log: .ops/pretrain_v9.log (or v8 / v10 via --log).
Step-line format (per-rank-0 only logged):
    [STEP/TOTAL] loss=X cls=X patch=X recon=X [koleo=X] lr=X mom=X
Epoch boundary marker:
    Epoch N/99
v10 has no `recon` field (no-MAE rescue variant). v8 and v9 do.

Run:
    python figures/extract_pretrain_loss.py
"""
import argparse
import os
import re
import csv
import sys


STEP_RE = re.compile(
    r"\[\s*(\d+)/\s*(\d+)\]\s+loss=[\d.]+\s+cls=[\d.]+\s+"
    r"patch=([\d.]+)\s+recon=([\d.]+)"
)
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/\d+\s*$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=".ops/pretrain_v9.log",
                   help="path to pretraining stdout log")
    p.add_argument("--out-csv", default="figures/data/pretrain_loss_curves.csv")
    p.add_argument("--run-name", default="v9_koleo",
                   help="short name for the audit trail")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.log):
        sys.exit(f"Log not found: {args.log}")

    # epoch -> list of (patch, recon) for step lines under that epoch
    by_epoch = {}
    cur_epoch = None
    with open(args.log) as f:
        for line in f:
            line = line.rstrip("\n")
            m = EPOCH_RE.match(line)
            if m:
                cur_epoch = int(m.group(1))
                by_epoch.setdefault(cur_epoch, [])
                continue
            if cur_epoch is None:
                continue
            m = STEP_RE.search(line)
            if m:
                patch = float(m.group(3))
                recon = float(m.group(4))
                by_epoch[cur_epoch].append((patch, recon))

    if not by_epoch:
        sys.exit("No step lines parsed — log format mismatch?")

    rows = []
    for ep in sorted(by_epoch):
        steps = by_epoch[ep]
        if not steps:
            continue
        patch_mean = sum(p for p, _ in steps) / len(steps)
        recon_mean = sum(r for _, r in steps) / len(steps)
        rows.append({
            "epoch": ep,
            "ibot_loss": round(patch_mean, 6),
            "mae_loss": round(recon_mean, 6),
            "n_steps_logged": len(steps),
        })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "ibot_loss",
                                          "mae_loss", "n_steps_logged"])
        w.writeheader()
        w.writerows(rows)

    print(f"Parsed {len(rows)} epochs from {args.log}")
    print(f"  iBOT (patch) range: {min(r['ibot_loss'] for r in rows):.4f} → "
          f"{max(r['ibot_loss'] for r in rows):.4f}")
    print(f"  MAE  (recon) range: {min(r['mae_loss'] for r in rows):.4f} → "
          f"{max(r['mae_loss'] for r in rows):.4f}")
    print(f"  Final epoch ({rows[-1]['epoch']}): iBOT={rows[-1]['ibot_loss']:.4f} "
          f"MAE={rows[-1]['mae_loss']:.4f}")
    print(f"\nWrote {args.out_csv}")

    # Quick verification of the App.I.1 textual claims:
    # iBOT: plateau past ep 60 (Δ < 1% of total descent in epochs 60→100)
    # MAE: flattens by ep 70
    final_ibot = rows[-1]["ibot_loss"]
    init_ibot = rows[0]["ibot_loss"]
    total_descent_ibot = init_ibot - final_ibot
    ep60_ibot = next((r["ibot_loss"] for r in rows if r["epoch"] == 60), None)
    if ep60_ibot is not None:
        delta_60_to_end = ep60_ibot - final_ibot
        pct_of_descent = 100 * delta_60_to_end / max(total_descent_ibot, 1e-9)
        print(f"\nApp.I.1 claim verification:")
        print(f"  iBOT: ep0={init_ibot:.4f} → ep60={ep60_ibot:.4f} → "
              f"ep{rows[-1]['epoch']}={final_ibot:.4f}")
        print(f"        descent ep60→end = {delta_60_to_end:+.4f} "
              f"({pct_of_descent:.2f}% of total descent)")
    ep70_mae = next((r["mae_loss"] for r in rows if r["epoch"] == 70), None)
    if ep70_mae is not None:
        final_mae = rows[-1]["mae_loss"]
        total_descent_mae = rows[0]["mae_loss"] - final_mae
        delta_mae = ep70_mae - final_mae
        pct = 100 * delta_mae / max(total_descent_mae, 1e-9)
        print(f"  MAE:  ep0={rows[0]['mae_loss']:.4f} → ep70={ep70_mae:.4f} → "
              f"ep{rows[-1]['epoch']}={final_mae:.4f}")
        print(f"        descent ep70→end = {delta_mae:+.4f} "
              f"({pct:.2f}% of total descent)")


if __name__ == "__main__":
    main()
