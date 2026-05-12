"""Animated rotation curve: red dot slides 0->90deg along x-axis,
two methods (EquiSSL ours C4 GE-RPE, SphereUFormer baseline) light up
their current-angle mIoU as the dot slides.

Source: outputs/rotation_curve/<prefix>angle<deg>_val.log + outputs/rotation_curve_sphereuformer/.
Output: figures/figs/rotation_curve_anim.mp4
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs" / "rotation_curve_anim.mp4"

ANGLES = np.array([0, 10, 20, 35, 45, 60, 90], dtype=float)
EQUISSL = np.array([0.6599, 0.6567, 0.6569, 0.6551, 0.6561, 0.6557, 0.6552])
SPHEREU = np.array([0.6298, 0.5922, 0.5377, 0.4406, 0.4097, 0.3595, 0.2961])

DENSE = np.linspace(0.0, 90.0, 181)
EQUISSL_D = np.interp(DENSE, ANGLES, EQUISSL)
SPHEREU_D = np.interp(DENSE, ANGLES, SPHEREU)


def main():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor("white")

    ax.plot(DENSE, EQUISSL_D * 100, color="#1f77b4", lw=2.4, label="EquiSSL (Ours)", zorder=3)
    ax.plot(DENSE, SPHEREU_D * 100, color="#d62728", lw=2.4, label="SphereUFormer (CVPR'25)", zorder=3)
    ax.scatter(ANGLES, EQUISSL * 100, s=42, color="#1f77b4", edgecolor="white", lw=1.2, zorder=4)
    ax.scatter(ANGLES, SPHEREU * 100, s=42, color="#d62728", edgecolor="white", lw=1.2, zorder=4)

    eq_line = ax.scatter([], [], s=180, color="#1f77b4", edgecolor="black", lw=1.6, zorder=6)
    su_line = ax.scatter([], [], s=180, color="#d62728", edgecolor="black", lw=1.6, zorder=6)
    cursor = ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.55, zorder=2)

    eq_text = ax.text(0, 0, "", fontsize=11, color="#1f77b4", weight="bold", ha="left", va="bottom", zorder=7)
    su_text = ax.text(0, 0, "", fontsize=11, color="#d62728", weight="bold", ha="left", va="top", zorder=7)
    angle_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=14, weight="bold",
                         va="top", ha="left",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6", edgecolor="#888"))

    ax.set_xlim(-3, 93)
    ax.set_ylim(25, 70)
    ax.set_xlabel("Rotation angle (deg)", fontsize=13)
    ax.set_ylabel("mIoU (%)", fontsize=13)
    ax.set_title("Rotation robustness on Stanford2D3D (val)", fontsize=14, weight="bold")
    ax.set_xticks([0, 10, 20, 35, 45, 60, 90])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.95)

    n_frames = len(DENSE)

    def update(i):
        a = DENSE[i]
        eq = EQUISSL_D[i] * 100
        su = SPHEREU_D[i] * 100
        cursor.set_xdata([a, a])
        eq_line.set_offsets([[a, eq]])
        su_line.set_offsets([[a, su]])
        eq_text.set_position((a + 1.5, eq + 0.4))
        eq_text.set_text(f"{eq:.2f}")
        su_text.set_position((a + 1.5, su - 0.4))
        su_text.set_text(f"{su:.2f}")
        angle_text.set_text(f"angle = {a:5.1f} deg")
        return cursor, eq_line, su_line, eq_text, su_text, angle_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=24, bitrate=2400, codec="libx264")
    anim.save(str(OUT), writer=writer)
    print(f"saved {OUT}  ({n_frames} frames)")


if __name__ == "__main__":
    main()
