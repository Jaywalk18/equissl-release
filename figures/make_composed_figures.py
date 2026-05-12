"""Compose PTH-rendered icosphere PNGs into paper-ready panel figures.

Output (all academic-style, Liberation Serif = TNR-metric-compatible):
  figures/figs/fig1_rotation_equivariance.png/.pdf   (2x2 panel, for Figure 1 motivation)
  figures/figs/fig_seg_comparison.png/.pdf           (1x4 panel: GT/SphereU/Std/EquiSSL)

Input PNGs (already generated):
  seg_std_upright.png, seg_std_rot90.png  — SphereUFormer baseline (−53% drop)
  seg_ge_upright.png,  seg_ge_rot90.png   — EquiSSL
  (seg_comparison uses the 4 methods rendered from scratch)

All text: Liberation Serif 11/12/14 pt; monochrome labels; clean grid.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "figures/figs"
os.makedirs(OUT, exist_ok=True)

# ---------- Fonts ----------
# Liberation Serif is metric-compatible with Times New Roman
FONT_BOLD = "${HOME_DIR}/.fonts/TimesNewRoman-Bold.ttf"
FONT_REG  = "${HOME_DIR}/.fonts/TimesNewRoman.ttf"
FONT_ITAL = "${HOME_DIR}/.fonts/TimesNewRoman.ttf"

def ft(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

# ---------- S2D3D class colormap (same as renderer) ----------
S2D3D_COLORS = np.array([
    [0.60, 0.60, 0.60],  [0.90, 0.75, 0.25],  [0.20, 0.60, 0.80],
    [0.55, 0.35, 0.20],  [0.85, 0.85, 0.95],  [0.95, 0.45, 0.45],
    [0.75, 0.55, 0.75],  [0.40, 0.40, 0.60],  [0.95, 0.75, 0.55],
    [0.55, 0.75, 0.45],  [0.85, 0.35, 0.60],  [0.70, 0.50, 0.25],
    [0.65, 0.80, 0.85],  [0.30, 0.55, 0.85],
], dtype=np.float32)

CLASSES = ["unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
           "column", "door", "floor", "sofa", "table", "wall", "window"]


def compose_panels(paths, labels, out_path, panel_size=560, gap=16,
                   title_h=44, caption_h=None, caption_text=None):
    """Generic 1×N composer: title above each panel, optional caption below."""
    N = len(paths)
    total_w = panel_size*N + gap*(N-1)
    total_h = title_h + panel_size + (caption_h or 0)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Titles (bold serif)
    f_title = ft(FONT_BOLD, 22)
    for i, (p, label) in enumerate(zip(paths, labels)):
        x = i * (panel_size + gap)
        # Draw title
        draw.text((x + panel_size//2, title_h//2 + 2), label,
                  fill=(20, 20, 20), font=f_title, anchor="mm")
        # Load + resize panel
        im = Image.open(p).convert("RGBA")
        # Auto-crop to content
        bbox = im.getbbox()
        if bbox:
            im = im.crop(bbox)
        # Square + resize
        W = max(im.size)
        sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        sq.paste(im, ((W-im.size[0])//2, (W-im.size[1])//2))
        sq = sq.resize((panel_size, panel_size), Image.LANCZOS)
        # Composite onto canvas (flatten alpha)
        bg = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
        bg.paste(sq, (0, 0), sq)
        canvas.paste(bg, (x, title_h))

    if caption_text and caption_h:
        f_cap = ft(FONT_ITAL, 14)
        draw.text((total_w // 2, title_h + panel_size + caption_h//2), caption_text,
                  fill=(80, 80, 80), font=f_cap, anchor="mm")

    canvas.save(out_path, optimize=True)
    # Also save PDF via reportlab-free route: convert to RGB already, use PIL
    canvas.save(out_path.replace(".png", ".pdf"), "PDF", resolution=300)
    print(f"Saved {out_path}  {canvas.size}")


def compose_2x2(paths, labels, out_path, panel_size=560, gap=18,
                title_h=42, row_label_w=200, legend_h=60, classes_to_show=None):
    """2×2 grid: row = model (Standard vs EquiSSL), col = condition (upright vs rot90).
    Row labels live in a dedicated left margin so they never overlap the panels."""
    cols = 2; rows = 2
    total_w = cols*panel_size + (cols-1)*gap + row_label_w
    total_h = title_h + rows*panel_size + (rows-1)*gap + legend_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    f_title = ft(FONT_BOLD, 22)
    f_row   = ft(FONT_BOLD, 20)
    f_row_sub = ft(FONT_REG, 14)
    f_legend_lbl = ft(FONT_REG, 14)

    # Column titles ("Upright" / "90° rotated")
    col_titles = ["(a)  Upright input", "(b)  90° rotated input"]
    for c in range(cols):
        x = row_label_w + c*(panel_size + gap)
        draw.text((x + panel_size//2, title_h//2 + 2), col_titles[c],
                  fill=(20, 20, 20), font=f_title, anchor="mm")

    # Row labels: main line + subtitle, both right-aligned in the left margin
    row_titles = [("SphereUFormer", "(published)"), ("EquiSSL", "(ours)")]

    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            x = row_label_w + c*(panel_size + gap)
            y = title_h + r*(panel_size + gap)
            im = Image.open(paths[idx]).convert("RGBA")
            bbox = im.getbbox()
            if bbox: im = im.crop(bbox)
            W = max(im.size)
            sq = Image.new("RGBA", (W, W), (0, 0, 0, 0))
            sq.paste(im, ((W-im.size[0])//2, (W-im.size[1])//2))
            sq = sq.resize((panel_size, panel_size), Image.LANCZOS)
            bg = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
            bg.paste(sq, (0, 0), sq)
            canvas.paste(bg, (x, y))

        # Row label in the left margin, vertically centered for the row
        main, sub = row_titles[r]
        label_x = row_label_w - 18
        label_y = title_h + r*(panel_size + gap) + panel_size//2
        draw.text((label_x, label_y - 10), main,
                  fill=(20, 20, 20), font=f_row, anchor="rm")
        draw.text((label_x, label_y + 14), sub,
                  fill=(110, 110, 110), font=f_row_sub, anchor="rm")

    # Legend at bottom
    if classes_to_show:
        legend_y = title_h + rows*panel_size + (rows-1)*gap + 18
        swatch = 18; lx = 14
        for c in classes_to_show:
            col = tuple((S2D3D_COLORS[c]*255).astype(int))
            draw.rectangle([lx, legend_y, lx+swatch, legend_y+swatch],
                           fill=col, outline=(80, 80, 80))
            draw.text((lx+swatch+6, legend_y+swatch//2), CLASSES[c],
                      fill=(30, 30, 30), font=f_legend_lbl, anchor="lm")
            lx += swatch + 8 + draw.textlength(CLASSES[c], font=f_legend_lbl) + 22

    canvas.save(out_path, optimize=True)
    canvas.save(out_path.replace(".png", ".pdf"), "PDF", resolution=300)
    print(f"Saved {out_path}  {canvas.size}")


# ============================================================
# Figure 1: 2×2 rotation equivariance
# rows = model, cols = condition
# ============================================================
classes_in_figure1 = [4, 3, 8, 9, 12, 5, 11, 2, 6]  # ceiling/bookcase/door/floor/wall/chair/table/board/clutter
compose_2x2(
    paths=[
        f"{OUT}/seg_std_upright.png", f"{OUT}/seg_std_rot90.png",   # row 1: SphereUFormer
        f"{OUT}/seg_ge_upright.png",  f"{OUT}/seg_ge_rot90.png",    # row 2: EquiSSL
    ],
    labels=None,
    out_path=f"{OUT}/fig1_rotation_equivariance.png",
    panel_size=580, gap=20, title_h=46, legend_h=62,
    classes_to_show=classes_in_figure1,
)

# ============================================================
# Figure: seg comparison — 4 methods side by side
# ============================================================
# Only run if the 4 panels exist; otherwise user should run make_seg_comparison first.
seg_panels = [
    f"{OUT}/_panel_Ground_truth.png",
    f"{OUT}/_panel_SphereUFormer.png",
    f"{OUT}/_panel_Standard_RPE.png",
    f"{OUT}/_panel_EquiSSL.png",
]
if all(os.path.exists(p) for p in seg_panels):
    compose_panels(
        paths=seg_panels,
        labels=["(a)  Ground truth", "(b)  SphereUFormer",
                "(c)  Standard RPE", "(d)  EquiSSL"],
        out_path=f"{OUT}/fig_seg_comparison.png",
        panel_size=560, gap=20, title_h=44,
    )
else:
    print(f"[skip] seg_comparison: {seg_panels[0]} not found, run make_seg_comparison.py first")

print("\nDone.")
