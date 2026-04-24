"""
generate_figure1.py
Run:  python generate_figure1.py
Outputs: figure1_architecture.png  (300 DPI, ~15 cm wide — ready for Word / LaTeX)
         figure1_architecture.pdf  (vector, perfect for LaTeX \\includegraphics)
Requires: matplotlib  (pip install matplotlib)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.5)
ax.axis("off")
fig.patch.set_facecolor("#FAFBFF")
ax.set_facecolor("#FAFBFF")

# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────
C = {
    "dicom":  "#2563EB",
    "seg":    "#0891B2",
    "risk":   "#DC2626",
    "heat":   "#D97706",
    "morph":  "#059669",
    "report": "#7C3AED",
    "bg":     "#F0F4FF",
    "line":   "#64748B",
    "text":   "#0F172A",
    "sub":    "#64748B",
    "white":  "#FFFFFF",
}

# ─────────────────────────────────────────────
# Helper: draw a card box
# ─────────────────────────────────────────────
def card(ax, x, y, w, h, color, title, subtitle1="", subtitle2="", badge=""):
    """Draw a rounded card with a top accent bar."""
    shadow = FancyBboxPatch((x + 0.06, y - 0.06), w, h,
                            boxstyle="round,pad=0.04",
                            linewidth=0, facecolor="#00000018", zorder=2)
    ax.add_patch(shadow)

    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04",
                         linewidth=1.6, edgecolor=color,
                         facecolor=color + "22", zorder=3)
    ax.add_patch(box)

    # Top accent bar
    bar = FancyBboxPatch((x, y + h - 0.18), w, 0.18,
                         boxstyle="round,pad=0.02",
                         linewidth=0, facecolor=color, zorder=4,
                         clip_on=False)
    ax.add_patch(bar)

    # Badge circle
    if badge:
        circ = plt.Circle((x + 0.32, y + h - 0.09), 0.13,
                           color=color, zorder=5)
        ax.add_patch(circ)
        ax.text(x + 0.32, y + h - 0.09, badge,
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=6)

    # Text
    ax.text(x + w / 2, y + h * 0.56, title,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=color, zorder=5)
    if subtitle1:
        ax.text(x + w / 2, y + h * 0.32, subtitle1,
                ha="center", va="center", fontsize=7.5,
                color=C["sub"], zorder=5)
    if subtitle2:
        ax.text(x + w / 2, y + h * 0.14, subtitle2,
                ha="center", va="center", fontsize=6.8,
                color=C["sub"], style="italic", zorder=5)


# ─────────────────────────────────────────────
# Helper: labelled arrow
# ─────────────────────────────────────────────
def arrow(ax, x0, y0, x1, y1, label="", color="#64748B", dashed=False):
    ls = (0, (4, 3)) if dashed else "-"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, linestyle=ls),
                zorder=4)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.17
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="#F1F5F9",
                          ec="#CBD5E1", lw=0.8))


# ─────────────────────────────────────────────
# STAGE 1 — DICOM Input
# ─────────────────────────────────────────────
card(ax, 0.3, 2.8, 2.0, 2.0,
     C["dicom"], "DICOM Input",
     "MRI / CT Scan Stack",
     "Raw acquisition data", badge="1")

# ─────────────────────────────────────────────
# ARROW 1 → 2
# ─────────────────────────────────────────────
arrow(ax, 2.30, 3.80, 2.75, 3.80, "NIfTI / OBJ", C["dicom"])

# ─────────────────────────────────────────────
# STAGE 2 — 3-D Segmentation
# ─────────────────────────────────────────────
card(ax, 2.75, 2.8, 2.2, 2.0,
     C["seg"], "3-D Segmentation",
     "Vessel mesh extraction",
     "Point-cloud generation", badge="2")

# ─────────────────────────────────────────────
# Trunk then vertical spine to 3 modules
# ─────────────────────────────────────────────
trunk_x = 5.65
spine_x = 6.05
mod_x   = 6.30

# Horizontal trunk
ax.annotate("", xy=(trunk_x, 3.80), xytext=(4.95, 3.80),
            arrowprops=dict(arrowstyle="-", color=C["line"], lw=1.8), zorder=4)
ax.text((4.95 + trunk_x) / 2, 3.97, "Point Cloud",
        ha="center", va="bottom", fontsize=7, color=C["line"],
        bbox=dict(boxstyle="round,pad=0.15", fc="#F1F5F9", ec="#CBD5E1", lw=0.8))

# Vertical spine
spine_y_top = 6.20
spine_y_bot = 1.40
ax.plot([spine_x, spine_x], [spine_y_bot, spine_y_top],
        color=C["line"], lw=1.8, zorder=3, ls="--", alpha=0.5)

# Dot at trunk junction
ax.plot(spine_x, 3.80, "o", color=C["line"], ms=5, zorder=5)

# "PARALLEL MODULES" badge
bx, by = spine_x - 0.75, 6.50
ax.text(spine_x, by + 0.25, "PARALLEL MODULES",
        ha="center", va="center", fontsize=7.5,
        fontweight="bold", color=C["dicom"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#EFF6FF",
                  ec=C["dicom"], lw=1.2))
ax.plot([spine_x, spine_x], [6.41, spine_y_top],
        color=C["dicom"], lw=1, ls=":", alpha=0.6, zorder=3)

# Branch arrows → modules
arrow(ax, spine_x, 5.60, mod_x, 5.60, color=C["risk"])
arrow(ax, spine_x, 3.80, mod_x, 3.80, color=C["heat"])
arrow(ax, spine_x, 2.00, mod_x, 2.00, color=C["morph"])

# ─────────────────────────────────────────────
# MODULE A — Risk Predictor
# ─────────────────────────────────────────────
card(ax, mod_x, 4.85, 3.2, 1.60,
     C["risk"], "Risk Predictor",
     "Graph Neural Network · BCE Loss · Calibration",
     "Output: Rupture Probability Score", badge="A")

# ─────────────────────────────────────────────
# MODULE B — Saliency Heatmap
# ─────────────────────────────────────────────
card(ax, mod_x, 3.05, 3.2, 1.60,
     C["heat"], "Saliency Heatmap",
     "Grad-CAM · XAI · 3-D Overlay",
     "Output: Visual Attention Map on Vessel", badge="B")

# ─────────────────────────────────────────────
# MODULE C — Morphology Analyzer
# ─────────────────────────────────────────────
card(ax, mod_x, 1.25, 3.2, 1.60,
     C["morph"], "Morphology Analyzer",
     "Curvature · Volume · Neck Ratio · Aspect Ratio",
     "Output: Geometric Feature Vector", badge="C")

# ─────────────────────────────────────────────
# Right-side merge spine
# ─────────────────────────────────────────────
merge_x   = 9.85
report_x  = 10.30
merge_top = 5.65
merge_bot = 2.05

ax.plot([merge_x, merge_x], [merge_bot, merge_top],
        color=C["line"], lw=1.8, zorder=3, alpha=0.5)

# Horizontal lines from modules
ax.plot([mod_x + 3.2, merge_x], [5.65, 5.65], color=C["risk"],  lw=1.4, alpha=0.7)
ax.plot([mod_x + 3.2, merge_x], [3.85, 3.85], color=C["heat"],  lw=1.4, alpha=0.7)
ax.plot([mod_x + 3.2, merge_x], [2.05, 2.05], color=C["morph"], lw=1.4, alpha=0.7)

ax.plot(merge_x, 5.65, "o", color=C["risk"],  ms=5, zorder=5)
ax.plot(merge_x, 3.85, "o", color=C["heat"],  ms=5, zorder=5)
ax.plot(merge_x, 2.05, "o", color=C["morph"], ms=5, zorder=5)

# Final arrow to report
arrow(ax, merge_x, 3.85, report_x, 3.85, "Fusion", C["report"])

# ─────────────────────────────────────────────
# OUTPUT — Clinical Report
# ─────────────────────────────────────────────
card(ax, report_x, 2.10, 4.30, 3.50,
     C["report"], "Clinical Report",
     "", "", badge="4")

# Sub-bullet points inside report box
bullets = [
    (C["risk"],   "Risk Score & Rupture Probability"),
    (C["heat"],   "Saliency Map (3-D Visualisation)"),
    (C["morph"],  "Geometric Measurements"),
    (C["report"], "NLP Clinical Explanation (PDF)"),
]
for i, (col, txt) in enumerate(bullets):
    bx = report_x + 0.35
    by = 4.60 - i * 0.50
    circ = plt.Circle((bx, by), 0.07, color=col, zorder=6)
    ax.add_patch(circ)
    ax.text(bx + 0.20, by, txt, va="center", fontsize=8,
            color=C["text"], zorder=6)

ax.text(report_x + 4.30 / 2, 5.20, "Clinical Report",
        ha="center", va="center", fontsize=12,
        fontweight="bold", color=C["report"], zorder=6)

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
ax.text(7.5, 7.28,
        "AneuXplain — Three-Module Processing Architecture",
        ha="center", va="top", fontsize=14,
        fontweight="bold", color=C["text"],
        path_effects=[pe.withStroke(linewidth=3, foreground="#FAFBFF")])

ax.plot([1.5, 13.5], [7.07, 7.07], color="#CBD5E1", lw=1.0)

# ─────────────────────────────────────────────
# Stage step labels (above boxes)
# ─────────────────────────────────────────────
for x, label in [(1.30, "① Input"), (3.85, "② Preprocess"),
                 (7.90, "③ Analysis"), (12.45, "④ Output")]:
    ax.text(x, 5.05 if label.startswith("③") else 4.99,
            label, ha="center", va="bottom", fontsize=8,
            color=C["sub"], fontstyle="italic")

# ─────────────────────────────────────────────
# Legend + caption bar
# ─────────────────────────────────────────────
legend_y = 0.60
legend_items = [
    (C["dicom"],  "DICOM Input"),
    (C["seg"],    "Segmentation"),
    (C["risk"],   "Risk Predictor"),
    (C["heat"],   "Saliency Heatmap"),
    (C["morph"],  "Morphology Analyzer"),
    (C["report"], "Clinical Report"),
]

# Legend background
legend_bg = FancyBboxPatch((0.25, 0.25), 14.5, 0.70,
                            boxstyle="round,pad=0.05",
                            linewidth=1, edgecolor="#E2E8F0",
                            facecolor="#F8FAFC", zorder=2)
ax.add_patch(legend_bg)

spacing = 14.5 / len(legend_items)
for i, (col, label) in enumerate(legend_items):
    lx = 0.60 + i * spacing + spacing / 2
    circ = plt.Circle((lx - 0.25, legend_y), 0.10, color=col, zorder=4)
    ax.add_patch(circ)
    ax.text(lx - 0.10, legend_y, label, va="center",
            fontsize=7.8, color=C["text"], zorder=4)

# Caption
ax.text(7.5, 0.07,
        "Figure 1 — Three-module architecture of the AneuXplain system "
        "(DICOM input → segmentation → three parallel modules: "
        "risk predictor, heatmap, morphology → clinical report).",
        ha="center", va="bottom", fontsize=7.5,
        color=C["sub"], style="italic")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
plt.tight_layout(pad=0)
plt.savefig("figure1_architecture.png", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig("figure1_architecture.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: figure1_architecture.png  (300 DPI)")
print("Saved: figure1_architecture.pdf  (vector)")
