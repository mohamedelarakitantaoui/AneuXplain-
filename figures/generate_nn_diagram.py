"""
generate_nn_diagram.py
Simple, clean PointNet diagram for capstone presentation slide.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_BG     = "#FFFFFF"
BLOCKS   = [
    ("#4A90D9", "Input\nPoint Cloud", "1,024 points\n(x, y, z)"),
    ("#5BA85A", "Shared\nMLP",        "Per-point\nfeature extraction"),
    ("#E8A838", "Global\nMax Pool",   "Aggregate all\npoints → 1 vector"),
    ("#E05C5C", "FC\nLayers",         "128 → 64 → 32 → 16"),
    ("#2ECC71", "Risk\nScore",        "0 = healthy\n1 = high risk"),
]

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

BW, BH = 1.9, 1.6
GAP    = 0.55
Y_BOX  = 2.5
xs     = [1.2 + i * (BW + GAP) for i in range(len(BLOCKS))]

for i, (x, (color, title, sub)) in enumerate(zip(xs, BLOCKS)):
    # shadow
    shadow = FancyBboxPatch((x - BW/2 + 0.07, Y_BOX - BH/2 - 0.07), BW, BH,
                             boxstyle="round,pad=0.12", linewidth=0,
                             facecolor="#cccccc", zorder=2)
    ax.add_patch(shadow)
    # box
    box = FancyBboxPatch((x - BW/2, Y_BOX - BH/2), BW, BH,
                          boxstyle="round,pad=0.12", linewidth=0,
                          facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, Y_BOX + 0.22, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=4)
    ax.text(x, Y_BOX - 0.35, sub, ha="center", va="center",
            fontsize=8.5, color="white", alpha=0.92, zorder=4)

    # arrow to next block
    if i < len(BLOCKS) - 1:
        x_next = xs[i + 1]
        ax.annotate("", xy=(x_next - BW/2 - 0.05, Y_BOX),
                    xytext=(x + BW/2 + 0.05, Y_BOX),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                   lw=2.2, mutation_scale=18), zorder=5)

# title
ax.text(6.5, 4.55, "PointNet — Aneurysm Risk Predictor",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#222222")

# training strip at the bottom
strip_items = [
    "Loss: BCE",
    "Optimizer: Adam  (lr = 2e-4)",
    "80 epochs  ·  batch 16",
    "Google Colab T4 GPU",
    "AUC-ROC: 0.834",
]
colors_strip = ["#4A90D9", "#5BA85A", "#E8A838", "#9B59B6", "#E05C5C"]
strip_x = [1.0, 3.5, 6.2, 9.0, 11.5]
for sx, sc, st in zip(strip_x, colors_strip, strip_items):
    ax.text(sx, 0.72, st, ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=sc, edgecolor="none"))

plt.tight_layout(pad=0.3)
out_path = "figures/nn_architecture_diagram.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {out_path}")
