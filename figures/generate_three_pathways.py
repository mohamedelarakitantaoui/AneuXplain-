"""
generate_three_pathways.py
Three-pathways-of-explanation diagram for capstone slide 9.
Three colored cards (risk / heatmap / morphology) converging on the
"check each other" value statement.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#0E1117"

CARDS = [
    {
        "color": "#E05C5C",   # red
        "num":   "01",
        "title": "RISK SCORE",
        "head":  "A single number\nfrom the AI",
        "bullets": [
            "Output: 0.0  -  1.0",
            "Source: PointNet classifier",
            "Sigmoid activation",
        ],
        "tag": "Tells you WHAT",
    },
    {
        "color": "#E8A838",   # orange
        "num":   "02",
        "title": "HEATMAP",
        "head":  "Where the AI\nis looking",
        "bullets": [
            "Per-point importance",
            "Overlay on 3D mesh",
            "Highlights driving region",
        ],
        "tag": "Tells you WHERE",
    },
    {
        "color": "#5BA85A",   # green
        "num":   "03",
        "title": "MORPHOLOGY REPORT",
        "head":  "Eight medical\nmeasurements",
        "bullets": [
            "Neck width / Dome height",
            "Aspect ratio / Size ratio",
            "Each vs literature range",
        ],
        "tag": "Tells you WHY",
    },
]

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")
fig.patch.set_facecolor(BG)

# title
ax.text(7.5, 8.05, "Three Pathways of Explanation",
        ha="center", va="center", fontsize=22, fontweight="bold", color="white")
ax.text(7.5, 7.55,
        "One prediction is not enough — three answers that check each other.",
        ha="center", va="center", fontsize=12, color="#9aa3b2", style="italic")

# cards
CW, CH = 4.0, 4.6
GAP    = 0.7
total_w = 3 * CW + 2 * GAP
x_start = (15 - total_w) / 2
Y_BOT   = 1.95

for i, card in enumerate(CARDS):
    x = x_start + i * (CW + GAP)

    # outer card (dark with colored border)
    outer = FancyBboxPatch((x, Y_BOT), CW, CH,
                           boxstyle="round,pad=0.05,rounding_size=0.25",
                           linewidth=3, edgecolor=card["color"],
                           facecolor="#161b24", zorder=2)
    ax.add_patch(outer)

    # colored header strip
    header_h = 0.85
    header = FancyBboxPatch((x, Y_BOT + CH - header_h), CW, header_h,
                            boxstyle="round,pad=0.05,rounding_size=0.25",
                            linewidth=0, facecolor=card["color"], zorder=3)
    ax.add_patch(header)
    # mask the bottom of the header so it sits flat against the card body
    ax.add_patch(plt.Rectangle((x + 0.05, Y_BOT + CH - header_h - 0.02),
                               CW - 0.10, 0.25, facecolor=card["color"],
                               linewidth=0, zorder=3))

    # number on the left of the header
    ax.text(x + 0.45, Y_BOT + CH - header_h / 2, card["num"],
            ha="left", va="center", fontsize=22, fontweight="bold",
            color="white", zorder=4)
    # title on the right of the header
    ax.text(x + CW - 0.3, Y_BOT + CH - header_h / 2, card["title"],
            ha="right", va="center", fontsize=12, fontweight="bold",
            color="white", zorder=4)

    # main headline inside the card
    ax.text(x + CW / 2, Y_BOT + CH - header_h - 0.85, card["head"],
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white", zorder=4)

    # bullets
    by = Y_BOT + CH - header_h - 1.85
    for b in card["bullets"]:
        ax.plot(x + 0.55, by + 0.05, marker="o", markersize=5,
                color=card["color"], zorder=4)
        ax.text(x + 0.85, by, b, ha="left", va="center",
                fontsize=10, color="#d6dbe5", zorder=4)
        by -= 0.45

    # tag pill at the bottom
    pill_y = Y_BOT + 0.45
    pill = FancyBboxPatch((x + 0.6, pill_y - 0.22), CW - 1.2, 0.5,
                          boxstyle="round,pad=0.02,rounding_size=0.25",
                          linewidth=0, facecolor=card["color"], alpha=0.18,
                          zorder=3)
    ax.add_patch(pill)
    ax.text(x + CW / 2, pill_y + 0.03, card["tag"],
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=card["color"], zorder=4)

    # arrow down from card to the value bar
    arrow = FancyArrowPatch((x + CW / 2, Y_BOT - 0.05),
                            (x + CW / 2, 1.05),
                            arrowstyle="-|>", mutation_scale=14,
                            color=card["color"], lw=2, zorder=2)
    ax.add_patch(arrow)

# bottom value bar
bar_y = 0.55
bar = FancyBboxPatch((x_start, bar_y - 0.45), total_w, 0.95,
                     boxstyle="round,pad=0.03,rounding_size=0.20",
                     linewidth=2, edgecolor="#ffffff",
                     facecolor="#1f2632", zorder=2)
ax.add_patch(bar)
ax.text(x_start + total_w / 2, bar_y + 0.18,
        "The three outputs check each other.",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="white", zorder=4)
ax.text(x_start + total_w / 2, bar_y - 0.20,
        "Agreement → trust the prediction      Disagreement → look more carefully",
        ha="center", va="center", fontsize=10.5, color="#9aa3b2",
        style="italic", zorder=4)

plt.tight_layout(pad=0.3)
out_path = "figures/three_pathways_diagram.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
