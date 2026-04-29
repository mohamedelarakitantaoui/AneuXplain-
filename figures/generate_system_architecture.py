"""
generate_system_architecture.py
System architecture diagram for capstone slide 11.
Minimal text, icon-driven, left-to-right flow.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BG     = "#0E1117"
CARD   = "#161b24"
TXT    = "#ffffff"
SUB    = "#9aa3b2"
BLUE   = "#4A90D9"
PURPLE = "#9B59B6"
RED    = "#E05C5C"
ORANGE = "#E8A838"
GREEN  = "#5BA85A"

fig, ax = plt.subplots(figsize=(16, 8.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8.5)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ---------------- title
ax.text(8, 8.05, "System Architecture",
        ha="center", va="center", fontsize=22, fontweight="bold", color=TXT)
ax.text(8, 7.6, "Frontend  →  Backend  →  Three outputs",
        ha="center", va="center", fontsize=11.5, color=SUB, style="italic")

# ---------------- helper: rounded card with header strip
def card(x, y, w, h, color, icon, title, sub):
    # body
    body = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.22",
                          linewidth=2.5, edgecolor=color,
                          facecolor=CARD, zorder=3)
    ax.add_patch(body)
    # icon circle
    cx, cy = x + w / 2, y + h - 0.85
    ax.add_patch(Circle((cx, cy), 0.55, facecolor=color, zorder=4))
    ax.text(cx, cy, icon, ha="center", va="center",
            fontsize=18, fontweight="bold", color="white", zorder=5)
    # title
    ax.text(x + w / 2, y + h - 1.95, title,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=TXT, zorder=4)
    # subtitle (tech stack / note)
    ax.text(x + w / 2, y + h - 2.55, sub,
            ha="center", va="center", fontsize=10, color=SUB,
            style="italic", zorder=4)

# arrow between cards
def arrow(x1, x2, y, color="#7d8696"):
    a = FancyArrowPatch((x1, y), (x2, y),
                        arrowstyle="-|>", mutation_scale=22,
                        color=color, lw=2.5, zorder=2)
    ax.add_patch(a)

# ---------------- four main blocks (clinician / frontend / backend / outputs-group)
Y_CARD = 4.0
CH = 2.9

# 1. Clinician
card(0.4, Y_CARD, 2.4, CH, BLUE, "MD", "Clinician", "uploads scan")
arrow(2.95, 3.85, Y_CARD + CH / 2)

# 2. Frontend
card(3.9, Y_CARD, 3.0, CH, PURPLE, "UI", "Frontend", "React 19  ·  Three.js")
arrow(7.05, 7.95, Y_CARD + CH / 2)

# 3. Backend
card(8.0, Y_CARD, 3.0, CH, ORANGE, "AI", "Backend", "FastAPI  ·  PyTorch")
arrow(11.15, 12.05, Y_CARD + CH / 2)

# 4. Outputs (a single grouping box that contains 3 mini cards)
OX, OY, OW = 12.1, Y_CARD - 0.2, 3.5
out_box = FancyBboxPatch((OX, OY), OW, CH + 0.4,
                         boxstyle="round,pad=0.05,rounding_size=0.22",
                         linewidth=2.5, edgecolor="#7d8696",
                         facecolor=CARD, zorder=3)
ax.add_patch(out_box)
ax.text(OX + OW / 2, OY + CH + 0.05, "Three Outputs",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=TXT, zorder=4)

mini_specs = [
    (RED,    "1", "Risk Score"),
    (ORANGE, "2", "Heatmap"),
    (GREEN,  "3", "Morphology"),
]
mini_h = 0.65
mini_y0 = OY + 0.35
for i, (c, ic, lab) in enumerate(mini_specs):
    my = mini_y0 + (2 - i) * (mini_h + 0.18)
    mini = FancyBboxPatch((OX + 0.25, my), OW - 0.5, mini_h,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          linewidth=0, facecolor=c, alpha=0.20, zorder=4)
    ax.add_patch(mini)
    ax.add_patch(Circle((OX + 0.6, my + mini_h / 2), 0.22,
                        facecolor=c, zorder=5))
    ax.text(OX + 0.6, my + mini_h / 2, ic, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6)
    ax.text(OX + 1.05, my + mini_h / 2, lab, ha="left", va="center",
            fontsize=11, fontweight="bold", color=TXT, zorder=6)

# ---------------- pipeline strip at the bottom
PIPE_Y = 2.1
PIPE_H = 1.2
PIPE_X = 0.5
PIPE_W = 15
pipe_bg = FancyBboxPatch((PIPE_X, PIPE_Y), PIPE_W, PIPE_H,
                         boxstyle="round,pad=0.04,rounding_size=0.18",
                         linewidth=1.5, edgecolor="#3a4150",
                         facecolor="#1a1f2a", zorder=2)
ax.add_patch(pipe_bg)
ax.text(PIPE_X + 0.4, PIPE_Y + PIPE_H - 0.28, "PIPELINE",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=SUB, zorder=3)

stages = [
    (BLUE,   "1", "Upload"),
    (PURPLE, "2", "Segmentation"),
    (ORANGE, "3", "Three-pathway analysis"),
    (GREEN,  "4", "Report"),
]
n = len(stages)
slot_w = (PIPE_W - 0.8) / n
for i, (c, num, lab) in enumerate(stages):
    cx = PIPE_X + 0.4 + slot_w * (i + 0.5)
    # number circle
    ax.add_patch(Circle((cx - 1.4, PIPE_Y + PIPE_H / 2 - 0.1),
                        0.30, facecolor=c, zorder=4))
    ax.text(cx - 1.4, PIPE_Y + PIPE_H / 2 - 0.1, num,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white", zorder=5)
    # label
    ax.text(cx - 1.0, PIPE_Y + PIPE_H / 2 - 0.1, lab,
            ha="left", va="center", fontsize=12, fontweight="bold",
            color=TXT, zorder=5)
    # connector arrow to the next stage
    if i < n - 1:
        a = FancyArrowPatch((cx + slot_w - 1.6, PIPE_Y + PIPE_H / 2 - 0.1),
                            (cx + slot_w - 1.55, PIPE_Y + PIPE_H / 2 - 0.1))
        # use a simple arrow line
        ax.annotate("", xy=(cx + slot_w * 0.95 - 1.4,
                            PIPE_Y + PIPE_H / 2 - 0.1),
                    xytext=(cx + 0.1, PIPE_Y + PIPE_H / 2 - 0.1),
                    arrowprops=dict(arrowstyle="-|>", color="#7d8696",
                                    lw=1.8, mutation_scale=14), zorder=3)

# ---------------- bottom message bar
MSG_Y = 0.55
msg = FancyBboxPatch((0.5, MSG_Y - 0.45), 15, 0.95,
                     boxstyle="round,pad=0.03,rounding_size=0.18",
                     linewidth=2, edgecolor="white",
                     facecolor="#1f2632", zorder=2)
ax.add_patch(msg)
ax.text(8, MSG_Y + 0.18, "Each pathway is independent.",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=TXT, zorder=3)
ax.text(8, MSG_Y - 0.22,
        "When one is uncertain, the other two still help the doctor.",
        ha="center", va="center", fontsize=10.5, color=SUB,
        style="italic", zorder=3)

plt.tight_layout(pad=0.3)
out_path = "figures/system_architecture_diagram.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
