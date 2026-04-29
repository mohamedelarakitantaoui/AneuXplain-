"""
generate_arch_slides.py
Generates the 4 architecture slides for the capstone presentation:
  Slide 11 — System Architecture (wide view)
  Slide 12 — Backend Zoom-in
  Slide 13 — DICOM Pipeline Zoom-in
  Slide 14 — Overall Architecture (everything combined)

Visual rules kept consistent across all 4:
  - dark navy background
  - rounded cards with thick colored borders
  - color code:  blue=user, purple=frontend, orange=backend,
                 red/orange/green = three outputs
  - one-line value statement at the bottom of every slide
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---------- shared palette ----------
BG     = "#0E1117"
CARD   = "#161b24"
TXT    = "#ffffff"
SUB    = "#9aa3b2"
LINE   = "#3a4150"

BLUE   = "#4A90D9"   # user / clinician
PURPLE = "#9B59B6"   # frontend
ORANGE = "#E8A838"   # backend
RED    = "#E05C5C"   # risk
GREEN  = "#5BA85A"   # morphology
TEAL   = "#2EC4B6"   # data pipeline accent


# =====================================================================
# helpers shared across all slides
# =====================================================================
def title_block(ax, x, y, big, small, w):
    ax.text(x, y,        big,   ha="center", va="center",
            fontsize=22, fontweight="bold", color=TXT)
    ax.text(x, y - 0.45, small, ha="center", va="center",
            fontsize=11.5, color=SUB, style="italic")


def value_bar(ax, x, y, w, big, small):
    bar = FancyBboxPatch((x, y - 0.45), w, 0.95,
                         boxstyle="round,pad=0.03,rounding_size=0.18",
                         linewidth=2, edgecolor="white",
                         facecolor="#1f2632", zorder=2)
    ax.add_patch(bar)
    ax.text(x + w / 2, y + 0.18, big, ha="center", va="center",
            fontsize=13, fontweight="bold", color=TXT, zorder=3)
    ax.text(x + w / 2, y - 0.20, small, ha="center", va="center",
            fontsize=10.5, color=SUB, style="italic", zorder=3)


def card_with_circle(ax, x, y, w, h, color, badge, title, sub,
                     badge_size=0.55, badge_fs=18):
    body = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.22",
                          linewidth=2.5, edgecolor=color,
                          facecolor=CARD, zorder=3)
    ax.add_patch(body)
    cx, cy = x + w / 2, y + h - 0.85
    ax.add_patch(Circle((cx, cy), badge_size, facecolor=color, zorder=4))
    ax.text(cx, cy, badge, ha="center", va="center",
            fontsize=badge_fs, fontweight="bold", color="white", zorder=5)
    ax.text(x + w / 2, y + h - 1.95, title,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=TXT, zorder=4)
    ax.text(x + w / 2, y + h - 2.55, sub,
            ha="center", va="center", fontsize=10, color=SUB,
            style="italic", zorder=4)


def hline_arrow(ax, x1, x2, y, color="#7d8696"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.5, mutation_scale=22), zorder=2)


def vdown_arrow(ax, x, y1, y2, color="#7d8696"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.5, mutation_scale=22), zorder=2)


# =====================================================================
# SLIDE 11 — System Architecture (wide view)
# =====================================================================
def slide_11():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    fig.patch.set_facecolor(BG)

    title_block(ax, 8, 8.05,
                "System Architecture",
                "Frontend  →  Backend  →  Three outputs", 16)

    Y_CARD = 4.2
    CH = 2.9

    card_with_circle(ax, 0.4, Y_CARD, 2.4, CH, BLUE,   "MD",
                     "Clinician", "uploads scan")
    hline_arrow(ax, 2.95, 3.85, Y_CARD + CH / 2)

    card_with_circle(ax, 3.9, Y_CARD, 3.0, CH, PURPLE, "UI",
                     "Frontend", "React 19  ·  Three.js")
    hline_arrow(ax, 7.05, 7.95, Y_CARD + CH / 2)

    card_with_circle(ax, 8.0, Y_CARD, 3.0, CH, ORANGE, "AI",
                     "Backend", "FastAPI  ·  PyTorch")
    hline_arrow(ax, 11.15, 12.05, Y_CARD + CH / 2)

    OX, OY, OW = 12.1, Y_CARD - 0.2, 3.5
    out_box = FancyBboxPatch((OX, OY), OW, CH + 0.4,
                             boxstyle="round,pad=0.05,rounding_size=0.22",
                             linewidth=2.5, edgecolor="#7d8696",
                             facecolor=CARD, zorder=3)
    ax.add_patch(out_box)
    ax.text(OX + OW / 2, OY + CH + 0.05, "Three Outputs",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=TXT, zorder=4)

    minis = [(RED, "1", "Risk Score"),
             (ORANGE, "2", "Heatmap"),
             (GREEN, "3", "Morphology")]
    mh = 0.65
    for i, (c, ic, lab) in enumerate(minis):
        my = OY + 0.35 + (2 - i) * (mh + 0.18)
        ax.add_patch(FancyBboxPatch((OX + 0.25, my), OW - 0.5, mh,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=0, facecolor=c, alpha=0.20,
                                    zorder=4))
        ax.add_patch(Circle((OX + 0.6, my + mh / 2), 0.22,
                            facecolor=c, zorder=5))
        ax.text(OX + 0.6, my + mh / 2, ic, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)
        ax.text(OX + 1.05, my + mh / 2, lab, ha="left", va="center",
                fontsize=11, fontweight="bold", color=TXT, zorder=6)

    # pipeline strip
    PIPE_Y, PIPE_H = 2.1, 1.2
    PIPE_X, PIPE_W = 0.5, 15
    ax.add_patch(FancyBboxPatch((PIPE_X, PIPE_Y), PIPE_W, PIPE_H,
                                boxstyle="round,pad=0.04,rounding_size=0.18",
                                linewidth=1.5, edgecolor=LINE,
                                facecolor="#1a1f2a", zorder=2))
    ax.text(PIPE_X + 0.4, PIPE_Y + PIPE_H - 0.28, "PIPELINE",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=SUB, zorder=3)
    stages = [(BLUE, "1", "Upload"),
              (PURPLE, "2", "Segmentation"),
              (ORANGE, "3", "Three-pathway analysis"),
              (GREEN, "4", "Report")]
    n = len(stages)
    slot_w = (PIPE_W - 0.8) / n
    for i, (c, num, lab) in enumerate(stages):
        cx = PIPE_X + 0.4 + slot_w * (i + 0.5)
        ax.add_patch(Circle((cx - 1.4, PIPE_Y + PIPE_H / 2 - 0.1), 0.30,
                            facecolor=c, zorder=4))
        ax.text(cx - 1.4, PIPE_Y + PIPE_H / 2 - 0.1, num,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=5)
        ax.text(cx - 1.0, PIPE_Y + PIPE_H / 2 - 0.1, lab,
                ha="left", va="center", fontsize=12,
                fontweight="bold", color=TXT, zorder=5)
        if i < n - 1:
            ax.annotate("", xy=(cx + slot_w * 0.95 - 1.4,
                                PIPE_Y + PIPE_H / 2 - 0.1),
                        xytext=(cx + 0.1, PIPE_Y + PIPE_H / 2 - 0.1),
                        arrowprops=dict(arrowstyle="-|>", color="#7d8696",
                                        lw=1.8, mutation_scale=14), zorder=3)

    value_bar(ax, 0.5, 0.55, 15,
              "Each pathway is independent.",
              "When one is uncertain, the other two still help the doctor.")

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/slide_11_system_architecture.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved -> figures/slide_11_system_architecture.png")


# =====================================================================
# SLIDE 12 — Backend Zoom-in
# =====================================================================
def slide_12():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    fig.patch.set_facecolor(BG)

    title_block(ax, 8, 8.05,
                "Inside the Backend",
                "FastAPI orchestrates four services — each one does one thing well.",
                16)

    # outer "Backend" wrapper to signal we are zoomed-in
    OX, OY, OW, OH = 0.5, 2.1, 15, 5.0
    wrapper = FancyBboxPatch((OX, OY), OW, OH,
                             boxstyle="round,pad=0.05,rounding_size=0.25",
                             linewidth=2.5, edgecolor=ORANGE,
                             facecolor="#181d27", zorder=1)
    ax.add_patch(wrapper)
    # tag at the top-left of the wrapper
    tag = FancyBboxPatch((OX + 0.3, OY + OH - 0.55), 2.8, 0.5,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=0, facecolor=ORANGE, zorder=2)
    ax.add_patch(tag)
    ax.text(OX + 1.7, OY + OH - 0.30, "BACKEND  (FastAPI)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=3)

    # 4 service cards
    services = [
        (BLUE,   "API",   "API Service",
         "FastAPI routes\nrequest validation",     "front door"),
        (PURPLE, "MESH",  "Mesh Service",
         "DICOM → 3D mesh\nvtk · trimesh",         "geometry"),
        (RED,    "AI",    "Inference Service",
         "PointNet + PyTorch\nrisk + heatmap",     "prediction"),
        (GREEN,  "MORPH", "Morphology Service",
         "8 measurements\nclinical explainer",     "clinical features"),
    ]
    n = len(services)
    GAP = 0.4
    inner_pad = 0.6
    avail = OW - 2 * inner_pad - (n - 1) * GAP
    SW = avail / n
    SH = 3.6
    SY = OY + 0.55
    for i, (c, badge, title, sub, tag_text) in enumerate(services):
        sx = OX + inner_pad + i * (SW + GAP)
        body = FancyBboxPatch((sx, SY), SW, SH,
                              boxstyle="round,pad=0.05,rounding_size=0.22",
                              linewidth=2.5, edgecolor=c,
                              facecolor=CARD, zorder=3)
        ax.add_patch(body)
        # badge
        cx, cy = sx + SW / 2, SY + SH - 0.85
        ax.add_patch(Circle((cx, cy), 0.55, facecolor=c, zorder=4))
        ax.text(cx, cy, badge, ha="center", va="center",
                fontsize=12 if len(badge) > 3 else 16,
                fontweight="bold", color="white", zorder=5)
        # title
        ax.text(sx + SW / 2, SY + SH - 1.95, title,
                ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=TXT, zorder=4)
        # sub (two-line tech stack / role)
        ax.text(sx + SW / 2, SY + SH - 2.65, sub,
                ha="center", va="center", fontsize=10,
                color=SUB, style="italic", zorder=4)
        # role pill
        pill_w = SW - 0.6
        ax.add_patch(FancyBboxPatch((sx + 0.3, SY + 0.25),
                                    pill_w, 0.45,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    linewidth=0, facecolor=c, alpha=0.20,
                                    zorder=3))
        ax.text(sx + SW / 2, SY + 0.47, tag_text,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=c, zorder=4)

        # connector arrow to next card
        if i < n - 1:
            ax.annotate("", xy=(sx + SW + GAP - 0.05, SY + SH / 2),
                        xytext=(sx + SW + 0.05, SY + SH / 2),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#7d8696",
                                        lw=2, mutation_scale=14), zorder=4)

    value_bar(ax, 0.5, 0.55, 15,
              "Independent services, one orchestrator.",
              "If one service fails, the others still deliver useful output.")

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/slide_12_backend_zoom.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved -> figures/slide_12_backend_zoom.png")


# =====================================================================
# SLIDE 13 — DICOM Pipeline Zoom-in
# =====================================================================
def slide_13():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    fig.patch.set_facecolor(BG)

    title_block(ax, 8, 8.05,
                "From Scan to 3D Mesh",
                "Five stages turn a stack of DICOM slices into 1,024 points the AI can read.",
                16)

    # outer wrapper to signal we are inside the data pipeline
    OX, OY, OW, OH = 0.5, 2.1, 15, 5.0
    wrapper = FancyBboxPatch((OX, OY), OW, OH,
                             boxstyle="round,pad=0.05,rounding_size=0.25",
                             linewidth=2.5, edgecolor=TEAL,
                             facecolor="#152022", zorder=1)
    ax.add_patch(wrapper)
    tag = FancyBboxPatch((OX + 0.3, OY + OH - 0.55), 3.0, 0.5,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=0, facecolor=TEAL, zorder=2)
    ax.add_patch(tag)
    ax.text(OX + 1.8, OY + OH - 0.30, "DICOM PIPELINE",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=3)

    # 5 stages (matches your real backend modules)
    stages = [
        (BLUE,   "1", "DICOM Loader",
         "read .dcm slices\n+ metadata",          "loader.py"),
        (PURPLE, "2", "Harmonizer",
         "normalize spacing\norientation",        "harmonizer.py"),
        (ORANGE, "3", "Segmenter",
         "find vessels\nbinary mask",             "segmenter.py"),
        (RED,    "4", "Mesher",
         "marching cubes\ntriangle mesh",         "mesher.py"),
        (GREEN,  "5", "Cropper",
         "click → crop\n1,024 points",            "cropper.py"),
    ]
    n = len(stages)
    GAP = 0.30
    inner_pad = 0.4
    avail = OW - 2 * inner_pad - (n - 1) * GAP
    SW = avail / n
    SH = 3.6
    SY = OY + 0.55
    for i, (c, num, title, sub, fn) in enumerate(stages):
        sx = OX + inner_pad + i * (SW + GAP)
        ax.add_patch(FancyBboxPatch((sx, SY), SW, SH,
                                    boxstyle="round,pad=0.05,rounding_size=0.22",
                                    linewidth=2.5, edgecolor=c,
                                    facecolor=CARD, zorder=3))
        # number badge
        cx, cy = sx + SW / 2, SY + SH - 0.85
        ax.add_patch(Circle((cx, cy), 0.50, facecolor=c, zorder=4))
        ax.text(cx, cy, num, ha="center", va="center",
                fontsize=18, fontweight="bold", color="white", zorder=5)
        # title
        ax.text(sx + SW / 2, SY + SH - 1.85, title,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color=TXT, zorder=4)
        # description
        ax.text(sx + SW / 2, SY + SH - 2.55, sub,
                ha="center", va="center", fontsize=9.5,
                color=SUB, style="italic", zorder=4)
        # filename pill
        ax.add_patch(FancyBboxPatch((sx + 0.25, SY + 0.25),
                                    SW - 0.5, 0.45,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    linewidth=0, facecolor=c, alpha=0.18,
                                    zorder=3))
        ax.text(sx + SW / 2, SY + 0.47, fn,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=c, zorder=4)

        if i < n - 1:
            ax.annotate("", xy=(sx + SW + GAP - 0.02, SY + SH / 2),
                        xytext=(sx + SW + 0.02, SY + SH / 2),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#7d8696",
                                        lw=2, mutation_scale=14), zorder=4)

    value_bar(ax, 0.5, 0.55, 15,
              "Every step is visible to the doctor.",
              "No black box before the AI — each stage can be inspected.")

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/slide_13_dicom_pipeline.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved -> figures/slide_13_dicom_pipeline.png")


# =====================================================================
# SLIDE 14 — Overall Architecture (everything combined)
# =====================================================================
def slide_14():
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.2); ax.axis("off")
    fig.patch.set_facecolor(BG)

    title_block(ax, 8, 8.75,
                "Putting It All Together",
                "Frontend  ·  backend services  ·  data pipeline  ·  three outputs",
                16)

    # ---------------- TIER 1: client (clinician + frontend)
    T1_Y, T1_H = 7.05, 1.10
    ax.add_patch(FancyBboxPatch((0.5, T1_Y), 15, T1_H,
                                boxstyle="round,pad=0.04,rounding_size=0.18",
                                linewidth=2, edgecolor=BLUE,
                                facecolor="#161e2a", zorder=2))
    ax.text(0.85, T1_Y + T1_H - 0.22, "CLIENT",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=BLUE, zorder=3)
    # MD pill
    ax.add_patch(Circle((2.0, T1_Y + 0.45), 0.32,
                        facecolor=BLUE, zorder=3))
    ax.text(2.0, T1_Y + 0.45, "MD", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=4)
    ax.text(2.55, T1_Y + 0.45, "Clinician",
            ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=TXT, zorder=4)
    # arrow to frontend
    ax.annotate("", xy=(6.0, T1_Y + 0.45), xytext=(4.4, T1_Y + 0.45),
                arrowprops=dict(arrowstyle="-|>", color="#7d8696",
                                lw=2, mutation_scale=16), zorder=3)
    # frontend pill
    ax.add_patch(FancyBboxPatch((6.0, T1_Y + 0.18), 8.5, 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=0, facecolor=PURPLE, alpha=0.30,
                                zorder=3))
    ax.add_patch(Circle((6.45, T1_Y + 0.45), 0.28,
                        facecolor=PURPLE, zorder=4))
    ax.text(6.45, T1_Y + 0.45, "UI", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=5)
    ax.text(6.95, T1_Y + 0.45,
            "Frontend  ·  React 19  ·  Three.js  ·  3D viewer  ·  heatmap overlay",
            ha="left", va="center", fontsize=11,
            fontweight="bold", color=TXT, zorder=5)

    # arrow down to tier 2
    vdown_arrow(ax, 8, T1_Y - 0.05, 6.65)

    # ---------------- TIER 2: backend (FastAPI + 4 services + data pipeline)
    T2_Y, T2_H = 3.35, 3.30
    ax.add_patch(FancyBboxPatch((0.5, T2_Y), 15, T2_H,
                                boxstyle="round,pad=0.04,rounding_size=0.18",
                                linewidth=2, edgecolor=ORANGE,
                                facecolor="#1c1a18", zorder=2))
    ax.text(0.85, T2_Y + T2_H - 0.22, "BACKEND  (FastAPI)",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=ORANGE, zorder=3)

    # 4 services row
    services = [(BLUE, "API"), (PURPLE, "MESH"),
                (RED, "AI"), (GREEN, "MORPH")]
    s_y = T2_Y + T2_H - 1.50
    s_h = 0.65
    s_w = 3.3
    s_gap = 0.25
    s_x0 = 0.5 + (15 - (4 * s_w + 3 * s_gap)) / 2
    for i, (c, lab) in enumerate(services):
        sx = s_x0 + i * (s_w + s_gap)
        ax.add_patch(FancyBboxPatch((sx, s_y), s_w, s_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=2, edgecolor=c,
                                    facecolor=CARD, zorder=3))
        ax.add_patch(Circle((sx + 0.45, s_y + s_h / 2), 0.22,
                            facecolor=c, zorder=4))
        ax.text(sx + 0.45, s_y + s_h / 2, lab[0] if len(lab) > 3 else lab,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white", zorder=5)
        ax.text(sx + 0.85, s_y + s_h / 2, lab,
                ha="left", va="center", fontsize=11,
                fontweight="bold", color=TXT, zorder=5)

    # data pipeline strip inside backend
    p_y, p_h = T2_Y + 0.40, 0.85
    ax.add_patch(FancyBboxPatch((0.85, p_y), 14.3, p_h,
                                boxstyle="round,pad=0.03,rounding_size=0.15",
                                linewidth=1.5, edgecolor=TEAL,
                                facecolor="#152022", zorder=3))
    ax.text(1.15, p_y + p_h - 0.20, "DATA PIPELINE",
            ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=TEAL, zorder=4)

    pipe_stages = ["DICOM", "Harmonize", "Segment",
                   "Mesh", "Crop", "1,024 pts"]
    pipe_colors = [BLUE, PURPLE, ORANGE, RED, GREEN, TEAL]
    pn = len(pipe_stages)
    p_inner = 2.55
    p_avail = 14.3 - p_inner - 0.3
    step = p_avail / (pn - 1)
    for i, (lab, c) in enumerate(zip(pipe_stages, pipe_colors)):
        cx = 0.85 + p_inner + i * step
        ax.add_patch(Circle((cx, p_y + 0.32), 0.18,
                            facecolor=c, zorder=4))
        ax.text(cx, p_y + 0.32, str(i + 1),
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=5)
        ax.text(cx, p_y + 0.62, lab,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=TXT, zorder=5)
        if i < pn - 1:
            ax.annotate("", xy=(cx + step - 0.22, p_y + 0.32),
                        xytext=(cx + 0.22, p_y + 0.32),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#7d8696",
                                        lw=1.4, mutation_scale=10), zorder=4)

    # arrow from tier 2 to tier 3
    vdown_arrow(ax, 8, T2_Y - 0.05, 2.55)

    # ---------------- TIER 3: three outputs
    T3_Y, T3_H = 1.50, 1.05
    ax.add_patch(FancyBboxPatch((0.5, T3_Y), 15, T3_H,
                                boxstyle="round,pad=0.04,rounding_size=0.18",
                                linewidth=2, edgecolor="#7d8696",
                                facecolor="#1a1f2a", zorder=2))
    ax.text(0.85, T3_Y + T3_H - 0.22, "THREE OUTPUTS",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=SUB, zorder=3)

    outs = [(RED, "1", "Risk Score"),
            (ORANGE, "2", "Heatmap"),
            (GREEN, "3", "Morphology Report")]
    ow = 4.5
    ogap = 0.35
    ox0 = 0.5 + (15 - (3 * ow + 2 * ogap)) / 2
    o_y = T3_Y + 0.20
    o_h = 0.55
    for i, (c, num, lab) in enumerate(outs):
        ox = ox0 + i * (ow + ogap)
        ax.add_patch(FancyBboxPatch((ox, o_y), ow, o_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=0, facecolor=c, alpha=0.22,
                                    zorder=3))
        ax.add_patch(Circle((ox + 0.4, o_y + o_h / 2), 0.20,
                            facecolor=c, zorder=4))
        ax.text(ox + 0.4, o_y + o_h / 2, num,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=5)
        ax.text(ox + ow / 2 + 0.15, o_y + o_h / 2, lab,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color=TXT, zorder=5)

    # ---------------- bottom value bar
    value_bar(ax, 0.5, 0.55, 15,
              "One scan in.  Three answers out.",
              "Each pathway is independently checkable — that is the clinical value.")

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/slide_14_overall_architecture.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved -> figures/slide_14_overall_architecture.png")


# =====================================================================
if __name__ == "__main__":
    slide_11()
    slide_12()
    slide_13()
    slide_14()
