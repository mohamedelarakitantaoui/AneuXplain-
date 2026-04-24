import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

CSV_PATH = "test_predictions.csv"
OUT_PATH = "figures/fig_7_1_roc_intra.pdf"
CURVE_COLOR = "#3B82F6"
YOUDEN_THRESHOLD = 0.27
AUC_VALUE = 0.834

df = pd.read_csv(CSV_PATH)
y_true = df["y_true"].values
y_pred_proba = df["y_pred_proba"].values

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

idx = int(np.argmin(np.abs(thresholds - YOUDEN_THRESHOLD)))
fpr_opt, tpr_opt = fpr[idx], tpr[idx]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color=CURVE_COLOR, lw=2, label="ROC curve")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")

ax.scatter(
    [fpr_opt], [tpr_opt],
    s=70, color=CURVE_COLOR, edgecolor="black", zorder=5,
)
ax.annotate(
    f"Youden-optimal\n(threshold = {YOUDEN_THRESHOLD:.2f})",
    xy=(fpr_opt, tpr_opt),
    xytext=(fpr_opt + 0.12, tpr_opt - 0.12),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

ax.text(
    0.97, 0.05,
    f"AUC = {AUC_VALUE:.3f}",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", lw=0.8),
)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Intra-dataset Test")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18), frameon=False)
ax.grid(alpha=0.3)

plt.savefig(OUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_PATH}")
