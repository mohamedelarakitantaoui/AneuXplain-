"""
Compute classification metrics from batch_results.csv.

Treats `group=patient` as the positive class and `group=control` as negative.
A subject is predicted positive when `predicted_class` is HIGH or CRITICAL,
and negative when LOW or MODERATE. Rows without a successful pipeline run
are reported separately and excluded from the metrics.

Also computes ROC-AUC on `risk_score` (continuous) and reports the
threshold-free ranking quality.

Usage:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --csv path/to/batch_results.csv
    python scripts/compute_metrics.py --positive-classes HIGH,CRITICAL,MODERATE
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_CSV = Path(r"C:\Users\buzok\Desktop\Test-demo\results\batch_results.csv")
DEFAULT_POSITIVE_CLASSES = {"HIGH", "CRITICAL"}


def roc_auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """Mann-Whitney U-based AUC. Returns None if either group is empty."""
    if not scores_pos or not scores_neg:
        return None
    all_scored = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    all_scored.sort(key=lambda t: t[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < len(all_scored):
        j = i
        while j < len(all_scored) and all_scored[j][0] == all_scored[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if all_scored[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += j - i
        i = j
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--positive-classes",
        type=str,
        default=",".join(sorted(DEFAULT_POSITIVE_CLASSES)),
        help="Comma-separated predicted_class values that count as positive.",
    )
    args = p.parse_args()

    if not args.csv.exists():
        sys.stderr.write(f"CSV not found: {args.csv}\n")
        return 1

    positive_classes = {c.strip().upper() for c in args.positive_classes.split(",") if c.strip()}

    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    n_total = len(rows)
    successes = [r for r in rows if r.get("pipeline_status") == "success"]
    n_skipped = n_total - len(successes)

    tp = fp = tn = fn = 0
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    confusion_examples: dict[str, list[str]] = {"TP": [], "FP": [], "TN": [], "FN": []}

    for r in successes:
        group = (r.get("group") or "").strip().lower()
        pred_cls = (r.get("predicted_class") or "").strip().upper()
        is_truth_pos = group == "patient"
        is_pred_pos = pred_cls in positive_classes

        sid = r.get("subject_id", "?")
        if is_truth_pos and is_pred_pos:
            tp += 1
            confusion_examples["TP"].append(sid)
        elif is_truth_pos and not is_pred_pos:
            fn += 1
            confusion_examples["FN"].append(sid)
        elif (not is_truth_pos) and is_pred_pos:
            fp += 1
            confusion_examples["FP"].append(sid)
        else:
            tn += 1
            confusion_examples["TN"].append(sid)

        try:
            score = float(r.get("risk_score") or "")
        except ValueError:
            score = None
        if score is not None:
            (pos_scores if is_truth_pos else neg_scores).append(score)

    n_eval = tp + tn + fp + fn
    accuracy = (tp + tn) / n_eval if n_eval else None
    sensitivity = tp / (tp + fn) if (tp + fn) else None  # recall on positives
    specificity = tn / (tn + fp) if (tn + fp) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision and sensitivity
        else None
    )
    auc = roc_auc(pos_scores, neg_scores)

    def fmt(x: float | None) -> str:
        return f"{x:.3f}" if x is not None else "n/a"

    print("=" * 70)
    print(f"AneuXplain metrics — {args.csv.name}")
    print("=" * 70)
    print(f"Total rows in CSV     : {n_total}")
    print(f"Excluded (non-success): {n_skipped}")
    print(f"Evaluated             : {n_eval}")
    print(f"Positive classes      : {sorted(positive_classes)}")
    print()
    print("Confusion matrix (truth = group, pred = predicted_class):")
    print(f"                     pred POS    pred NEG")
    print(f"  truth POS (patient)  {tp:>5}        {fn:>5}")
    print(f"  truth NEG (control)  {fp:>5}        {tn:>5}")
    print()
    print(f"Accuracy             : {fmt(accuracy)}")
    print(f"Sensitivity (recall) : {fmt(sensitivity)}")
    print(f"Specificity          : {fmt(specificity)}")
    print(f"Precision (PPV)      : {fmt(precision)}")
    print(f"F1                   : {fmt(f1)}")
    print(f"ROC-AUC (risk_score) : {fmt(auc)}   "
          f"[{len(pos_scores)} pos vs {len(neg_scores)} neg with scores]")
    print()
    if confusion_examples["FP"]:
        print(f"False positives ({len(confusion_examples['FP'])}): "
              + ", ".join(confusion_examples["FP"]))
    if confusion_examples["FN"]:
        print(f"False negatives ({len(confusion_examples['FN'])}): "
              + ", ".join(confusion_examples["FN"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
