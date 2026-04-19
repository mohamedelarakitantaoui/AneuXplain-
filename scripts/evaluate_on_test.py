"""
Evaluate a trained risk predictor on the locked patient-level TEST split.

Usage:
    python -m scripts.evaluate_on_test [--checkpoint models/risk_predictor_v2.pth]

Outputs:
    outputs/test_evaluation_<timestamp>.csv     per-sample predictions
    outputs/test_metrics_<timestamp>.json       summary metrics + sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from training.models import LabeledArteryDataset, RiskPredictorV2  # noqa: E402
from training.scripts.patient_split import (  # noqa: E402
    DEFAULT_SPLIT_FILE,
    load_splits,
)


def _classification_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float):
    pred = (probs >= threshold).astype(np.int32)
    truth = (labels >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels, all_filenames = [], [], []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            labels = batch["label"].to(device)
            logits = model(points, return_logits=True)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_filenames.extend(list(batch["filename"]))
    return np.array(all_probs), np.array(all_labels), all_filenames


def main():
    parser = argparse.ArgumentParser(description="Evaluate risk predictor on TEST split")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "risk_predictor_v2.pth",
        help="Path to model checkpoint (default: models/risk_predictor_v2.pth)",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "combined_labels.csv",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.split_file.exists():
        raise FileNotFoundError(
            f"Patient split file not found: {args.split_file}\n"
            "Generate it first via: python -m training.scripts.patient_split --verify"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split file: {args.split_file}")

    splits = load_splits(args.split_file)
    test_indices = splits["test_indices"]
    print(f"Test samples: {len(test_indices)} from {len(splits['test_patients'])} patients")

    # Build dataset (no augmentation)
    base_dataset = LabeledArteryDataset(
        labels_csv=str(args.labels_csv),
        num_points=args.num_points,
        augment=False,
    )

    # Sanity: dataset row count must match split row count
    total_indices = len(splits["train_indices"]) + len(splits["val_indices"]) + len(splits["test_indices"])
    if total_indices != len(base_dataset.samples):
        raise RuntimeError(
            f"Dataset size ({len(base_dataset.samples)}) does not match split "
            f"index count ({total_indices}). Regenerate the split."
        )

    test_subset = Subset(base_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    model = RiskPredictorV2(latent_dim=args.latent_dim).to(device)
    state = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)

    probs, labels, filenames = run_inference(model, test_loader, device)
    truth = (labels >= 0.5).astype(np.int32)

    # Threshold=0.5 metrics
    base_metrics = _classification_metrics(probs, labels, threshold=0.5)
    cm = confusion_matrix(truth, (probs >= 0.5).astype(np.int32), labels=[0, 1])

    # ROC-AUC + PR-AUC
    try:
        roc_auc = float(roc_auc_score(truth, probs))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(truth, probs))
    except ValueError:
        pr_auc = float("nan")

    # Threshold sweep
    sweep_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep = [_classification_metrics(probs, labels, t) for t in sweep_thresholds]

    # Youden-optimal threshold (maximizes sensitivity + specificity - 1)
    fine_thresholds = np.linspace(0.01, 0.99, 99)
    youden_scores = []
    for t in fine_thresholds:
        m = _classification_metrics(probs, labels, t)
        youden_scores.append(m["recall"] + m["specificity"] - 1.0)
    best_idx = int(np.argmax(youden_scores))
    youden_threshold = float(fine_thresholds[best_idx])
    youden_metrics = _classification_metrics(probs, labels, youden_threshold)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Samples: {len(probs)}  (Healthy: {(truth == 0).sum()}, Diseased: {(truth == 1).sum()})")
    print(f"\nThreshold = 0.50:")
    print(f"  Confusion matrix [[TN FP][FN TP]]: {cm.tolist()}")
    print(f"  Accuracy   : {base_metrics['accuracy']:.4f}")
    print(f"  Precision  : {base_metrics['precision']:.4f}")
    print(f"  Recall     : {base_metrics['recall']:.4f}")
    print(f"  Specificity: {base_metrics['specificity']:.4f}")
    print(f"  F1         : {base_metrics['f1']:.4f}")
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    print("\nThreshold sweep:")
    print(f"{'thr':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
          f"{'acc':>7} {'rec':>7} {'spec':>7} {'F1':>7}")
    for m in sweep:
        print(
            f"{m['threshold']:>5.2f} "
            f"{m['TP']:>5d} {m['FP']:>5d} {m['FN']:>5d} {m['TN']:>5d} "
            f"{m['accuracy']:>7.4f} {m['recall']:>7.4f} "
            f"{m['specificity']:>7.4f} {m['f1']:>7.4f}"
        )

    print(f"\nYouden-optimal threshold: {youden_threshold:.3f}")
    print(
        f"  At Youden thr: acc={youden_metrics['accuracy']:.4f}, "
        f"rec={youden_metrics['recall']:.4f}, "
        f"spec={youden_metrics['specificity']:.4f}, "
        f"F1={youden_metrics['f1']:.4f}"
    )

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "filename": filenames,
        "label": truth,
        "prob": probs,
        "pred_at_0.5": (probs >= 0.5).astype(int),
    })
    csv_path = out_dir / f"test_evaluation_{timestamp}.csv"
    pred_df.to_csv(csv_path, index=False)

    summary = {
        "timestamp": timestamp,
        "checkpoint": str(args.checkpoint),
        "split_file": str(args.split_file),
        "n_samples": int(len(probs)),
        "n_healthy": int((truth == 0).sum()),
        "n_diseased": int((truth == 1).sum()),
        "metrics_at_0.5": base_metrics,
        "confusion_matrix_tn_fp_fn_tp": cm.tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_sweep": sweep,
        "youden_threshold": youden_threshold,
        "metrics_at_youden": youden_metrics,
    }
    json_path = out_dir / f"test_metrics_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-sample predictions: {csv_path}")
    print(f"Saved summary metrics       : {json_path}")


if __name__ == "__main__":
    main()
