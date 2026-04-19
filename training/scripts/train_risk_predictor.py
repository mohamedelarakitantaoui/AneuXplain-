"""
Train Risk Predictor - Binary Classification with BCEWithLogitsLoss

Uses ground-truth binary labels (vessel=0, aneurysm=1) from the IntrA
dataset folder structure. The model outputs raw logits; sigmoid is
applied only at inference.

Splits are PATIENT-LEVEL (no fragment from the same AN<id> appears in
more than one of train/val/test) and locked to data/patient_splits.json.
Augmentation is applied only on the training subset via AugmentedSubset.

Usage:
    python -m training.scripts.train_risk_predictor
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import shutil

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from training.models import RiskPredictorV2, LabeledArteryDataset, AugmentedSubset
from training.scripts.patient_split import (
    DEFAULT_SPLIT_FILE,
    load_or_create_patient_splits,
)


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    pred = (probs >= threshold).astype(np.int32)
    truth = (labels >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return accuracy, sensitivity, specificity


def train_risk_predictor(config: dict):
    """Train the risk predictor with given configuration."""

    print("=" * 60)
    print("RISK PREDICTOR TRAINING (Binary - BCEWithLogitsLoss)")
    print("Patient-level train/val/test split")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    labels_csv = PROJECT_ROOT / "data" / "combined_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels not found: {labels_csv}\n"
            "Run  python -m training.scripts.prepare_labels  first."
        )

    print(f"\nLoading labels from: {labels_csv}")
    base_dataset = LabeledArteryDataset(
        labels_csv=str(labels_csv),
        num_points=config['num_points'],
        augment=False,  # always False on the base; AugmentedSubset wraps train
    )

    # Patient-level splits (load if locked, else compute and persist)
    labels_df = pd.read_csv(labels_csv)
    # Filter labels_df to only the rows that resolved to existing files,
    # so indices align with base_dataset.samples.
    sample_filenames = [s['filename'] for s in base_dataset.samples]
    labels_df = labels_df[labels_df['filename'].isin(set(sample_filenames))].reset_index(drop=True)
    if len(labels_df) != len(base_dataset.samples):
        raise RuntimeError(
            f"Row count mismatch after filtering: labels_df={len(labels_df)} "
            f"vs dataset={len(base_dataset.samples)}"
        )

    splits = load_or_create_patient_splits(labels_df, path=DEFAULT_SPLIT_FILE)
    print(
        f"Patient split (seed={splits['seed']}): "
        f"train={len(splits['train_patients'])}p/{len(splits['train_indices'])}s, "
        f"val={len(splits['val_patients'])}p/{len(splits['val_indices'])}s, "
        f"test={len(splits['test_patients'])}p/{len(splits['test_indices'])}s"
    )

    train_subset = AugmentedSubset(base_dataset, splits['train_indices'])
    val_subset = Subset(base_dataset, splits['val_indices'])
    test_subset = Subset(base_dataset, splits['test_indices'])

    # Weighted sampler for class imbalance (50/50 batches)
    train_labels = [base_dataset.samples[i]['score'] for i in splits['train_indices']]
    train_binary = [1 if s >= 0.5 else 0 for s in train_labels]

    class_counts = [train_binary.count(0), train_binary.count(1)]
    class_weights = [1.0 / c if c > 0 else 1.0 for c in class_counts]
    sample_weights = [class_weights[l] for l in train_binary]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
    print(f"Class distribution (train): Healthy={class_counts[0]}, Diseased={class_counts[1]}")

    # Model (V2: outputs logits, deeper MLP head)
    model = RiskPredictorV2(latent_dim=config['latent_dim']).to(device)

    # BCEWithLogitsLoss with pos_weight computed from actual class ratio
    num_healthy = class_counts[0]
    num_diseased = class_counts[1]
    pw = num_healthy / num_diseased if num_diseased > 0 else 5.0
    pos_weight = torch.tensor([pw]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Loss: BCEWithLogitsLoss (pos_weight={pw:.2f} = {num_healthy}/{num_diseased})")

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_gap': [], 'val_mae': [], 'val_auc': [],
        'val_sens': [], 'val_spec': [],
    }
    best_val_loss = float('inf')
    best_val_gap = -float('inf')
    best_val_auc = -float('inf')
    best_val_acc = -float('inf')
    best_epoch = 0
    patience = config.get('patience', 15)
    epochs_without_improvement = 0  # tracked on GAP

    checkpoints_dir = PROJECT_ROOT / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Early stopping: patience={patience}")

    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            points = batch['points'].to(device)
            labels = batch['label'].to(device)  # (B, 1) - 0.0 or 1.0

            optimizer.zero_grad()

            logits = model(points, return_logits=True)  # (B, 1)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                predicted_class = (probs >= 0.5).float()
                train_correct += (predicted_class == (labels >= 0.5).float()).sum().item()
                train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                labels = batch['label'].to(device)

                logits = model(points, return_logits=True)
                loss = criterion(logits, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                predicted_class = (probs >= 0.5).float()
                val_correct += (predicted_class == (labels >= 0.5).float()).sum().item()
                val_total += labels.size(0)

                all_preds.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Prediction gap (mean diseased score - mean healthy score)
        high_risk_preds = all_preds[all_labels >= 0.5]
        low_risk_preds = all_preds[all_labels < 0.5]
        gap = (np.mean(high_risk_preds) - np.mean(low_risk_preds)
               if len(high_risk_preds) > 0 and len(low_risk_preds) > 0 else 0)
        mae = np.mean(np.abs(all_preds - all_labels))

        # AUC-ROC (requires both classes present)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.0

        _, sens, spec = _binary_metrics(all_preds, all_labels, threshold=0.5)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_gap'].append(gap)
        history['val_mae'].append(mae)
        history['val_auc'].append(auc)
        history['val_sens'].append(sens)
        history['val_spec'].append(spec)

        scheduler.step()

        status_tags = []

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_loss.pth")
            status_tags.append("best_loss")

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_val_auc.pth")
            status_tags.append("best_auc")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_val_accuracy.pth")
            status_tags.append("best_acc")

        # Early stopping tracks GAP - retained for backwards compat
        if gap > best_val_gap:
            best_val_gap = gap
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_gap.pth")
            status_tags.append("best_gap")
        else:
            epochs_without_improvement += 1

        status = " ".join(status_tags)

        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {train_acc:.2%}/{val_acc:.2%} | "
            f"AUC: {auc:.3f} | Sens: {sens:.3f} | Spec: {spec:.3f} | "
            f"MAE: {mae:.4f} | Gap: {gap:+.3f} | {status}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} - no improvement for {patience} epochs")
            break

    # Save the final-epoch checkpoint
    torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_last.pth")

    # Deploy best-val-auc as the canonical checkpoint
    best_auc_path = checkpoints_dir / "risk_predictor_best_val_auc.pth"
    deploy_path = PROJECT_ROOT / "models" / "risk_predictor_v2.pth"
    if best_auc_path.exists():
        model.load_state_dict(torch.load(str(best_auc_path), map_location=device, weights_only=True))
        shutil.copy2(best_auc_path, deploy_path)
        print(f"\nBest-val-AUC checkpoint deployed to: {deploy_path}")

    # Save final model snapshot under training/saved_models/
    final_path = PROJECT_ROOT / "training" / "saved_models" / "risk_predictor_v2.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}")
    print(f"  Best epoch (gap):  {best_epoch}")
    print(f"  Best Val AUC:      {best_val_auc:.3f}")
    print(f"  Best Val Acc:      {best_val_acc:.2%}")
    print(f"  Best Val Gap:      {best_val_gap:+.3f}")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")

    # Plot training history
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy (threshold=0.5)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_auc'], color='red')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC-ROC')
    axes[2].set_title('Validation AUC-ROC'); axes[2].grid(True, alpha=0.3)

    axes[3].plot(history['val_sens'], label='Sensitivity')
    axes[3].plot(history['val_spec'], label='Specificity')
    axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('Rate')
    axes[3].set_title('Val Sens/Spec'); axes[3].legend(); axes[3].grid(True, alpha=0.3)

    axes[4].plot(history['val_gap'], color='purple')
    axes[4].axhline(y=0.7, color='k', linestyle='--', alpha=0.5, label='Target')
    axes[4].set_xlabel('Epoch'); axes[4].set_ylabel('Gap (Diseased - Healthy)')
    axes[4].set_title('Prediction Gap'); axes[4].legend(); axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(checkpoints_dir / "risk_training_v2.png", dpi=150)
    plt.close()

    return model, history


if __name__ == "__main__":
    config = {
        'num_points': 2048,
        'latent_dim': 128,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'epochs': 80,
        'patience': 15,
    }

    train_risk_predictor(config)
