"""
Train Risk Predictor - End-to-End Training

Trains a Risk Predictor from scratch (no pre-trained encoder).
This ensures the encoder learns disease-specific features.

Usage:
    python -m training.scripts.train_risk_predictor
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np

from training.models import RiskPredictor, LabeledArteryDataset


def train_risk_predictor(config: dict):
    """Train the risk predictor with given configuration."""
    
    print("=" * 60)
    print("RISK PREDICTOR TRAINING (End-to-End)")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset
    labels_csv = PROJECT_ROOT / "data" / "combined_labels.csv"
    if not labels_csv.exists():
        labels_csv = PROJECT_ROOT / "data" / "curvature_labels.csv"
    
    print(f"\nLoading labels from: {labels_csv}")
    dataset = LabeledArteryDataset(
        labels_csv=str(labels_csv),
        num_points=config['num_points'],
        augment=True
    )
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Weighted sampler for class imbalance
    train_indices = train_dataset.indices
    train_labels = [dataset.samples[i]['score'] for i in train_indices]
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
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Class distribution: Low={class_counts[0]}, High={class_counts[1]}")
    
    # Model
    model = RiskPredictor(latent_dim=config['latent_dim']).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_gap': []
    }
    best_val_gap = -float('inf')
    best_val_loss = float('inf')
    
    checkpoints_dir = PROJECT_ROOT / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            points = batch['points'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            predictions = model(points)
            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted_class = (predictions >= 0.5).float()
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
                
                predictions = model(points)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted_class = (predictions >= 0.5).float()
                val_correct += (predicted_class == (labels >= 0.5).float()).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate gap (high risk avg - low risk avg)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        high_risk_preds = all_preds[all_labels >= 0.5]
        low_risk_preds = all_preds[all_labels < 0.5]
        gap = np.mean(high_risk_preds) - np.mean(low_risk_preds) if len(high_risk_preds) > 0 and len(low_risk_preds) > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_gap'].append(gap)
        
        scheduler.step()
        
        # Save best by gap
        status = ""
        if gap > best_val_gap:
            best_val_gap = gap
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_gap.pth")
            status = "✓ BEST GAP"
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / "risk_predictor_best_loss.pth")
            status = status or "✓ BEST LOSS"
        
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.2%}/{val_acc:.2%} | "
              f"Gap: {gap:+.3f} | {status}")
    
    # Save final model
    final_path = PROJECT_ROOT / "training" / "saved_models" / "risk_predictor.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_gap'], color='purple')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Gap (High - Low)')
    axes[2].set_title('Prediction Gap')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(checkpoints_dir / "risk_training.png", dpi=150)
    plt.close()
    
    return model, history


if __name__ == "__main__":
    config = {
        'num_points': 2048,
        'latent_dim': 128,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'epochs': 100
    }
    
    train_risk_predictor(config)
