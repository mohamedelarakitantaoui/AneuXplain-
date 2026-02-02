"""
Train Autoencoder - Full Dataset Training

Trains a PointNet Autoencoder on the complete dataset including:
- Complete arteries
- Generated healthy vessels
- Generated aneurysms

Usage:
    python -m training.scripts.train_autoencoder
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from training.models import Autoencoder, ChamferDistanceLoss, MultiSourceDataset


def train_autoencoder(config: dict):
    """Train the autoencoder with given configuration."""
    
    print("=" * 60)
    print("AUTOENCODER TRAINING")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data folders
    data_folders = [
        PROJECT_ROOT / "IntrA" / "complete",
        PROJECT_ROOT / "IntrA" / "generated" / "vessel" / "obj",
        PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj",
    ]
    data_folders = [str(f) for f in data_folders if f.exists()]
    
    print(f"\nLoading data from {len(data_folders)} folders...")
    dataset = MultiSourceDataset(
        folder_list=data_folders,
        num_points=config['num_points'],
        augment=True
    )
    
    # Train/Val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = Autoencoder(
        latent_dim=config['latent_dim'],
        num_points=config['num_points']
    ).to(device)
    
    criterion = ChamferDistanceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    checkpoints_dir = PROJECT_ROOT / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            points = batch['points'].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(points)
            loss = criterion(points, reconstructed)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                reconstructed = model(points)
                loss = criterion(points, reconstructed)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / "autoencoder_best.pth")
            status = "✓ SAVED"
        else:
            status = ""
        
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | {status}")
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"autoencoder_epoch{epoch}.pth")
    
    # Save final model
    final_path = PROJECT_ROOT / "training" / "saved_models" / "autoencoder.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Chamfer Distance')
    plt.title('Autoencoder Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(checkpoints_dir / "autoencoder_training.png", dpi=150)
    plt.close()
    
    return model, history


if __name__ == "__main__":
    config = {
        'num_points': 2048,
        'latent_dim': 128,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50
    }
    
    train_autoencoder(config)
