"""
Train Conditional VAE for Risk-Conditioned Artery Generation

Trains a ConditionalVAE that learns to reconstruct arteries conditioned
on their risk score.  After training, calling generate_healthy(x, 0.1)
produces a low-risk version of input x in a single forward pass.

Loss = Chamfer(recon, x) + beta * KL(q(z|x) || N(0,I))
       + gamma * RiskConsistency(recon, target_risk)

The risk predictor is FROZEN — it acts as a learned evaluation function.

Usage:
    python -m training.scripts.train_cvae
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt

from training.models import (
    ConditionalVAE,
    RiskPredictorV2,
    ChamferDistanceLoss,
    LabeledArteryDataset,
)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over the batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_cvae(config: dict):
    """Train the Conditional VAE."""

    print("=" * 60)
    print("CONDITIONAL VAE TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset (same labeled dataset used for risk predictor)
    # ------------------------------------------------------------------
    labels_csv = PROJECT_ROOT / "data" / "combined_labels.csv"
    if not labels_csv.exists():
        labels_csv = PROJECT_ROOT / "data" / "curvature_labels.csv"

    print(f"\nLoading labels from: {labels_csv}")
    dataset = LabeledArteryDataset(
        labels_csv=str(labels_csv),
        num_points=config["num_points"],
        augment=True,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Weighted sampler for class balance
    train_indices = train_dataset.indices
    train_labels = [dataset.samples[i]["score"] for i in train_indices]
    train_binary = [1 if s >= 0.5 else 0 for s in train_labels]
    class_counts = [train_binary.count(0), train_binary.count(1)]
    class_weights = [1.0 / c if c > 0 else 1.0 for c in class_counts]
    sample_weights = [class_weights[l] for l in train_binary]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Class distribution: Low={class_counts[0]}, High={class_counts[1]}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    cvae = ConditionalVAE(
        latent_dim=config["latent_dim"], num_points=config["num_points"]
    ).to(device)

    # Frozen risk predictor for consistency loss
    risk_predictor = RiskPredictorV2(latent_dim=config["latent_dim"]).to(device)
    rp_path = PROJECT_ROOT / "models" / "risk_predictor_v2.pth"
    if not rp_path.exists():
        raise FileNotFoundError(
            f"Risk predictor weights not found: {rp_path}\n"
            "Train the risk predictor first."
        )
    risk_predictor.load_state_dict(
        torch.load(str(rp_path), map_location=device, weights_only=True)
    )
    risk_predictor.eval()
    for p in risk_predictor.parameters():
        p.requires_grad = False
    print(f"Frozen risk predictor loaded from {rp_path.name}")

    chamfer = ChamferDistanceLoss()
    optimizer = optim.Adam(cvae.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-5
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_chamfer": [],
        "val_chamfer": [],
        "train_kl": [],
        "val_kl": [],
        "train_risk_cons": [],
        "val_risk_cons": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience = config.get("patience", 10)
    epochs_without_improvement = 0

    checkpoints_dir = PROJECT_ROOT / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    beta_max = config.get("beta_max", 0.01)
    beta_warmup = config.get("beta_warmup", 20)
    gamma = config.get("gamma", 0.5)

    print(f"\nHyperparams: beta_max={beta_max}, beta_warmup={beta_warmup}, gamma={gamma}")
    print(f"Epochs: {config['epochs']}, LR: {config['learning_rate']}")
    print(f"Early stopping: patience={patience}")
    print()

    for epoch in range(1, config["epochs"] + 1):
        # Beta annealing: 0 -> beta_max over warmup epochs
        if epoch <= beta_warmup:
            beta = beta_max * (epoch / beta_warmup)
        else:
            beta = beta_max

        # --- Train ---
        cvae.train()
        t_loss = t_ch = t_kl = t_rc = 0.0

        for batch in train_loader:
            points = batch["points"].to(device)
            labels = batch["label"].to(device).squeeze(-1)  # (B,)

            optimizer.zero_grad()

            recon, mu, logvar = cvae(points, labels)

            loss_chamfer = chamfer(recon, points)
            loss_kl = kl_divergence(mu, logvar)

            # Risk consistency: recon should have same risk as the label.
            # Gradients flow through recon to the CVAE; risk predictor
            # weights are frozen so they act as a fixed evaluation function.
            recon_risk = torch.sigmoid(risk_predictor(recon, return_logits=True)).squeeze(-1)
            loss_risk_cons = nn.functional.mse_loss(recon_risk, labels)

            loss = loss_chamfer + beta * loss_kl + gamma * loss_risk_cons
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            optimizer.step()

            t_loss += loss.item()
            t_ch += loss_chamfer.item()
            t_kl += loss_kl.item()
            t_rc += loss_risk_cons.item()

        n = len(train_loader)
        t_loss /= n; t_ch /= n; t_kl /= n; t_rc /= n

        # --- Val ---
        cvae.eval()
        v_loss = v_ch = v_kl = v_rc = 0.0

        with torch.no_grad():
            for batch in val_loader:
                points = batch["points"].to(device)
                labels = batch["label"].to(device).squeeze(-1)

                recon, mu, logvar = cvae(points, labels)

                loss_chamfer = chamfer(recon, points)
                loss_kl = kl_divergence(mu, logvar)
                recon_risk = torch.sigmoid(risk_predictor(recon, return_logits=True)).squeeze(-1)
                loss_risk_cons = nn.functional.mse_loss(recon_risk, labels)

                loss = loss_chamfer + beta * loss_kl + gamma * loss_risk_cons

                v_loss += loss.item()
                v_ch += loss_chamfer.item()
                v_kl += loss_kl.item()
                v_rc += loss_risk_cons.item()

        nv = len(val_loader)
        v_loss /= nv; v_ch /= nv; v_kl /= nv; v_rc /= nv

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_chamfer"].append(t_ch)
        history["val_chamfer"].append(v_ch)
        history["train_kl"].append(t_kl)
        history["val_kl"].append(v_kl)
        history["train_risk_cons"].append(t_rc)
        history["val_risk_cons"].append(v_rc)

        scheduler.step()

        # Save best + early stopping
        status = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(cvae.state_dict(), checkpoints_dir / "cvae_best.pth")
            status = "BEST"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Loss: {t_loss:.4f}/{v_loss:.4f} | "
            f"Chamfer: {t_ch:.5f}/{v_ch:.5f} | "
            f"KL: {t_kl:.3f}/{v_kl:.3f} | "
            f"RiskCons: {t_rc:.4f}/{v_rc:.4f} | "
            f"beta={beta:.5f} | {status}"
        )

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} epochs")
            break

    # ------------------------------------------------------------------
    # Load best checkpoint, save & deploy
    # ------------------------------------------------------------------
    best_path = checkpoints_dir / "cvae_best.pth"
    if best_path.exists():
        cvae.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=True))

    final_path = PROJECT_ROOT / "training" / "saved_models" / "cvae.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cvae.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    deploy_path = PROJECT_ROOT / "models" / "cvae.pth"
    if best_path.exists():
        shutil.copy2(best_path, deploy_path)
        print(f"Best model deployed to: {deploy_path}")

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_chamfer"], label="Train")
    axes[1].plot(history["val_chamfer"], label="Val")
    axes[1].set_title("Chamfer Distance")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["train_kl"], label="Train")
    axes[2].plot(history["val_kl"], label="Val")
    axes[2].set_title("KL Divergence")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    axes[3].plot(history["train_risk_cons"], label="Train")
    axes[3].plot(history["val_risk_cons"], label="Val")
    axes[3].set_title("Risk Consistency (MSE)")
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(checkpoints_dir / "cvae_training.png", dpi=150)
    plt.close()
    print(f"Training plot saved: {checkpoints_dir / 'cvae_training.png'}")

    return cvae, history


if __name__ == "__main__":
    config = {
        "num_points": 2048,
        "latent_dim": 128,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "epochs": 70,
        # CVAE-specific
        "beta_max": 0.01,      # KL weight (annealed from 0)
        "beta_warmup": 20,     # Epochs to anneal beta from 0 -> beta_max
        "gamma": 0.5,          # Risk consistency weight
        "patience": 10,
    }

    train_cvae(config)
