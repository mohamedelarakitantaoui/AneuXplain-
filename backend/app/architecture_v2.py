"""
architecture_v2.py - Improved Neural Network Architectures for Artery Analysis

V2 Changes:
- RiskPredictorV2 with BatchNorm in MLP head
- Outputs LOGITS (use return_logits=False for probabilities)
- Better weight initialization
- Designed for continuous risk score prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    PointNet-style Encoder for point cloud compression.
    Compresses a point cloud of 2048 points into a compact latent vector.
    """

    def __init__(self, latent_dim: int = 128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional layers (shared MLP per point)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, dim=2)[0]
        latent = self.fc(x)
        return latent


class RiskPredictorV2(nn.Module):
    """
    Improved Risk Prediction Model for Continuous Risk Scores.

    Key changes from V1:
    1. BatchNorm after every Linear layer (stabilizes training)
    2. Outputs LOGITS by default, apply sigmoid for probabilities
    3. Deeper MLP head with proper regularization
    4. Designed for continuous risk ∈ [0, 1], not binary classification
    """

    def __init__(self, latent_dim: int = 128, dropout_rate: float = 0.3):
        super(RiskPredictorV2, self).__init__()

        self.encoder = Encoder(latent_dim=latent_dim)

        # Deeper MLP head with BatchNorm and Dropout
        self.mlp_head = nn.Sequential(
            # Layer 1: latent_dim -> 64
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3: 32 -> 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),

            # Output: 16 -> 1 (raw logits, no sigmoid)
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input point cloud, shape (B, 3, N)
            return_logits: If True, return raw logits. If False, return probabilities.

        Returns:
            Risk score logits (B, 1) or probabilities (B, 1)
        """
        latent = self.encoder(x)
        logits = self.mlp_head(latent)

        if return_logits:
            return logits

        return torch.sigmoid(logits)

    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        """Get calibrated risk probability in [0, 1]."""
        return self.forward(x, return_logits=False)


class Decoder(nn.Module):
    """Decoder for reconstructing point clouds from latent vectors."""

    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.output_size = num_points * 3

        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, self.output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        x = x.view(batch_size, 3, self.num_points)
        return x


class Autoencoder(nn.Module):
    """Complete PointNet Autoencoder combining Encoder and Decoder."""

    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, num_points=num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for risk-conditioned artery generation.

    Encoder  → (mu, logvar)  in R^latent_dim
    Decoder  ← (z, risk)     from R^(latent_dim+1)

    generate_healthy(x, target_risk) does healing in one forward pass.
    """

    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.encoder = Encoder(latent_dim=latent_dim * 2)
        self.decoder = Decoder(latent_dim=latent_dim + 1, num_points=num_points)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, risk_label: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        risk = risk_label.view(-1, 1)
        z_cond = torch.cat([z, risk], dim=1)
        recon = self.decoder(z_cond)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def decode(self, z: torch.Tensor, risk_label: torch.Tensor) -> torch.Tensor:
        risk = risk_label.view(-1, 1)
        z_cond = torch.cat([z, risk], dim=1)
        return self.decoder(z_cond)

    def generate_healthy(self, x: torch.Tensor, target_risk: float = 0.1) -> torch.Tensor:
        """One-pass healing: encode → mean → decode at target risk."""
        mu, _ = self.encode(x)
        target = torch.full((mu.shape[0], 1), target_risk, device=mu.device)
        z_cond = torch.cat([mu, target], dim=1)
        return self.decoder(z_cond)


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss for point cloud comparison."""

    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure points are in (B, N, 3) format
        if x.shape[1] == 3 and x.shape[2] != 3:
            x = x.transpose(1, 2)
        if y.shape[1] == 3 and y.shape[2] != 3:
            y = y.transpose(1, 2)

        # Compute pairwise distances
        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        diff = x_expanded - y_expanded
        distances = torch.sum(diff ** 2, dim=-1)

        # Find minimum distances
        min_dist_x_to_y, _ = torch.min(distances, dim=2)
        min_dist_y_to_x, _ = torch.min(distances, dim=1)

        # Compute Chamfer Distance
        chamfer_x = torch.mean(min_dist_x_to_y, dim=1)
        chamfer_y = torch.mean(min_dist_y_to_x, dim=1)
        chamfer_loss = torch.mean(chamfer_x + chamfer_y)

        return chamfer_loss
