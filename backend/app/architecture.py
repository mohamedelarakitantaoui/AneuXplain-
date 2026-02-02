"""
architecture.py - Neural Network Architectures for Artery Analysis

Contains the PointNet-based Autoencoder and Risk Predictor models
for 3D artery point cloud analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    PointNet-style Encoder for point cloud compression.
    
    Compresses a point cloud of 2048 points into a compact latent vector.
    Uses Conv1d layers (shared MLP) followed by global max pooling.
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
        
        # Fully connected layer to latent space
        self.fc = nn.Linear(1024, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a point cloud to a latent vector.
        
        Args:
            x: Input point cloud, shape (B, 3, N) where N=2048
            
        Returns:
            Latent vector, shape (B, latent_dim)
        """
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global Max Pooling: (B, 1024, N) -> (B, 1024)
        x = torch.max(x, dim=2)[0]
        
        # Project to latent space
        latent = self.fc(x)
        
        return latent


class Decoder(nn.Module):
    """
    Decoder for reconstructing point clouds from latent vectors.
    
    Takes the compressed latent representation and expands it
    back to a full point cloud using fully connected layers.
    """
    
    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.output_size = num_points * 3
        
        # Fully connected layers for expansion
        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc4 = nn.Linear(1024, self.output_size)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector to a point cloud.
        
        Args:
            z: Latent vector, shape (B, latent_dim)
            
        Returns:
            Reconstructed point cloud, shape (B, 3, num_points)
        """
        batch_size = z.shape[0]
        
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        
        # Reshape to point cloud format: (B, 6144) -> (B, 3, 2048)
        x = x.view(batch_size, 3, self.num_points)
        
        return x


class Autoencoder(nn.Module):
    """
    Complete PointNet Autoencoder combining Encoder and Decoder.
    
    Learns to encode and decode 3D artery point clouds through
    a compressed latent representation.
    """
    
    def __init__(self, latent_dim: int = 128, num_points: int = 2048):
        super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, num_points=num_points)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a point cloud to its latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to a point cloud."""
        return self.decoder(z)


class RiskPredictor(nn.Module):
    """
    Risk Prediction Model: PointNet Encoder + MLP Head.
    
    Takes a 3D artery point cloud and outputs a risk score (0-1).
    Uses its own encoder (not shared with autoencoder) for end-to-end learning.
    """
    
    def __init__(self, latent_dim: int = 128):
        super(RiskPredictor, self).__init__()
        
        self.encoder = Encoder(latent_dim=latent_dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict risk score from point cloud.
        
        Args:
            x: Input point cloud, shape (B, 3, N)
            
        Returns:
            Risk score, shape (B, 1) in range [0, 1]
        """
        latent = self.encoder(x)
        risk = self.mlp_head(latent)
        return risk


class ChamferDistanceLoss(nn.Module):
    """
    Chamfer Distance Loss for comparing two point clouds.
    
    Permutation-invariant loss that finds nearest neighbors
    between two point sets.
    """
    
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Chamfer Distance between two point clouds.
        
        Args:
            x: First point cloud, shape (B, 3, N) or (B, N, 3)
            y: Second point cloud, shape (B, 3, M) or (B, M, 3)
            
        Returns:
            Chamfer distance loss (scalar tensor)
        """
        # Ensure points are in (B, N, 3) format
        if x.shape[1] == 3 and x.shape[2] != 3:
            x = x.transpose(1, 2)
        if y.shape[1] == 3 and y.shape[2] != 3:
            y = y.transpose(1, 2)
        
        # Compute pairwise distances
        x_expanded = x.unsqueeze(2)  # (B, N, 1, 3)
        y_expanded = y.unsqueeze(1)  # (B, 1, M, 3)
        
        diff = x_expanded - y_expanded  # (B, N, M, 3)
        distances = torch.sum(diff ** 2, dim=-1)  # (B, N, M)
        
        # Find minimum distances
        min_dist_x_to_y, _ = torch.min(distances, dim=2)  # (B, N)
        min_dist_y_to_x, _ = torch.min(distances, dim=1)  # (B, M)
        
        # Compute Chamfer Distance
        chamfer_x = torch.mean(min_dist_x_to_y, dim=1)
        chamfer_y = torch.mean(min_dist_y_to_x, dim=1)
        chamfer_loss = torch.mean(chamfer_x + chamfer_y)
        
        return chamfer_loss
