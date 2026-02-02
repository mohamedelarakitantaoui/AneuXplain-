"""
Artery Risk Analysis - Data Loading Utilities

Dataset classes for loading 3D artery meshes as point clouds.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh
from typing import List, Optional, Dict, Any


class PointCloudDataset(Dataset):
    """
    Base dataset for loading point clouds from mesh files.
    
    Features:
    - Loads .obj, .ply, .stl mesh files
    - Samples points from mesh surface
    - Normalizes to unit sphere
    - Optional data augmentation
    """
    
    def __init__(
        self,
        file_list: List[str],
        num_points: int = 2048,
        augment: bool = False
    ):
        super().__init__()
        self.file_list = file_list
        self.num_points = num_points
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def load_and_sample(self, file_path: str) -> np.ndarray:
        """Load mesh and sample points from surface."""
        mesh = trimesh.load(file_path, force='mesh')
        result = trimesh.sample.sample_surface(mesh, count=self.num_points)
        points = np.array(result[0], dtype=np.float32)
        return self.normalize_points(points)
    
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere centered at origin."""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 0:
            points_centered = points_centered / max_dist
        return points_centered
    
    def apply_augmentation(self, points: np.ndarray) -> np.ndarray:
        """Apply random augmentations to point cloud."""
        if not self.augment:
            return points
        
        # Random rotation around Z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points = points @ rotation.T
        
        # Random jitter
        points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        points *= scale
        
        return points


class MultiSourceDataset(PointCloudDataset):
    """
    Dataset that loads point clouds from multiple folders.
    Used for UNSUPERVISED autoencoder training.
    """
    
    def __init__(
        self,
        folder_list: List[str],
        num_points: int = 2048,
        augment: bool = False
    ):
        # Collect all .obj files from all folders
        file_list = []
        for folder in folder_list:
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.obj"))
                file_list.extend(files)
                print(f"  Found {len(files)} files in {folder}")
            else:
                print(f"  WARNING: Folder not found: {folder}")
        
        super().__init__(file_list, num_points, augment)
        print(f"  Total files: {len(self.file_list)}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_list[idx]
        filename = os.path.basename(file_path)
        
        try:
            points = self.load_and_sample(file_path)
            points = self.apply_augmentation(points)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Convert to tensor: (N, 3) -> (3, N)
        points_tensor = torch.tensor(points, dtype=torch.float32).T
        
        return {
            'points': points_tensor,
            'filename': filename
        }


class LabeledArteryDataset(PointCloudDataset):
    """
    Dataset for SUPERVISED training with risk labels from CSV.
    """
    
    def __init__(
        self,
        labels_csv: str,
        num_points: int = 2048,
        augment: bool = False,
        base_path: Optional[str] = None
    ):
        self.labels_df = pd.read_csv(labels_csv)
        self.base_path = base_path
        
        # Build file list and labels
        self.samples = []
        missing_files = 0
        
        for _, row in self.labels_df.iterrows():
            # Handle different CSV formats
            if 'data_folder' in row:
                file_path = os.path.join(row['data_folder'], row['filename'])
            elif base_path:
                file_path = os.path.join(base_path, row['filename'])
            else:
                file_path = row['filename']
            
            if os.path.exists(file_path):
                score_col = 'curvature_score' if 'curvature_score' in row else 'risk_score'
                self.samples.append({
                    'file_path': file_path,
                    'filename': row['filename'],
                    'score': row[score_col],
                    'source': row.get('source', 'unknown')
                })
            else:
                missing_files += 1
        
        file_list = [s['file_path'] for s in self.samples]
        super().__init__(file_list, num_points, augment)
        
        aug_status = "ON" if augment else "OFF"
        print(f"[LabeledDataset] Loaded {len(self.samples)} samples (Augmentation: {aug_status})")
        if missing_files > 0:
            print(f"  WARNING: {missing_files} files not found!")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        try:
            points = self.load_and_sample(sample['file_path'])
            points = self.apply_augmentation(points)
        except Exception as e:
            print(f"Error loading {sample['file_path']}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        points_tensor = torch.tensor(points, dtype=torch.float32).T
        
        return {
            'points': points_tensor,
            'label': torch.tensor([sample['score']], dtype=torch.float32),
            'filename': sample['filename'],
            'source': sample['source']
        }
    
    def get_class_weights(self, threshold: float = 0.5) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = [1 if s['score'] >= threshold else 0 for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        total = len(labels)
        weights = [total / (2 * c) if c > 0 else 1.0 for c in class_counts]
        sample_weights = [weights[l] for l in labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


class BalancedArteryDataset(LabeledArteryDataset):
    """
    Dataset with automatic oversampling for class balance.
    """
    
    def __init__(
        self,
        labels_csv: str,
        num_points: int = 2048,
        augment: bool = False,
        base_path: Optional[str] = None,
        low_risk_threshold: float = 0.3,
        oversample_factor: int = 5
    ):
        super().__init__(labels_csv, num_points, augment, base_path)
        
        # Separate and oversample
        low_risk = [s for s in self.samples if s['score'] < low_risk_threshold]
        high_risk = [s for s in self.samples if s['score'] >= low_risk_threshold]
        
        print(f"  Original - Low Risk: {len(low_risk)}, High Risk: {len(high_risk)}")
        
        # Oversample minority class
        oversampled = low_risk * oversample_factor
        self.samples = oversampled + high_risk
        random.shuffle(self.samples)
        
        self.file_list = [s['file_path'] for s in self.samples]
        print(f"  Balanced - Low Risk: {len(oversampled)}, High Risk: {len(high_risk)}")
        print(f"  Total samples: {len(self.samples)}")
