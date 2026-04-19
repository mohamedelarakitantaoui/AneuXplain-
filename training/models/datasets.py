"""
Artery Risk Analysis - Data Loading Utilities

Dataset classes for loading 3D artery meshes as point clouds.

Augmentation policy
-------------------
For supervised training (LabeledArteryDataset), the base dataset NEVER
augments inside ``__getitem__``. Augmentation is applied through the
``AugmentedSubset`` wrapper, which is meant to be used on the *training*
subset only. Validation and test subsets should use a plain
``torch.utils.data.Subset``, so they always see raw (non-augmented) samples.

The ``MultiSourceDataset`` (used for unsupervised autoencoder training)
keeps its own internal augmentation flow because there is no train/val
split there.
"""

import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh
from typing import List, Optional, Dict, Any, Sequence


class PointCloudDataset(Dataset):
    """
    Base dataset for loading point clouds from mesh files.

    Features:
    - Loads .obj, .ply, .stl mesh files
    - Samples points from mesh surface
    - Normalizes to unit sphere
    - Optional data augmentation (driven by self.augment flag; subclasses
      may opt out and rely on ``AugmentedSubset`` instead)
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
        """Apply random augmentations to point cloud (only when self.augment)."""
        if not self.augment:
            return points
        return augment_points(points)


def augment_points(points: np.ndarray) -> np.ndarray:
    """
    Apply the standard IntrA point-cloud augmentations:
      * random rotation around the Z axis
      * Gaussian jitter (sigma=0.01)
      * uniform scaling in [0.9, 1.1]

    Operates on (N, 3) numpy arrays and returns the same shape.
    """
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
    points = points + np.random.normal(0, 0.01, points.shape).astype(np.float32)

    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    points = points * scale

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

    Always returns RAW (non-augmented) samples. Augmentation must be
    applied via ``AugmentedSubset`` on the training split.
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
                score_col = 'risk_score' if 'risk_score' in row else 'curvature_score'
                self.samples.append({
                    'file_path': file_path,
                    'filename': row['filename'],
                    'score': row[score_col],
                    'source': row.get('source', 'unknown')
                })
            else:
                missing_files += 1

        file_list = [s['file_path'] for s in self.samples]
        # NOTE: we deliberately force augment=False on the base dataset.
        # The `augment` argument is accepted for backward compatibility
        # but is now a no-op; use AugmentedSubset on the train split.
        if augment:
            warnings.warn(
                "LabeledArteryDataset(augment=True) is ignored. "
                "Wrap the training Subset with AugmentedSubset instead.",
                stacklevel=2,
            )
        super().__init__(file_list, num_points, augment=False)

        print(
            f"[LabeledDataset] Loaded {len(self.samples)} samples "
            "(Augmentation: handled by AugmentedSubset on training split)"
        )
        if missing_files > 0:
            print(f"  WARNING: {missing_files} files not found!")

    def load_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Load a raw (non-augmented) sample as numpy points + metadata."""
        sample = self.samples[idx]
        points = self.load_and_sample(sample['file_path'])  # (N, 3) np.float32
        return {
            'points_np': points,
            'label': float(sample['score']),
            'filename': sample['filename'],
            'source': sample['source'],
        }

    @staticmethod
    def to_tensor_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Pack a raw dict into the tensor dict consumed by the model."""
        points_tensor = torch.tensor(raw['points_np'], dtype=torch.float32).T
        return {
            'points': points_tensor,
            'label': torch.tensor([raw['label']], dtype=torch.float32),
            'filename': raw['filename'],
            'source': raw['source'],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            raw = self.load_raw_sample(idx)
        except Exception as e:  # noqa: BLE001
            print(f"Error loading {self.samples[idx]['file_path']}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        return self.to_tensor_dict(raw)

    def get_class_weights(self, threshold: float = 0.5) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = [1 if s['score'] >= threshold else 0 for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        total = len(labels)
        weights = [total / (2 * c) if c > 0 else 1.0 for c in class_counts]
        sample_weights = [weights[l] for l in labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


class AugmentedSubset(Dataset):
    """
    Wraps a ``LabeledArteryDataset`` (or any dataset exposing
    ``load_raw_sample`` + ``to_tensor_dict``) and applies augmentation in
    ``__getitem__``. Intended for the TRAINING subset only.
    """

    def __init__(
        self,
        base: LabeledArteryDataset,
        indices: Sequence[int],
    ):
        if not hasattr(base, "load_raw_sample") or not hasattr(base, "to_tensor_dict"):
            raise TypeError(
                "AugmentedSubset requires a base dataset that exposes "
                "load_raw_sample() and to_tensor_dict()."
            )
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        try:
            raw = self.base.load_raw_sample(idx)
        except Exception as e:  # noqa: BLE001
            print(f"Error loading index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        raw['points_np'] = augment_points(raw['points_np'])
        return self.base.to_tensor_dict(raw)


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
