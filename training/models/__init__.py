"""
Training Models Package

Shared architectures and datasets for training.
"""

from .architectures import (
    Encoder,
    Decoder,
    Autoencoder,
    ConditionalVAE,
    RiskPredictor,
    RiskPredictorV2,
    ChamferDistanceLoss
)

from .datasets import (
    PointCloudDataset,
    MultiSourceDataset,
    LabeledArteryDataset,
    BalancedArteryDataset,
    AugmentedSubset,
    augment_points,
)

__all__ = [
    # Architectures
    "Encoder",
    "Decoder",
    "Autoencoder",
    "ConditionalVAE",
    "RiskPredictor",
    "RiskPredictorV2",
    "ChamferDistanceLoss",
    # Datasets
    "PointCloudDataset",
    "MultiSourceDataset",
    "LabeledArteryDataset",
    "BalancedArteryDataset",
    "AugmentedSubset",
    "augment_points",
]
