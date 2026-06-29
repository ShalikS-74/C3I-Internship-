"""Datasets for synthetic satellite imagery generation.

This module provides PyTorch datasets for loading paired (RGB, mask) data
from the real satellite dataset for GAN training.

Classes:
    PairedSatelliteDataset: Base dataset for image-mask pairs.
    PairedSatelliteDatasetWithAugmentation: Dataset with augmentation.

Functions:
    create_dataloader: Factory function for creating DataLoaders.
    find_image_mask_pairs: Utility for discovering paired files.
"""

from .paired_dataset import (
    PairedSatelliteDataset,
    PairedSatelliteDatasetWithAugmentation,
    create_dataloader,
    find_image_mask_pairs,
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
)

__all__ = [
    "PairedSatelliteDataset",
    "PairedSatelliteDatasetWithAugmentation",
    "create_dataloader",
    "find_image_mask_pairs",
    "IMAGE_EXTENSIONS",
    "MASK_EXTENSIONS",
]