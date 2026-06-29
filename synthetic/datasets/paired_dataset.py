"""Paired dataset for loading satellite RGB images and building masks.

This module provides a PyTorch Dataset for loading paired (image, mask) data
from the real satellite dataset. It is designed to align with the existing
dataset.py pipeline to ensure consistent preprocessing between GAN training
and segmentation training.

IMPORTANT: This module reuses preprocessing from the existing dataset.py
to avoid pipeline divergence. Images are normalized to [0, 1] range.

Tensor shapes:
    Input (file paths):
        image_path: str
        mask_path: str
    
    Output (after transforms):
        image: [3, H, W] where H=W=512, float32 in [0, 1]
        mask: [1, H, W] where H=W=512, float32 in {0, 1}

Usage:
    >>> from synthetic.config import get_default_config
    >>> from synthetic.datasets.paired_dataset import PairedSatelliteDataset
    >>> 
    >>> config = get_default_config()
    >>> dataset = PairedSatelliteDataset(config)
    >>> image, mask = dataset[0]
    >>> print(image.shape, mask.shape)
    torch.Size([3, 512, 512]) torch.Size([1, 512, 512])
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any, Sequence
import random as python_random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import SyntheticConfig, PathConfig

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Aligned with existing dataset.py
# =============================================================================

# Support multiple image extensions (same as dataset.py)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# =============================================================================
# HELPER FUNCTIONS - Reuse patterns from dataset.py
# =============================================================================

def _iter_files(
    folder: Path, 
    extensions: Sequence[str], 
    recursive: bool = True
) -> List[Path]:
    """Iterate over files with given extensions in directory.

    Args:
        folder: Directory to search.
        extensions: Allowed file extensions.
        recursive: Whether to search recursively.

    Returns:
        Sorted list of matching file paths.
    """
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files)

def _load_rgb_image(path: Path, image_size: int | None = None) -> np.ndarray:
    """Load and preprocess an RGB image.

    This function aligns with dataset.py to ensure consistent preprocessing.
    Normalizes to [0, 1] range.

    Args:
        path: Path to image file.
        image_size: Target size (H = W). If None, no resize.

    Returns:
        Image array [H, W, 3] float32 in [0, 1].

    Raises:
        ValueError: If image cannot be read or has unsupported format.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    # Handle different image formats
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] >= 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")

    # Resize if needed
    if image_size is not None:
        image = cv2.resize(
            image, 
            (image_size, image_size), 
            interpolation=cv2.INTER_AREA
        )

    # Normalize to [0, 1] - same as dataset.py
    original_dtype = image.dtype
    image = image.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        image /= float(np.iinfo(original_dtype).max)
    elif image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)

def _load_binary_mask(path: Path, image_size: int | None = None) -> np.ndarray:
    """Load and preprocess a binary mask.

    This function aligns with dataset.py to ensure consistent preprocessing.

    Args:
        path: Path to mask file.
        image_size: Target size (H = W). If None, no resize.

    Returns:
        Mask array [H, W] float32 in {0, 1}.

    Raises:
        ValueError: If mask cannot be read.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    # Resize if needed - use nearest to preserve binary values
    if image_size is not None:
        mask = cv2.resize(
            mask, 
            (image_size, image_size), 
            interpolation=cv2.INTER_NEAREST
        )

    return (mask > 0).astype(np.float32)

def find_image_mask_pairs(
    image_dir: Path,
    mask_dir: Path,
    image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
    mask_extensions: Sequence[str] = MASK_EXTENSIONS,
    recursive: bool = True,
) -> List[Tuple[Path, Path]]:
    """Match image and mask files by filename stem.

    This function aligns with dataset.py::find_image_mask_pairs.

    Args:
        image_dir: Directory containing images.
        mask_dir: Directory containing masks.
        image_extensions: Allowed image extensions.
        mask_extensions: Allowed mask extensions.
        recursive: Whether to search recursively.

    Returns:
        List of (image_path, mask_path) tuples.

    Raises:
        FileNotFoundError: If directories don't exist.
        ValueError: If no matching pairs found.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    # Build mapping of mask stems to paths
    masks_by_stem: Dict[str, Path] = {}
    for mask_path in _iter_files(mask_dir, mask_extensions, recursive):
        masks_by_stem.setdefault(mask_path.stem, mask_path)

    # Find matching pairs
    pairs: List[Tuple[Path, Path]] = []
    for image_path in _iter_files(image_dir, image_extensions, recursive):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise ValueError(
            "No matching image/mask pairs found. Files must share the same stem, "
            f"for example image_001.tif and image_001.png. "
            f"Image dir: {image_dir}, Mask dir: {mask_dir}"
        )

    return pairs

# =============================================================================
# DATASET CLASS
# =============================================================================

class PairedSatelliteDataset(Dataset):
    """Dataset for paired satellite RGB images and building masks.

    Loads image-mask pairs from the real satellite dataset for training
    the StyleGAN-inspired generator. Preprocessing is aligned with the existing
    dataset.py to avoid pipeline divergence.

    Images are normalized to [0, 1] range (same as segmentation pipeline).
    For GAN training, you may want to convert to [-1, 1] in the training loop.

    Attributes:
        config: SyntheticConfig instance.
        image_dir: Directory containing RGB images.
        mask_dir: Directory containing building masks.
        image_size: Target image size (H = W).
        transform: Optional additional transforms.
        pairs: List of (image_path, mask_path) tuples.
        rng: Random number generator for reproducible augmentation.

    Example:
        >>> config = get_default_config()
        >>> dataset = PairedSatelliteDataset(config)
        >>> print(len(dataset))
        4205
        >>> image, mask = dataset[0]
        >>> image.shape
        torch.Size([3, 512, 512])
        >>> image.min(), image.max()
        (0.0, 1.0)
    """

    def __init__(
        self,
        config: Optional[SyntheticConfig] = None,
        path_config: Optional[PathConfig] = None,
        image_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        image_dir: Optional[Path] = None,
        mask_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        recursive: bool = True,
    ) -> None:
        """Initialize the paired dataset.

        Args:
            config: SyntheticConfig instance. If provided, uses config values.
            path_config: Optional separate PathConfig. Overrides config.paths.
            image_size: Override image size from config.
            transform: Optional additional transforms to apply.
            image_dir: Override image directory from config.
            mask_dir: Override mask directory from config.
            seed: Random seed for reproducible augmentation. Defaults to config.training.seed.
            recursive: Whether to search directories recursively.

        Raises:
            FileNotFoundError: If image or mask directories don't exist.
            ValueError: If no valid image-mask pairs found.
        """
        # Resolve configuration
        if config is None:
            config = SyntheticConfig()

        self.config = config
        self.path_config = path_config if path_config else config.paths

        # Resolve directories
        if image_dir is not None:
            self.image_dir = Path(image_dir)
        else:
            self.image_dir = self.path_config.get_real_image_dir()

        if mask_dir is not None:
            self.mask_dir = Path(mask_dir)
        else:
            self.mask_dir = self.path_config.get_real_mask_dir()

        # Configuration
        self.image_size = image_size if image_size else config.model.image_size
        self.transform = transform
        self.recursive = recursive

        # Initialize RNG with seed for reproducible augmentation
        seed = seed if seed is not None else config.training.seed
        self.rng = python_random.Random(seed)

        # Discover and validate pairs
        self.pairs = find_image_mask_pairs(
            self.image_dir,
            self.mask_dir,
            recursive=recursive,
        )

        logger.info(
            f"Initialized PairedSatelliteDataset with {len(self.pairs)} pairs. "
            f"Image dir: {self.image_dir}, Mask dir: {self.mask_dir}"
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of image-mask pairs.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single image-mask pair.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image, mask) tensors.
            image: [3, H, W] float32 in [0, 1]
            mask: [1, H, W] float32 in {0, 1}

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        image_path, mask_path = self.pairs[idx]

        # Load image and mask using aligned preprocessing
        image = _load_rgb_image(image_path, self.image_size)
        mask = _load_binary_mask(mask_path, self.image_size)

        # Verify dimensions match before proceeding
        if image.shape[:2] != mask.shape[:2]:
            logger.warning(
                f"Dimension mismatch for sample {idx}: "
                f"image {image.shape[:2]} vs mask {mask.shape[:2]}. "
                f"Paths: {image_path}, {mask_path}"
            )

        # Apply additional transforms if provided
        if self.transform is not None:
            # For albumentations-style transforms
            if hasattr(self.transform, '__call__'):
                try:
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]
                except TypeError:
                    # Not an albumentations transform, try direct call
                    image, mask = self.transform(image, mask)

        # Convert to tensors
        # image: [H, W, 3] -> [3, H, W]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # mask: [H, W] -> [1, H, W]
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).float()

        return image_tensor, mask_tensor

    def get_file_paths(self, idx: int) -> Tuple[Path, Path]:
        """Get file paths for a sample without loading.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image_path, mask_path).
        """
        return self.pairs[idx]

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics from masks only.

        This method is efficient because it only loads masks,
        not images. Useful for deriving quality filter thresholds
        from real data.

        Returns:
            Dictionary with statistics:
                - num_samples: Number of samples
                - building_counts: List of building counts per mask
                - coverages: List of coverage ratios
                - building_areas: List of all building areas (flattened)
                - coverage_percentiles: Coverage percentiles (50, 95, 99)
                - building_count_percentiles: Building count percentiles
        """
        logger.info("Computing dataset statistics from masks only...")

        building_counts = []
        coverages = []
        building_areas = []

        try:
            from scipy import ndimage
            has_scipy = True
        except ImportError:
            logger.warning("scipy not available, skipping building count computation")
            has_scipy = False

        for idx in range(len(self)):
            # Only load mask, not image
            image_path, mask_path = self.pairs[idx]
            mask = _load_binary_mask(mask_path, self.image_size)

            # Compute coverage
            coverage = mask.sum() / mask.size
            coverages.append(coverage)

            # Compute building count and areas
            if has_scipy:
                labeled, num_features = ndimage.label(mask)
                building_counts.append(num_features)

                # Get area of each building
                for label_id in range(1, num_features + 1):
                    area = int((labeled == label_id).sum())
                    building_areas.append(area)

        stats = {
            'num_samples': len(self),
            'building_counts': building_counts,
            'coverages': coverages,
            'building_areas': building_areas,
            'coverage_percentiles': {
                50: float(np.percentile(coverages, 50)) if coverages else 0.0,
                95: float(np.percentile(coverages, 95)) if coverages else 0.0,
                99: float(np.percentile(coverages, 99)) if coverages else 0.0,
            },
            'building_count_percentiles': {
                50: int(np.percentile(building_counts, 50)) if building_counts else 0,
                95: int(np.percentile(building_counts, 95)) if building_counts else 0,
                99: int(np.percentile(building_counts, 99)) if building_counts else 0,
            },
            'max_coverage': float(max(coverages)) if coverages else 0.0,
            'max_building_count': max(building_counts) if building_counts else 0,
        }

        logger.info(
            f"Statistics: {len(self)} samples, "
            f"coverage 99th: {stats['coverage_percentiles'][99]:.4f}, "
            f"building count 99th: {stats['building_count_percentiles'][99]}"
        )

        return stats

class PairedSatelliteDatasetWithAugmentation(PairedSatelliteDataset):
    """Dataset with built-in augmentation for paired data.

    Applies consistent geometric augmentations to both image and mask
    to maintain pairing integrity. Uses seeded RNG for reproducibility.

    Attributes:
        flip_prob: Probability of horizontal flip.
        rotate_prob: Probability of 90-degree rotation.

    Example:
        >>> dataset = PairedSatelliteDatasetWithAugmentation(config)
        >>> image, mask = dataset[0]
    """

    def __init__(
        self,
        config: Optional[SyntheticConfig] = None,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        **kwargs
    ) -> None:
        """Initialize the augmented dataset.

        Args:
            config: SyntheticConfig instance.
            flip_prob: Probability of horizontal flip.
            rotate_prob: Probability of random rotation (0, 90, 180, 270).
            **kwargs: Additional arguments passed to PairedSatelliteDataset.
        """
        super().__init__(config=config, **kwargs)

        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get augmented image-mask pair.

        Args:
            idx: Index of the sample.

        Returns:
            Augmented (image, mask) tensors.
        """
        image, mask = super().__getitem__(idx)

        # Apply consistent augmentation using seeded RNG
        image, mask = self._augment_pair(image, mask)

        return image, mask

    def _augment_pair(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent augmentation to image-mask pair.

        Uses the dataset's RNG for reproducibility.

        Args:
            image: Image tensor [C, H, W].
            mask: Mask tensor [1, H, W].

        Returns:
            Augmented (image, mask) tensors.
        """
        # Horizontal flip
        if self.rng.random() < self.flip_prob:
            image = torch.flip(image, dims=[2])  # Flip along W
            mask = torch.flip(mask, dims=[2])

        # Random rotation (0, 90, 180, 270 degrees)
        if self.rng.random() < self.rotate_prob:
            k = self.rng.randint(1, 3)  # Number of 90-degree rotations (1-3)
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        return image, mask

# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(
    config: SyntheticConfig,
    use_augmentation: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the paired dataset.

    Args:
        config: SyntheticConfig instance.
        use_augmentation: Whether to use augmentation.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        DataLoader instance.
    """
    if use_augmentation:
        dataset = PairedSatelliteDatasetWithAugmentation(config=config)
    else:
        dataset = PairedSatelliteDataset(config=config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
        drop_last=drop_last,
    )

    return dataloader
