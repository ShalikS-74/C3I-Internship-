"""Dataset utilities for SpaceNet-style building footprint segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _iter_files(folder: Path, extensions: Sequence[str], recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def find_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
    mask_extensions: Sequence[str] = MASK_EXTENSIONS,
    recursive: bool = True,
) -> list[tuple[Path, Path]]:
    """Match image and mask files by filename stem."""

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    masks_by_stem: dict[str, Path] = {}
    for mask_path in _iter_files(mask_dir, mask_extensions, recursive):
        masks_by_stem.setdefault(mask_path.stem, mask_path)

    pairs: list[tuple[Path, Path]] = []
    for image_path in _iter_files(image_dir, image_extensions, recursive):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise ValueError(
            "No matching image/mask pairs found. Files must share the same stem, "
            "for example image_001.tif and image_001.png."
        )

    return pairs


def _load_rgb_image(path: Path, image_size: int | None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] >= 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")

    if image_size is not None:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    original_dtype = image.dtype
    image = image.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        image /= float(np.iinfo(original_dtype).max)
    elif image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)


def _load_binary_mask(path: Path, image_size: int | None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    if image_size is not None:
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    return (mask > 0).astype(np.float32)


class BuildingFootprintDataset(Dataset):
    """PyTorch Dataset for satellite images and binary building masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int | None = 256,
        recursive: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.pairs = find_image_mask_pairs(image_dir, mask_dir, recursive=recursive)

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]

        image = _load_rgb_image(image_path, self.image_size)
        mask = _load_binary_mask(mask_path, self.image_size)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Create reproducible train/validation subsets."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two matched samples are required for a train/validation split.")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_count = min(val_count, len(indices) - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_dataloaders(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_size: int | None = 256,
    batch_size: int = 4,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from image/mask folders."""

    dataset = BuildingFootprintDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        max_samples=max_samples,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
