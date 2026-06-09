"""Metrics and post-processing helpers for building masks."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def logits_to_binary_tensor(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert raw model logits to a binary tensor."""

    probabilities = torch.sigmoid(logits.detach())
    return (probabilities >= threshold).float()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batch mean Intersection over Union for binary segmentation."""

    preds = logits_to_binary_tensor(logits, threshold)
    targets = (targets > 0.5).float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection

    return ((intersection + eps) / (union + eps)).mean()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batch mean Dice score for binary segmentation."""

    preds = logits_to_binary_tensor(logits, threshold)
    targets = (targets > 0.5).float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    total = preds.sum(dim=dims) + targets.sum(dim=dims)

    return ((2.0 * intersection + eps) / (total + eps)).mean()


def binary_mask_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert one prediction tensor to a 2D uint8 NumPy mask."""

    if logits.ndim == 4:
        raise ValueError("Pass a single prediction, not a full batch.")

    mask = logits_to_binary_tensor(logits, threshold).cpu().numpy()
    return binary_mask_to_uint8(mask)


def binary_mask_to_uint8(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize a mask-like array to 2D uint8 values of 0 or 1."""

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3:
        mask = mask[:, :, 0]

    return (mask > 0).astype(np.uint8)


def count_buildings(binary_mask: np.ndarray | torch.Tensor, min_area: int = 20) -> int:
    """Count connected building regions in a predicted binary mask."""

    mask = binary_mask_to_uint8(binary_mask)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    count = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            count += 1

    return count


def building_coverage_percentage(binary_mask: np.ndarray | torch.Tensor) -> float:
    """Compute building pixel coverage as a percentage of the image."""

    mask = binary_mask_to_uint8(binary_mask)
    return float(mask.sum() / mask.size * 100.0)


def summarize_building_mask(
    binary_mask: np.ndarray | torch.Tensor,
    min_area: int = 20,
) -> dict[str, float | int]:
    """Return count and coverage for one binary building mask."""

    return {
        "building_count": count_buildings(binary_mask, min_area=min_area),
        "building_coverage_percent": building_coverage_percentage(binary_mask),
    }
