"""Urban metric extraction from binary building masks."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def to_binary_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a mask-like array to a 2D uint8 binary mask."""

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {mask.shape}.")

    return (mask > 0).astype(np.uint8)


def extract_building_count(
    binary_mask: np.ndarray | torch.Tensor,
    min_area: int = 1,
) -> int:
    """Count connected building components in a binary mask."""

    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    mask = to_binary_mask(binary_mask)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    building_count = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            building_count += 1

    return building_count


def extract_building_coverage(binary_mask: np.ndarray | torch.Tensor) -> float:
    """Return building coverage as percent of total image pixels."""

    mask = to_binary_mask(binary_mask)
    building_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    if total_pixels == 0:
        raise ValueError("Cannot compute coverage for an empty mask.")

    return building_pixels / total_pixels * 100.0


def compare_ground_truth_prediction(
    ground_truth_mask: np.ndarray | torch.Tensor,
    predicted_mask: np.ndarray | torch.Tensor,
    min_area: int = 1,
) -> dict[str, float | int]:
    """Compare building count and coverage between ground truth and prediction."""

    ground_truth_count = extract_building_count(ground_truth_mask, min_area=min_area)
    predicted_count = extract_building_count(predicted_mask, min_area=min_area)
    ground_truth_coverage = extract_building_coverage(ground_truth_mask)
    predicted_coverage = extract_building_coverage(predicted_mask)

    return {
        "ground_truth_count": ground_truth_count,
        "predicted_count": predicted_count,
        "count_error": abs(predicted_count - ground_truth_count),
        "ground_truth_coverage": ground_truth_coverage,
        "predicted_coverage": predicted_coverage,
        "coverage_error": abs(predicted_coverage - ground_truth_coverage),
    }
