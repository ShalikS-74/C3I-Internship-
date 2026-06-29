"""Evaluation utilities for synthetic satellite imagery.

This module provides quality assessment and filtering for generated
(RGB, mask) pairs, ensuring only high-quality samples are used
for training augmentation.
"""

from .quality_filter import (
    QualityFilter,
    AdaptiveQualityFilter,
    QualityMetrics,
    compute_filter_statistics,
)

__all__ = [
    "QualityFilter",
    "AdaptiveQualityFilter",
    "QualityMetrics",
    "compute_filter_statistics",
]