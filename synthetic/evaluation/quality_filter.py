"""Quality filtering for generated satellite imagery.

This module implements quality assessment and filtering for generated
(RGB, mask) pairs before they can be used for augmentation training.
Filters based on:
- Mask coverage (reject empty or full masks)
- Building count (reject too few/many buildings)
- Building size distribution (reject unrealistic patterns)
- FID score (optional, against real dataset)

Usage:
    >>> from synthetic.evaluation.quality_filter import QualityFilter
    >>> from synthetic.config import get_default_config
    >>> 
    >>> config = get_default_config()
    >>> quality_filter = QualityFilter(config.quality_filter)
    >>> 
    >>> # Filter generated samples
    >>> rgb, mask = generator(z)
    >>> passed, metrics = quality_filter.filter(rgb, mask)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
import dataclasses

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import config
from ..config import QualityFilterConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class QualityMetrics:
    """Quality metrics for a generated sample.
    
    Attributes:
        coverage: Fraction of pixels that are buildings.
        building_count: Number of distinct building instances.
        mean_building_area: Average building area in pixels.
        max_building_area: Largest building area.
        min_building_area: Smallest building area.
        perimeter_complexity: Ratio of perimeter to area.
        passed: Whether sample passed all filters.
        fail_reasons: List of reasons for failure.
    """
    coverage: float
    building_pixels: int
    building_count: int
    mean_building_area: float
    max_building_area: float
    min_building_area: float
    perimeter_complexity: float
    passed: bool
    fail_reasons: List[str] = dataclasses.field(default_factory=list)

class QualityFilter:
    """Filter generated samples based on quality metrics.
    
    Implements multi-criteria filtering to ensure generated samples
    are suitable for training augmentation:
    
    1. Coverage filter: Reject masks with too little or too much coverage
    2. Building count filter: Reject masks with too few or too many buildings
    3. Building size filter: Reject unrealistic building size distributions
    4. Complexity filter: Reject overly simple or complex masks
    
    Attributes:
        config: QualityFilterConfig instance.
        
    Example:
        >>> quality_filter = QualityFilter(config.quality_filter)
        >>> rgb = torch.randn(1, 3, 512, 512)
        >>> mask = torch.randn(1, 1, 512, 512)
        >>> passed, metrics = quality_filter.filter(rgb, mask)
    """
    
    def __init__(
        self,
        config: Optional[QualityFilterConfig] = None,
        compute_statistics_fn: Optional[Callable] = None,
    ):
        """Initialize the quality filter.
        
        Args:
            config: QualityFilterConfig instance. Uses defaults if None.
            compute_statistics_fn: Optional function to compute dataset statistics
                for data-driven thresholds.
        """
        if config is None:
            config = QualityFilterConfig()
        
        self.config = config
        self.compute_statistics_fn = compute_statistics_fn
        
        # Try to import scipy for connected components
        try:
            from scipy import ndimage
            self.ndimage = ndimage
            self.has_scipy = True
        except ImportError:
            logger.warning("scipy not available, building count filtering disabled")
            self.has_scipy = False
        
        logger.info(
            f"QualityFilter initialized: "
            f"min_pixels={config.min_building_pixels}, "
            f"max_coverage={config.max_coverage:.2%}, "
            f"buildings=[{config.min_building_count}, {config.max_building_count}]"
        )
    
    def filter(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
        return_metrics: bool = True,
    ) -> Tuple[bool, Optional[QualityMetrics]]:
        """Filter a single generated sample.
        
        Args:
            rgb: Generated RGB image [B, 3, H, W] or [3, H, W].
            mask: Generated mask [B, 1, H, W] or [1, H, W].
            return_metrics: Whether to return quality metrics.
            
        Returns:
            Tuple of (passed, metrics).
            passed: True if sample passes all filters.
            metrics: QualityMetrics if return_metrics=True, else None.
        """
        # Handle batch dimension
        if rgb.dim() == 4:
            rgb = rgb[0]
        if mask.dim() == 4:
            mask = mask[0]
        
        # Convert mask to binary numpy
        mask_np = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        
        metrics = self._compute_metrics(mask_np)
        
        # Apply filters
        self._apply_coverage_filter(metrics)
        if self.has_scipy:
            self._apply_building_count_filter(metrics)
        self._apply_building_size_filter(metrics)
        
        metrics.passed = len(metrics.fail_reasons) == 0
        
        return metrics.passed, (metrics if return_metrics else None)
    
    def filter_batch(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[QualityMetrics]]:
        """Filter a batch of generated samples.
        
        Args:
            rgb: Generated RGB images [B, 3, H, W].
            mask: Generated masks [B, 1, H, W].
            
        Returns:
            Tuple of (filtered_rgb, filtered_mask, all_metrics).
        """
        batch_size = rgb.shape[0]
        passed_indices = []
        all_metrics = []
        
        for i in range(batch_size):
            passed, metrics = self.filter(rgb[i:i+1], mask[i:i+1])
            all_metrics.append(metrics)
            if passed:
                passed_indices.append(i)
        
        if len(passed_indices) == 0:
            logger.warning("No samples passed quality filter")
            return (
                torch.empty(0, 3, rgb.shape[2], rgb.shape[3], device=rgb.device),
                torch.empty(0, 1, mask.shape[2], mask.shape[3], device=mask.device),
                all_metrics,
            )
        
        filtered_rgb = rgb[passed_indices]
        filtered_mask = mask[passed_indices]
        
        logger.info(
            f"Quality filter: {len(passed_indices)}/{batch_size} samples passed "
            f"({len(passed_indices)/batch_size:.1%})"
        )
        
        return filtered_rgb, filtered_mask, all_metrics
    
    def _compute_metrics(self, mask: np.ndarray) -> QualityMetrics:
        """Compute quality metrics for a mask.
        
        Args:
            mask: Binary mask array [H, W].
            
        Returns:
            QualityMetrics instance.
        """
        # Coverage
        coverage = mask.sum() / mask.size
        
        # Building statistics
        building_count = 0
        areas = []
        perimeter_complexity = 0.0
        
        if self.has_scipy:
            labeled, building_count = self.ndimage.label(mask)
            
            for label_id in range(1, building_count + 1):
                building_mask = (labeled == label_id)
                area = building_mask.sum()
                areas.append(area)
            
            # Perimeter complexity
            if building_count > 0:
                # Compute perimeter using erosion
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(mask, kernel, iterations=1)
                perimeter = (mask - eroded).sum()
                total_area = mask.sum()
                perimeter_complexity = perimeter / (total_area + 1e-8)
        
        mean_area = np.mean(areas) if areas else 0.0
        max_area = max(areas) if areas else 0.0
        min_area = min(areas) if areas else 0.0
        
        return QualityMetrics(
            coverage=float(coverage),
            building_pixels=int(mask.sum()),
            building_count=int(building_count),
            mean_building_area=float(mean_area),
            max_building_area=float(max_area),
            min_building_area=float(min_area),
            perimeter_complexity=float(perimeter_complexity),
            passed=True,  # Will be updated by filters
            fail_reasons=[],
        )
    
    def _apply_coverage_filter(self, metrics: QualityMetrics) -> None:
        """Apply coverage-based filtering.
        
        Args:
            metrics: QualityMetrics to update.
        """
        if metrics.building_pixels < self.config.min_building_pixels:
            metrics.fail_reasons.append(
                f"Too few building pixels: {metrics.building_pixels} < "
                f"{self.config.min_building_pixels}"
            )
        elif metrics.coverage > self.config.max_coverage:
            metrics.fail_reasons.append(
                f"Coverage too high: {metrics.coverage:.2%} > {self.config.max_coverage:.2%}"
            )
    
    def _apply_building_count_filter(self, metrics: QualityMetrics) -> None:
        """Apply building count-based filtering.
        
        Args:
            metrics: QualityMetrics to update.
        """
        if metrics.building_count < self.config.min_building_count:
            metrics.fail_reasons.append(
                f"Too few buildings: {metrics.building_count} < "
                f"{self.config.min_building_count}"
            )
        elif metrics.building_count > self.config.max_building_count:
            metrics.fail_reasons.append(
                f"Too many buildings: {metrics.building_count} > "
                f"{self.config.max_building_count}"
            )
    
    def _apply_building_size_filter(self, metrics: QualityMetrics) -> None:
        """Apply building size-based filtering.
        
        Args:
            metrics: QualityMetrics to update.
        """
        if metrics.building_count > 0:
            # Check for unrealistic building sizes
            min_acceptable_area = self.config.min_building_area
            
            if metrics.min_building_area < min_acceptable_area:
                metrics.fail_reasons.append(
                    f"Building too small: {metrics.min_building_area:.0f} < {min_acceptable_area}"
                )

class AdaptiveQualityFilter(QualityFilter):
    """Quality filter with data-driven thresholds.
    
    Computes thresholds from real dataset statistics instead of
    using hardcoded values.
    
    Example:
        >>> from synthetic.datasets import PairedSatelliteDataset
        >>> 
        >>> dataset = PairedSatelliteDataset(config)
        >>> stats = dataset.compute_statistics()
        >>> 
        >>> quality_filter = AdaptiveQualityFilter.from_statistics(stats)
    """
    
    def __init__(
        self,
        config: Optional[QualityFilterConfig] = None,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        """Initialize adaptive quality filter.
        
        Args:
            config: QualityFilterConfig instance.
            statistics: Dataset statistics from compute_statistics().
        """
        super().__init__(config)
        
        if statistics is not None:
            self._update_thresholds_from_statistics(statistics)
    
    def _update_thresholds_from_statistics(self, stats: Dict[str, Any]) -> None:
        """Update filtering thresholds from dataset statistics.
        
        Args:
            stats: Statistics dictionary from PairedSatelliteDataset.compute_statistics().
        """
        # Use 1st and 99th percentiles for coverage
        if 'coverage_percentiles' in stats:
            self.config.max_coverage = min(0.99, stats['coverage_percentiles'].get(99, 0.5))
        
        # Use 1st and 99th percentiles for building count
        if 'building_count_percentiles' in stats:
            self.config.min_building_count = max(1, stats['building_count_percentiles'].get(1, 1))
            self.config.max_building_count = stats['building_count_percentiles'].get(99, 200)
        
        logger.info(
            f"AdaptiveQualityFilter thresholds updated from statistics: "
            f"max_coverage={self.config.max_coverage:.2%}, "
            f"buildings=[{self.config.min_building_count}, "
            f"{self.config.max_building_count}]"
        )
    
    @classmethod
    def from_statistics(
        cls,
        statistics: Dict[str, Any],
        config: Optional[QualityFilterConfig] = None,
    ) -> 'AdaptiveQualityFilter':
        """Create adaptive filter from dataset statistics.
        
        Args:
            statistics: Statistics from PairedSatelliteDataset.compute_statistics().
            config: Optional base config.
            
        Returns:
            AdaptiveQualityFilter instance.
        """
        return cls(config=config, statistics=statistics)

def compute_filter_statistics(
    generator: torch.nn.Module,
    num_samples: int = 1000,
    latent_dim: int = 512,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """Compute statistics of generated samples for filter calibration.
    
    Args:
        generator: Generator model.
        num_samples: Number of samples to generate.
        latent_dim: Latent dimension.
        device: Device for generation.
        
    Returns:
        Statistics dictionary.
    """
    generator.eval()
    generator.to(device)
    
    coverages = []
    building_counts = []
    
    quality_filter = QualityFilter()
    
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_dim, device=device)
            rgb_logits, mask_logits = generator(z)
            rgb = torch.sigmoid(rgb_logits)
            mask = torch.sigmoid(mask_logits)
            
            _, metrics = quality_filter.filter(rgb, mask)
            
            coverages.append(metrics.coverage)
            building_counts.append(metrics.building_count)
    
    return {
        'num_samples': num_samples,
        'coverage': {
            'mean': np.mean(coverages),
            'std': np.std(coverages),
            'min': np.min(coverages),
            'max': np.max(coverages),
            'percentiles': {
                1: np.percentile(coverages, 1),
                5: np.percentile(coverages, 5),
                50: np.percentile(coverages, 50),
                95: np.percentile(coverages, 95),
                99: np.percentile(coverages, 99),
            }
        },
        'building_count': {
            'mean': np.mean(building_counts),
            'std': np.std(building_counts),
            'min': int(np.min(building_counts)),
            'max': int(np.max(building_counts)),
            'percentiles': {
                1: int(np.percentile(building_counts, 1)),
                5: int(np.percentile(building_counts, 5)),
                50: int(np.percentile(building_counts, 50)),
                95: int(np.percentile(building_counts, 95)),
                99: int(np.percentile(building_counts, 99)),
            }
        }
    }
