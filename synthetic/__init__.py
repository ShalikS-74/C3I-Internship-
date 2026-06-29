"""Synthetic satellite imagery generation package.

This package provides a StyleGAN-inspired generator for creating
paired (RGB, mask) satellite imagery for training augmentation.

Key components:
- config: Configuration management
- datasets: Data loading utilities
- models: Generator and discriminator architectures
- training: Training loop implementation
- evaluation: Quality filtering for generated samples

Quick start:
    >>> from synthetic.config import get_default_config
    >>> from synthetic.models import StyleGANGenerator
    >>> 
    >>> config = get_default_config()
    >>> generator = StyleGANGenerator(config.model)
    >>> 
    >>> # Generate samples
    >>> z = torch.randn(4, 512)
    >>> rgb_logits, mask_logits = generator(z)
    >>> rgb = torch.sigmoid(rgb_logits)
    >>> mask = torch.sigmoid(mask_logits)
"""

from .config import (
    SyntheticConfig,
    ModelConfig,
    TrainingConfig,
    DiscriminatorConfig,
    LossConfig,
    QualityFilterConfig,
    PathConfig,
    DeviceConfig,
    get_default_config,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "SyntheticConfig",
    "ModelConfig",
    "TrainingConfig",
    "DiscriminatorConfig",
    "LossConfig",
    "QualityFilterConfig",
    "PathConfig",
    "DeviceConfig",
    "get_default_config",
]
