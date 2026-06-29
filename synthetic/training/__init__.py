"""Training utilities for StyleGAN satellite imagery generation.

This module provides the training loop implementation including:
- Trainer class for managing training
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
"""

from .trainer import Trainer, train_from_config, get_lr_scheduler

__all__ = [
    "Trainer",
    "train_from_config",
    "get_lr_scheduler",
]