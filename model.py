"""Model factory for binary building segmentation using DeepLabV3+."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def get_deeplabv3plus_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Create a DeepLabV3+ model with a single-logit binary segmentation head."""

    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
