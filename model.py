"""Model factory for binary building segmentation."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def get_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Create a U-Net with a single-logit binary segmentation head."""

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
