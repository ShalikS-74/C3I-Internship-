"""Model factory for binary building segmentation using DeepLabV3+ with scSE attention."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.modules as md
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3PlusWithSCSE(nn.Module):
    """DeepLabV3+ with scSE attention on decoder output — minimal change."""

    DECODER_CHANNELS = 256

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        self.backbone = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=self.DECODER_CHANNELS,  # expose raw decoder features
            activation=None,
        )

        # scSE block — same module smp.Unet uses internally
        self.attention = md.SCSEModule(
            in_channels=self.DECODER_CHANNELS,
            reduction=16,
        )

        # Restore the final logit head
        self.head = nn.Conv2d(self.DECODER_CHANNELS, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)           # (B, 256, H', W')
        features = self.attention(features)   # scSE applied
        logits = self.head(features)          # (B, classes, H', W')

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
        return logits


def get_deeplabv3plus_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Create DeepLabV3+ with scSE attention for binary segmentation."""

    return DeepLabV3PlusWithSCSE(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
