"""Model factory for binary building segmentation.

Supported architectures:
    - DeepLabV3+ + scSE  (get_deeplabv3plus_model)
    - FCN-32s             (get_fcn_model)
    - FCN-32s + scSE      (get_fcn_scse_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.modules as md


# ---------------------------------------------------------------------------
# DeepLabV3+ + scSE
# ---------------------------------------------------------------------------

class DeepLabV3PlusWithSCSE(nn.Module):
    """DeepLabV3+ with scSE attention on decoder output."""

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
            classes=self.DECODER_CHANNELS,
            activation=None,
        )

        self.attention = md.SCSEModule(
            in_channels=self.DECODER_CHANNELS,
            reduction=16,
        )

        self.head = nn.Conv2d(self.DECODER_CHANNELS, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        features = self.attention(features)
        logits = self.head(features)

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


# ---------------------------------------------------------------------------
# FCN-32s
# ---------------------------------------------------------------------------

class FCN32s(nn.Module):
    """FCN-32s: ResNet34 encoder → 1×1 conv head → 32× bilinear upsample.

    No skip connections, no ASPP — faithful FCN-32s (Long et al. 2015).

    Channel arithmetic:
      ResNet34 stage-4 → 512 channels  (always true for ResNet34)
      head: Conv2d(512, classes, 1)    → classes channels
      upsample: bilinear to input res  → [B, classes, H, W]
    """

    ENCODER_OUT_CHANNELS = 512  # ResNet34 final stage — always 512

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        actual_out = self.encoder.out_channels[-1]
        assert actual_out == self.ENCODER_OUT_CHANNELS, (
            f"Expected encoder final channels={self.ENCODER_OUT_CHANNELS}, "
            f"got {actual_out}. Use ResNet34."
        )

        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.ENCODER_OUT_CHANNELS, classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        deep_features = features[-1]        # (B, 512, H/32, W/32)
        logits = self.head(deep_features)   # (B, classes, H/32, W/32)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )   # (B, classes, H, W)

        return logits


def get_fcn_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Create an FCN-32s model for binary segmentation."""

    return FCN32s(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


# ---------------------------------------------------------------------------
# FCN-32s + scSE
# ---------------------------------------------------------------------------

class FCN32sWithSCSE(nn.Module):
    """FCN-32s with scSE attention on encoder output before the head.

    Design follows the same philosophy as DeepLabV3PlusWithSCSE:
      encoder → feature map → scSE → head → upsample → logits

    Channel arithmetic:
      ResNet34 stage-4 → 512 channels         (B, 512, H/32, W/32)
      scSE: SCSEModule(512)                   (B, 512, H/32, W/32)  unchanged
      head: Conv2d(512, classes, 1)           (B, classes, H/32, W/32)
      upsample: bilinear to input resolution  (B, classes, H, W)

    512 is hardcoded safely — ResNet34 final stage is always 512.
    The assert in __init__ catches any wrong encoder at construction time.
    """

    ENCODER_OUT_CHANNELS = 512  # ResNet34 final stage — always 512

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        actual_out = self.encoder.out_channels[-1]
        assert actual_out == self.ENCODER_OUT_CHANNELS, (
            f"Expected encoder final channels={self.ENCODER_OUT_CHANNELS}, "
            f"got {actual_out}. Use ResNet34."
        )

        # scSE applied to the 512-channel encoder output
        self.attention = md.SCSEModule(
            in_channels=self.ENCODER_OUT_CHANNELS,
            reduction=16,
        )

        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.ENCODER_OUT_CHANNELS, classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        features = self.encoder(x)
        deep_features = features[-1]              # (B, 512, H/32, W/32)
        deep_features = self.attention(deep_features)  # (B, 512, H/32, W/32)
        logits = self.head(deep_features)         # (B, classes, H/32, W/32)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )   # (B, classes, H, W)

        return logits


def get_fcn_scse_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> torch.nn.Module:
    """Create an FCN-32s + scSE model for binary segmentation."""

    return FCN32sWithSCSE(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
