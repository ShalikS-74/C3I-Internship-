"""Model factory for binary building segmentation.

Supported architectures:
    - FCN-32s  (get_fcn_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# FCN-32s
# ---------------------------------------------------------------------------

class FCN32s(nn.Module):
    """FCN-32s: ResNet34 encoder → 1×1 conv head → 32× bilinear upsample.

    Faithfully implements Long et al. 2015 FCN-32s:
      - No skip connections.
      - No ASPP.
      - Single prediction from the deepest feature map (stride-32).

    Channel arithmetic:
      ResNet34 stage-4 output → 512 channels  (always true for ResNet34)
      head: Conv2d(512, classes, 1)            → classes channels
      upsample: bilinear to input resolution   → [B, classes, H, W]
    """

    # ResNet34 final encoder stage is always 512 — safe to hardcode.
    ENCODER_OUT_CHANNELS = 512

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        # Use smp encoder — gives us pretrained ResNet34 with clean API
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        # Verify channel count at runtime — catches wrong encoder silently
        actual_out = self.encoder.out_channels[-1]
        assert actual_out == self.ENCODER_OUT_CHANNELS, (
            f"Expected encoder final channels={self.ENCODER_OUT_CHANNELS}, "
            f"got {actual_out}. Use ResNet34."
        )

        # FCN head: single 1×1 conv, no bias needed before upsample
        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),                          # light regularisation
            nn.Conv2d(self.ENCODER_OUT_CHANNELS, classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]   # (H, W) — needed for upsample target

        # encoder returns list: [stem, s1, s2, s3, s4, s5]
        # we only need the last feature map (stride-32)
        features = self.encoder(x)
        deep_features = features[-1]   # (B, 512, H/32, W/32)

        logits = self.head(deep_features)   # (B, classes, H/32, W/32)

        # 32× upsample back to input resolution — FCN-32s defining operation
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
