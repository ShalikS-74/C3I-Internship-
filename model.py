"""
model.py
Segmentation model factory for building footprint detection.
Supports: DeepLabV3+, FCN, U-Net, PSPNet (all optionally with SCSE).
Encoder: ResNet34, pretrained on ImageNet.
Output: [B, 1, H, W] raw logits.

SCSE placement rationale:
  PSPNet:     after PPM output [B, 512, H/32, W/32] — recalibrates global context
              before seghead upsamples x32 to full resolution. PRIMARY location.
  DeepLabV3+: after ASPP decoder output [B, 256, H/4, W/4] before seghead.
  FCN/FPN:    after FPN decoder output before seghead.
  U-Net:      native SMP decoder_attention_type="scse" (per skip connection).
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation_models_pytorch is required. "
        "pip install segmentation-models-pytorch"
    ) from e


# ---------------------------------------------------------------------------
# SCSEModule: try SMP's built-in first, fallback to local implementation
# ---------------------------------------------------------------------------
try:
    from segmentation_models_pytorch.base.modules import SCSEModule
except ImportError:
    try:
        from segmentation_models_pytorch.modules import SCSEModule
    except ImportError:
        class SCSEModule(nn.Module):
            """
            Squeeze-and-Channel/Spatial Excitation module.
            cSE: global avg pool -> FC -> ReLU -> FC -> sigmoid -> channel recalibration
            sSE: 1x1 conv -> sigmoid -> spatial recalibration
            Output: x * cSE(x) + x * sSE(x)
            """
            def __init__(self, in_channels: int, reduction: int = 16):
                super().__init__()
                r = max(1, in_channels // reduction)
                self.cSE = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, r, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(r, in_channels, kernel_size=1, bias=False),
                    nn.Sigmoid(),
                )
                self.sSE = nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.cSE(x) + x * self.sSE(x)


# ---------------------------------------------------------------------------
# PSPNet + SCSE wrapper
# ---------------------------------------------------------------------------
class PSPNetWithSCSE(nn.Module):
    """
    PSPNet with SCSE attention at two locations:

    1. After the final encoder stage only (encoder_scse[-1]).
       PSPDecoder.forward() accepts exactly ONE tensor — features[-1].
       Only the last encoder feature feeds the PPM, so only that SCSE
       gate has a gradient path to the loss.

    2. After the Pyramid Pooling Module output (ppm_scse). PRIMARY location.
       PPM aggregates global multi-scale context into [B, 512, H/32, W/32].
       SCSE here recalibrates channels + spatial weights before the
       segmentation head upsamples x32 to full resolution.

    Tensor flow (ResNet34, 512x512 input):
        x                [B,  3, 512, 512]
        features[0]      [B,  3, 512, 512]  raw input
        features[1]      [B, 64, 256, 256]
        features[2]      [B, 64, 128, 128]
        features[3]      [B,128,  64,  64]
        features[4]      [B,256,  32,  32]
        features[5]      [B,512,  16,  16]  <- encoder_scse applied here only
        decoder_output   [B,512,  16,  16]  PSPDecoder(features[-1]) output
        ppm_scse output  [B,512,  16,  16]  PRIMARY SCSE
        seghead output   [B,  1, 512, 512]  final logits

    NOTE: PSPDecoder.forward() signature is forward(self, x) — takes ONE
    tensor, not a list. Passing *features would raise TypeError.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation=None,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        scse_reduction: int = 16,
    ):
        super().__init__()

        self.pspnet = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            psp_out_channels=psp_out_channels,
            psp_use_batchnorm=psp_use_batchnorm,
            psp_dropout=psp_dropout,
        )

        # encoder.out_channels for resnet34 = (3, 64, 64, 128, 256, 512)
        # Only the LAST encoder feature feeds PSPDecoder — apply SCSE there only.
        last_encoder_channels = self.pspnet.encoder.out_channels[-1]  # 512
        self.encoder_scse = SCSEModule(
            last_encoder_channels,
            reduction=max(1, last_encoder_channels // scse_reduction),
        )

        # PRIMARY SCSE: after PPM output [B, psp_out_channels, H/32, W/32]
        self.ppm_scse = SCSEModule(
            psp_out_channels,
            reduction=max(1, psp_out_channels // scse_reduction),
        )

        # Direct references — avoids double nesting in state_dict keys
        self._encoder = self.pspnet.encoder
        self._decoder = self.pspnet.decoder
        self._seghead = self.pspnet.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: encode -> list of feature maps
        features = self._encoder(x)

        # Step 2: SCSE on the last encoder feature only
        # features[-1] shape: [B, 512, H/32, W/32]
        last_feature = self.encoder_scse(features[-1])

        # Step 3: PSPDecoder takes exactly ONE tensor (not *features list)
        # Output shape: [B, psp_out_channels, H/32, W/32]
        decoder_output = self._decoder(last_feature)

        # Step 4: PRIMARY SCSE — recalibrate PPM output before seghead
        # [B, 512, H/32, W/32] -> [B, 512, H/32, W/32]
        decoder_output = self.ppm_scse(decoder_output)

        # Step 5: segmentation head — conv + upsample -> [B, classes, H, W]
        return self._seghead(decoder_output)

    @property
    def encoder(self):
        return self._encoder


# ---------------------------------------------------------------------------
# DeepLabV3+ + SCSE wrapper
# ---------------------------------------------------------------------------
class DeepLabV3PlusWithSCSE(nn.Module):
    """
    DeepLabV3+ with SCSE applied after the ASPP decoder output,
    before the segmentation head.

    SMP's DeepLabV3Plus has no decoder_attention_type parameter.
    We intercept between decoder and seghead.

    Tensor flow (ResNet34, 512x512 input):
        decoder output  [B, 256, 128, 128]  (H/4, W/4)
        scse output     [B, 256, 128, 128]
        seghead output  [B,   1, 512, 512]
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        scse_reduction: int = 16,
    ):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
        )

        # DeepLabV3Plus decoder outputs [B, decoder_channels, H/4, W/4]
        self.scse = SCSEModule(
            decoder_channels,
            reduction=max(1, decoder_channels // scse_reduction),
        )

        self._encoder = self.model.encoder
        self._decoder = self.model.decoder
        self._seghead = self.model.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encoder(x)
        decoder_output = self._decoder(*features)   # [B, 256, H/4, W/4]
        decoder_output = self.scse(decoder_output)  # [B, 256, H/4, W/4]
        return self._seghead(decoder_output)         # [B, 1, H, W]

    @property
    def encoder(self):
        return self._encoder


# ---------------------------------------------------------------------------
# FPN (FCN-equivalent) + SCSE wrapper
# ---------------------------------------------------------------------------
class FPNWithSCSE(nn.Module):
    """
    FPN with SCSE applied after the FPN decoder output, before seghead.

    SMP's FPN has no native SCSE parameter.
    Output channels depend on decoder_merge_policy:
      "add" -> decoder_segmentation_channels (128)
      "cat" -> decoder_segmentation_channels * num_encoder_stages (varies)
    We probe at init time to get the exact channel count safely.

    Tensor flow (ResNet34, 512x512, merge_policy="add"):
        decoder output  [B, 128, 128, 128]  (H/4, W/4)
        scse output     [B, 128, 128, 128]
        seghead output  [B,   1, 512, 512]
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        scse_reduction: int = 16,
    ):
        super().__init__()

        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_pyramid_channels=decoder_pyramid_channels,
            decoder_segmentation_channels=decoder_segmentation_channels,
            decoder_merge_policy=decoder_merge_policy,
        )

        # Probe decoder output channels at init — handles any merge_policy
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            feats = self.model.encoder(dummy)
            dec_out = self.model.decoder(*feats)
            fpn_out_channels = dec_out.shape[1]

        self.scse = SCSEModule(
            fpn_out_channels,
            reduction=max(1, fpn_out_channels // scse_reduction),
        )

        self._encoder = self.model.encoder
        self._decoder = self.model.decoder
        self._seghead = self.model.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encoder(x)
        decoder_output = self._decoder(*features)
        decoder_output = self.scse(decoder_output)
        return self._seghead(decoder_output)

    @property
    def encoder(self):
        return self._encoder


# ---------------------------------------------------------------------------
# Supported model names
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = [
    "deeplabv3plus",
    "deeplabv3plus_scse",
    "fcn",
    "fcn_scse",
    "unet_scse",
    "pspnet",
    "pspnet_scse",
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_model(
    model_name: str,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Returns a segmentation model with output [B, classes, H, W] raw logits.

    Args:
        model_name:      One of SUPPORTED_MODELS.
        encoder_name:    SMP encoder name (default: resnet34).
        encoder_weights: Pretrained weights source or None (default: imagenet).
        in_channels:     Input image channels (default: 3).
        classes:         Output classes (default: 1 for binary segmentation).

    Returns:
        nn.Module producing [B, classes, H, W] raw logits (no activation).

    Raises:
        ValueError: if model_name is not in SUPPORTED_MODELS.
    """
    name = model_name.lower().strip()

    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )

    common = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    # ------------------------------------------------------------------
    # DeepLabV3+
    # ------------------------------------------------------------------
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common, activation=None)

    if name == "deeplabv3plus_scse":
        return DeepLabV3PlusWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            scse_reduction=16,
        )

    # ------------------------------------------------------------------
    # FCN (FPN as FCN-equivalent in SMP)
    # ------------------------------------------------------------------
    if name == "fcn":
        return smp.FPN(**common, activation=None)

    if name == "fcn_scse":
        return FPNWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_merge_policy="add",
            scse_reduction=16,
        )

    # ------------------------------------------------------------------
    # U-Net + SCSE (native SMP support — SCSE per decoder block)
    # ------------------------------------------------------------------
    if name == "unet_scse":
        return smp.Unet(
            **common,
            activation=None,
            decoder_attention_type="scse",
        )

    # ------------------------------------------------------------------
    # PSPNet
    # ------------------------------------------------------------------
    if name == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
        )

    if name == "pspnet_scse":
        return PSPNetWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            scse_reduction=16,
        )

    raise ValueError(f"Unhandled model: {name}")  # unreachable — safety net


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
def get_fcn_scse_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Alias kept for backward compatibility. Use get_model('fcn_scse') instead."""
    return get_model(
        model_name="fcn_scse",
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def get_deeplabv3plus_scse_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Alias kept for backward compatibility. Use get_model('deeplabv3plus_scse') instead."""
    return get_model(
        model_name="deeplabv3plus_scse",
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path: str, metadata: dict = None) -> None:
    """Save model state dict and optional metadata to path."""
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    print(f"[Checkpoint] Saved -> {path}")


def load_checkpoint(model: nn.Module, path: str, strict: bool = True) -> dict:
    """
    Load checkpoint into model in-place.

    Supports:
      - New format: {"model_state_dict": ..., "metadata": ...}
      - Old format: raw state dict
      - DataParallel prefix: "module." stripped automatically

    Returns:
        metadata dict (empty if not present)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint
        metadata = {}

    # Strip DataParallel prefix if present
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Checkpoint] Missing keys    ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    print(f"[Checkpoint] Loaded <- {path}")
    return metadata


# ---------------------------------------------------------------------------
# Shape test  (python model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Shape verification — all models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 512, 512).to(device)

    for model_name in SUPPORTED_MODELS:
        try:
            model = get_model(model_name).to(device)
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (2, 1, 512, 512), f"Bad shape: {out.shape}"
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(
                f"  OK   {model_name:<25} "
                f"output={tuple(out.shape)}  "
                f"params={params:.1f}M"
            )
        except Exception as e:
            print(f"  FAIL {model_name:<25} {e}")

    print("=" * 60)
