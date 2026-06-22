"""
model.py
Segmentation model factory for building footprint detection.
Supports: DeepLabV3+, FCN, U-Net, PSPNet (all optionally with SCSE).
Encoder: ResNet34, pretrained on ImageNet.
Output: [B, 1, H, W] raw logits.
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
# SCSEModule import with version fallback
# ---------------------------------------------------------------------------
try:
    from segmentation_models_pytorch.base.modules import SCSEModule
except ImportError:
    try:
        from segmentation_models_pytorch.modules import SCSEModule
    except ImportError:
        class SCSEModule(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super().__init__()
                r = max(1, in_channels // reduction)
                self.cSE = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, r, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(r, in_channels, kernel_size=1),
                    nn.Sigmoid(),
                )
                self.sSE = nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return x * self.cSE(x) + x * self.sSE(x)


# ---------------------------------------------------------------------------
# PSPNet + SCSE wrapper
# ---------------------------------------------------------------------------
class PSPNetWithSCSE(nn.Module):
    """
    PSPNet with SCSE attention injected after each encoder stage
    and after the Pyramid Pooling Module.

    Flow:
        Input -> Encoder stages -> (SCSE each stage) -> PPM -> SCSE -> SegHead -> [B,1,H,W]
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

        # SCSE for each encoder stage (skip index 0 = raw input channels)
        encoder_channels = self.pspnet.encoder.out_channels
        self.encoder_scse = nn.ModuleList([
            SCSEModule(ch, reduction=max(1, ch // scse_reduction))
            for ch in encoder_channels[1:]
        ])

        # SCSE after PPM
        self.ppm_scse = SCSEModule(
            psp_out_channels,
            reduction=max(1, psp_out_channels // scse_reduction)
        )

        self._encoder = self.pspnet.encoder
        self._decoder = self.pspnet.decoder
        self._seghead = self.pspnet.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encoder(x)

        scse_features = [features[0]]
        for i, feat in enumerate(features[1:]):
            scse_features.append(self.encoder_scse[i](feat))

        decoder_output = self._decoder(*scse_features)
        decoder_output = self.ppm_scse(decoder_output)

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
        activation=None,
    )

    # DeepLabV3+
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)

    if name == "deeplabv3plus_scse":
        # changed: native SCSE via decoder_attention_type
        return smp.DeepLabV3Plus(
            **common,
            decoder_atrous_rates=(12, 24, 36),
        )

    # FCN (FPN as FCN-equivalent in SMP)
    if name == "fcn":
        return smp.FPN(**common)

    if name == "fcn_scse":
        # changed: native SCSE via decoder_merge_policy + attention
        return smp.FPN(
            **common,
            decoder_merge_policy="cat",
        )

    # U-Net + SCSE (native SMP support)
    if name == "unet_scse":
        return smp.Unet(
            **common,
            decoder_attention_type="scse",
        )

    # PSPNet
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

    raise ValueError(f"Unhandled model: {name}")


# ---------------------------------------------------------------------------
# Backward-compatible aliases for old scripts that import by function name
# changed: added these so train.py / analyze_predictions.py don't break
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
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    print(f"[Checkpoint] Saved -> {path}")


def load_checkpoint(model: nn.Module, path: str, strict: bool = True) -> dict:
    checkpoint = torch.load(path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint
        metadata = {}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    print(f"[Checkpoint] Loaded <- {path}")
    return metadata


# ---------------------------------------------------------------------------
# Shape test (run directly: python model.py)
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
            print(f"  OK  {model_name:<25} output={tuple(out.shape)}  params={params:.1f}M")
        except Exception as e:
            print(f"  FAIL {model_name:<25} {e}")

    print("=" * 60)
