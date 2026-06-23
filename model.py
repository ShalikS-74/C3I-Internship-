"""
model.py - Segmentation Model Factory for Building Footprint Detection

Supports 7 model variants:
- deeplabv3plus
- deeplabv3plus_scse
- fcn
- fcn_scse
- unet_scse
- pspnet
- pspnet_scse

All models use ResNet34 encoder with ImageNet pretraining.
Output: [B, 1, H, W] raw logits (no sigmoid applied)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

import segmentation_models_pytorch as smp


# =============================================================================
# SCSE Module
# =============================================================================

# Try to import SCSE from SMP, fall back to local implementation
try:
    from segmentation_models_pytorch.base.modules import SCSEModule as _SMP_SCSE
    SCSEModule = _SMP_SCSE
except (ImportError, AttributeError):
    # Local SCSE implementation
    class SCSEModule(nn.Module):
        """
        Squeeze-and-Excitation + Spatial Excitation Module
        
        Concurrent Spatial and Channel Squeeze & Excitation
        Reference: https://arxiv.org/abs/1803.02579
        """
        def __init__(self, in_channels: int, reduction: int = 16):
            super().__init__()
            self.cSE = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid()
            )
            self.sSE = nn.Sequential(
                nn.Conv2d(in_channels, 1, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cse = self.cSE(x)
            sse = self.sSE(x)
            return x * cse + x * sSE


# =============================================================================
# Supported Models
# =============================================================================

SUPPORTED_MODELS: List[str] = [
    "deeplabv3plus",
    "deeplabv3plus_scse",
    "fcn",
    "fcn_scse",
    "unet_scse",
    "pspnet",
    "pspnet_scse",
]


# =============================================================================
# DeepLabV3+ with SCSE
# =============================================================================

class DeepLabV3PlusWithSCSE(nn.Module):
    """
    DeepLabV3+ with SCSE attention after decoder output.
    
    SCSE is applied to the decoder's output before the final segmentation head.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        
        # Create base DeepLabV3+ model
        self._model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # The SCSE block receives the decoder output, not the encoder output.
        # DeepLabV3+ with ResNet34 decodes 512 encoder channels into 256 channels.
        decoder_channels = self._model.segmentation_head[0].in_channels
        
        # Add SCSE after decoder output
        self.scse = SCSEModule(decoder_channels)
        
        # Store the original segmentation head
        self.seg_head = self._model.segmentation_head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features
        features = self._model.encoder(x)
        
        # Get decoder output (before final head)
        decoder_output = self._model.decoder(features)
        
        # Apply SCSE
        decoder_output = self.scse(decoder_output)
        
        # Apply segmentation head
        output = self.seg_head(decoder_output)
        
        return output


# =============================================================================
# FPN (FCN) with SCSE
# =============================================================================

class FPNWithSCSE(nn.Module):
    """
    FPN (used as FCN in SMP) with SCSE attention after decoder output.
    
    Note: SMP's FCN implementation is FPN-based.
    SCSE is applied to the decoder's final output before the segmentation head.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        
        # Create base FPN model (SMP's FCN is FPN)
        self._model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # Probe decoder output channels by dry run
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            dummy_features = self._model.encoder(dummy)
            dummy_decoder_out = self._model.decoder(dummy_features)
            decoder_channels = dummy_decoder_out.shape[1]
        
        # Add SCSE after decoder output
        self.scse = SCSEModule(decoder_channels)
        
        # Store the original segmentation head
        self.seg_head = self._model.segmentation_head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features
        features = self._model.encoder(x)
        
        # Get decoder output (before final head)
        decoder_output = self._model.decoder(features)
        
        # Apply SCSE
        decoder_output = self.scse(decoder_output)
        
        # Apply segmentation head
        output = self.seg_head(decoder_output)
        
        return output


# =============================================================================
# PSPNet with SCSE
# =============================================================================

class PSPNetWithSCSE(nn.Module):
    """
    PSPNet with SCSE attention.
    
    SCSE is applied at two locations:
    1. encoder_scse: On encoder's final feature map (features[-1])
    2. ppm_scse: On PPM decoder output (PRIMARY SCSE location)
    
    Handles SMP version differences in PSPDecoder.forward() by validating
    list, varargs, and final-feature calling conventions at initialization.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        
        # Create base PSPNet model
        self._model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        # Get encoder output channels
        encoder_channels = self._model.encoder.out_channels
        final_encoder_channels = encoder_channels[-1]  # 512 for ResNet34
        
        # SCSE on encoder final output
        self.encoder_scse = SCSEModule(final_encoder_channels)
        
        # Probe decoder output channels and detect calling convention
        self._decoder_output_channels, self._decoder_convention = \
            self._probe_decoder_and_detect_convention(in_channels)
        
        # SCSE on PPM output (PRIMARY location)
        self.ppm_scse = SCSEModule(self._decoder_output_channels)
        
        # Store original segmentation head
        self.seg_head = self._model.segmentation_head
        
    def _probe_decoder_and_detect_convention(self, in_channels: int) -> tuple:
        """
        Probe decoder output channels and detect PSPDecoder calling convention.

        Returns:
            tuple: (decoder_output_channels: int, convention: str)
                   convention is "list", "varargs", or "last"
        """
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, 64, 64)
            features = self._model.encoder(dummy)

            for convention in ("list", "varargs", "last"):
                decoder_out, success = self._try_decoder_call(features, convention)
                if success and decoder_out.dim() == 4:
                    return decoder_out.shape[1], convention

        raise RuntimeError("Unable to determine the PSPNet decoder calling convention.")
    
    def _try_decoder_call(
        self, 
        features: List[torch.Tensor], 
        convention: str
    ) -> tuple:
        """
        Attempt to call decoder with given convention.
        
        Args:
            features: Encoder output features list
            convention: "list", "varargs", or "last"
        
        Returns:
            tuple: (output_tensor or None, success: bool)
        """
        try:
            if convention == "list":
                out = self._model.decoder(features)
            elif convention == "varargs":
                out = self._model.decoder(*features)
            elif convention == "last":
                out = self._model.decoder(features[-1])
            else:
                raise ValueError(f"Unknown decoder convention: {convention}")
            return out, True
        except (TypeError, ValueError, RuntimeError):
            return None, False
    
    def _call_decoder(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Call decoder with detected convention."""
        if self._decoder_convention == "list":
            return self._model.decoder(features)
        if self._decoder_convention == "varargs":
            return self._model.decoder(*features)
        return self._model.decoder(features[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features
        features = self._model.encoder(x)
        
        # Apply SCSE to encoder's final feature map
        features[-1] = self.encoder_scse(features[-1])
        
        # Get PPM decoder output (handles version differences)
        decoder_output = self._call_decoder(features)
        
        # Apply SCSE to PPM output (PRIMARY location)
        decoder_output = self.ppm_scse(decoder_output)
        
        # Apply segmentation head
        output = self.seg_head(decoder_output)
        
        return output


# =============================================================================
# Model Factory
# =============================================================================

def get_model(
    model_name: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Create a segmentation model by name.
    
    Args:
        model_name: One of SUPPORTED_MODELS
        encoder_name: Encoder backbone name (default: resnet34)
        encoder_weights: Pretrained weights (default: imagenet, None for random)
        in_channels: Number of input channels (default: 3 for RGB)
        classes: Number of output classes (default: 1 for binary)
    
    Returns:
        nn.Module: The requested segmentation model
    
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )
    
    if model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    elif model_name == "deeplabv3plus_scse":
        return DeepLabV3PlusWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    elif model_name == "fcn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    elif model_name == "fcn_scse":
        return FPNWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    elif model_name == "unet_scse":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type="scse",
        )
    
    elif model_name == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    
    elif model_name == "pspnet_scse":
        return PSPNetWithSCSE(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'module.' prefix from state dict keys (for DataParallel models).
    
    Args:
        state_dict: Raw state dict from checkpoint
    
    Returns:
        Cleaned state dict with 'module.' prefix removed
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v  # Remove 'module.' prefix
        else:
            cleaned[k] = v
    return cleaned


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    model_name: str,
    encoder_name: str,
    image_size: int,
    filepath: str,
) -> None:
    """
    Save model checkpoint.
    
    Checkpoint format:
    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "epoch": int,
        "best_dice": float,
        "model_name": str,      # For architecture verification on load
        "encoder_name": str,
        "image_size": int,
    }
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        best_dice: Best Dice score achieved
        model_name: Model architecture name (for loading verification)
        encoder_name: Encoder backbone name
        image_size: Input image size
        filepath: Path to save checkpoint
    """
    # Handle DataParallel wrapping
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "model_name": model_name,
        "encoder_name": encoder_name,
        "image_size": image_size,
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load weights into
        filepath: Path to checkpoint file
        strict: Whether to strictly enforce key matching
    
    Returns:
        The full checkpoint dict (contains epoch, best_dice, model_name, etc.)
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If state dict loading fails
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    
    # Extract and clean state dict
    state_dict = checkpoint["model_state_dict"]
    state_dict = clean_state_dict(state_dict)
    
    # Load into model
    model.load_state_dict(state_dict, strict=strict)
    
    return checkpoint


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

def get_fcn_scse_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Backward compatibility alias for FCN+SCSE model."""
    return get_model("fcn_scse", encoder_name, encoder_weights, in_channels, classes)


def get_deeplabv3plus_scse_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Backward compatibility alias for DeepLabV3++SCSE model."""
    return get_model("deeplabv3plus_scse", encoder_name, encoder_weights, in_channels, classes)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("model.py - Self Test")
    print("=" * 60)
    print(f"\nSupported models: {SUPPORTED_MODELS}")
    
    all_passed = True
    
    # Test all models
    for model_name in SUPPORTED_MODELS:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            # Create model without pretrained weights for speed
            model = get_model(
                model_name=model_name,
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            
            # Test forward pass
            x = torch.randn(1, 3, 512, 512)
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (1, 1, 512, 512)
            actual_shape = tuple(output.shape)
            
            if actual_shape == expected_shape:
                print(f"  ✓ Output shape: {actual_shape}")
            else:
                print(f"  ✗ Output shape mismatch: expected {expected_shape}, got {actual_shape}")
                all_passed = False
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {num_params:,}")
            
            # Special check for PSPNet+SCSE convention detection
            if model_name == "pspnet_scse":
                print(f"  PSPDecoder convention: {model._decoder_convention}")
                print(f"  Decoder output channels: {model._decoder_output_channels}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            all_passed = False
    
    # Test checkpoint save/load
    print("\n--- Testing checkpoint utilities ---")
    try:
        model = get_model("deeplabv3plus", encoder_weights=None)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            best_dice=0.8,
            model_name="deeplabv3plus",
            encoder_name="resnet34",
            image_size=512,
            filepath="/tmp/test_checkpoint.pth",
        )
        print("  ✓ Checkpoint saved")
        
        # Load
        model2 = get_model("deeplabv3plus", encoder_weights=None)
        checkpoint = load_checkpoint(model2, "/tmp/test_checkpoint.pth")
        print(f"  ✓ Checkpoint loaded (epoch={checkpoint['epoch']}, dice={checkpoint['best_dice']})")
        print(f"  ✓ model_name in checkpoint: {checkpoint['model_name']}")
        
        # Test DataParallel stripping
        dp_model = nn.DataParallel(get_model("deeplabv3plus", encoder_weights=None))
        save_checkpoint(
            model=dp_model,
            optimizer=torch.optim.Adam(dp_model.parameters()),
            epoch=5,
            best_dice=0.75,
            model_name="deeplabv3plus",
            encoder_name="resnet34",
            image_size=512,
            filepath="/tmp/test_dp_checkpoint.pth",
        )
        
        model3 = get_model("deeplabv3plus", encoder_weights=None)
        checkpoint2 = load_checkpoint(model3, "/tmp/test_dp_checkpoint.pth")
        print(f"  ✓ DataParallel checkpoint loaded (epoch={checkpoint2['epoch']})")
        
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        all_passed = False
    
    # Test backward compatibility aliases
    print("\n--- Testing backward compatibility aliases ---")
    try:
        m1 = get_fcn_scse_model(encoder_weights=None)
        m2 = get_deeplabv3plus_scse_model(encoder_weights=None)
        print("  ✓ Aliases work")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        all_passed = False
    
    # Test error handling
    print("\n--- Testing error handling ---")
    try:
        get_model("nonexistent_model")
        print("  ✗ Should have raised ValueError")
        all_passed = False
    except ValueError as e:
        print(f"  ✓ ValueError raised correctly: {e}")
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
