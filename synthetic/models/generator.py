"""StyleGAN-inspired generator for paired satellite RGB and mask generation.

IMPORTANT: This is NOT a full StyleGAN2 implementation.

This generator is inspired by StyleGAN but intentionally simplified for
paired RGB-mask satellite imagery generation. Key differences from StyleGAN2:

1. Uses AdaIN instead of weight modulation/demodulation (simpler, stable)
2. Includes noise injection for stochastic detail
3. No progressive growing (trains directly at 512x512)
4. Dual output heads for paired (RGB, mask) generation

For a production StyleGAN2 implementation, see the official NVIDIA code:
https://github.com/NVlabs/stylegan2-ada-pytorch

Tensor shape flow:
    z: [B, 512] (random latent)
        ↓ mapping network (8 FC layers with pixel_norm)
    w: [B, 512] (style vector)
        ↓ synthesis network (progressive upsampling)
    features: [B, 512, 4, 4] → [B, 32, 512, 512]
        ↓ dual heads (separate for RGB and mask)
    rgb_logits: [B, 3, 512, 512] (apply sigmoid for [0, 1])
    mask_logits: [B, 1, 512, 512] (apply sigmoid for probability)

Usage:
    >>> from synthetic.config import get_default_config
    >>> from synthetic.models.generator import StyleGANGenerator
    >>> 
    >>> config = get_default_config()
    >>> generator = StyleGANGenerator(config.model)
    >>> 
    >>> z = torch.randn(4, 512)  # Batch of 4
    >>> rgb_logits, mask_logits = generator(z)
    >>> rgb = torch.sigmoid(rgb_logits)  # [0, 1]
    >>> mask_prob = torch.sigmoid(mask_logits)
"""

import logging
import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum channels at highest resolution (don't go below this)
MIN_FINAL_CHANNELS = 32

# =============================================================================
# HELPER MODULES
# =============================================================================

class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization.
    
    Normalizes each feature vector independently to unit length.
    Used in the mapping network to prevent magnitude explosion.
    
    Input shape: [B, C] or [B, C, H, W]
    Output shape: Same as input
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """Initialize PixelNorm.
        
        Args:
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each feature vector.
        
        Args:
            x: Input tensor of shape [B, C, ...].
            
        Returns:
            Normalized tensor with same shape.
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class NoiseInjection(nn.Module):
    """Injects learned noise into feature maps.
    
    A key component of StyleGAN for stochastic variation.
    Each spatial position gets a different random perturbation.
    
    Input:
        x: [B, C, H, W] feature map
        noise: [B, 1, H, W] or None (will be generated)
        
    Output: [B, C, H, W] noisy feature map
    """
    
    def __init__(self, channels: int):
        """Initialize NoiseInjection.
        
        Args:
            channels: Number of feature channels (unused but kept for API consistency).
        """
        super().__init__()
        # Learnable noise scaling factor per channel
        self.noise_scale = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add scaled noise to feature map.
        
        Args:
            x: Feature map [B, C, H, W].
            noise: Optional noise tensor [B, 1, H, W]. Generated if None.
            
        Returns:
            Feature map with added noise [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype
            )
        
        return x + noise * self.noise_scale

class AdaIN(nn.Module):
    """Adaptive Instance Normalization.
    
    Applies instance normalization followed by style-based affine
    transformation. This is the core of StyleGAN's style injection.
    
    Note: This is StyleGAN v1 style. StyleGAN2 uses weight modulation
    instead, but AdaIN is simpler and more stable for initial experiments.
    
    Input:
        x: [B, C, H, W] feature map
        w: [B, style_dim] style vector
        
    Output: [B, C, H, W] stylized feature map
    """
    
    def __init__(
        self, 
        channels: int, 
        style_dim: int = 512,
        epsilon: float = 1e-8
    ):
        """Initialize AdaIN.
        
        Args:
            channels: Number of feature channels.
            style_dim: Dimension of style vector w.
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        
        self.channels = channels
        self.style_dim = style_dim
        self.epsilon = epsilon
        
        # Learnable scale and bias from style vector
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)
        
        # Initialize for identity-ish transform
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize near identity while retaining latent dependence."""
        nn.init.normal_(self.style_scale.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.style_scale.bias)
        nn.init.normal_(self.style_bias.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.style_bias.bias)
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization.
        
        Args:
            x: Feature map [B, C, H, W].
            w: Style vector [B, style_dim].
            
        Returns:
            Stylized feature map [B, C, H, W].
        """
        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True) + self.epsilon  # [B, C, 1, 1]
        x_norm = (x - mean) / std  # [B, C, H, W]
        
        # Style modulation
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        return x_norm * scale + bias

class StyledConv(nn.Module):
    """Convolution with style injection and noise.
    
    Structure:
        Conv2d (kernel_size, stride=1, padding=kernel_size//2)
        NoiseInjection
        AdaIN
        LeakyReLU
    
    Input:
        x: [B, in_channels, H, W]
        w: [B, style_dim]
        noise: [B, 1, H, W] (optional)
        
    Output: [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        style_dim: int = 512,
        use_noise: bool = True,
    ):
        """Initialize StyledConv.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            style_dim: Style dimension.
            use_noise: Whether to inject noise.
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2
        )
        
        self.use_noise = use_noise
        if use_noise:
            self.noise = NoiseInjection(out_channels)
        
        self.adain = AdaIN(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize convolution weights with He initialization."""
        nn.init.kaiming_normal_(
            self.conv.weight, 
            a=0.2, 
            mode='fan_in', 
            nonlinearity='leaky_relu'
        )
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply styled convolution.
        
        Args:
            x: Input feature map [B, C_in, H, W].
            w: Style vector [B, style_dim].
            noise: Optional noise tensor.
            
        Returns:
            Output feature map [B, C_out, H, W].
        """
        x = self.conv(x)
        
        if self.use_noise:
            x = self.noise(x, noise)
        
        x = self.adain(x, w)
        x = self.activation(x)
        
        return x

class UpsampleBlock(nn.Module):
    """Upsampling block with style injection and noise.
    
    Structure:
        Nearest neighbor upsample (2x)
        StyledConv 1 (3x3)
        StyledConv 2 (3x3)
    
    Input:
        x: [B, C_in, H, W]
        w: [B, style_dim]
        
    Output: [B, C_out, H*2, W*2]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int = 512,
        use_noise: bool = True,
    ):
        """Initialize UpsampleBlock.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            style_dim: Style dimension.
            use_noise: Whether to inject noise.
        """
        super().__init__()
        
        self.conv1 = StyledConv(in_channels, out_channels, 3, style_dim, use_noise)
        self.conv2 = StyledConv(out_channels, out_channels, 3, style_dim, use_noise)
    
    def forward(
        self, 
        x: torch.Tensor, 
        w: torch.Tensor
    ) -> torch.Tensor:
        """Upsample and apply convolutions.
        
        Args:
            x: Input feature map [B, C_in, H, W].
            w: Style vector [B, style_dim].
            
        Returns:
            Upsampled feature map [B, C_out, H*2, W*2].
        """
        # Nearest neighbor upsample
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Two styled conv blocks
        x = self.conv1(x, w)
        x = self.conv2(x, w)
        
        return x

# =============================================================================
# MAPPING NETWORK
# =============================================================================

class MappingNetwork(nn.Module):
    """Maps latent z to style w.
    
    Structure:
        PixelNorm
        FC (512 → 512) × num_layers
        LeakyReLU after each FC
    
    Input: z [B, latent_dim]
    Output: w [B, latent_dim]
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 8,
    ):
        """Initialize MappingNetwork.
        
        Args:
            latent_dim: Dimension of latent vectors.
            num_layers: Number of FC layers.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        layers = []
        layers.append(PixelNorm())
        
        for _ in range(num_layers):
            fc = nn.Linear(latent_dim, latent_dim)
            nn.init.kaiming_normal_(
                fc.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu'
            )
            nn.init.zeros_(fc.bias)
            layers.append(fc)
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent z to style w.
        
        Args:
            z: Latent vector [B, latent_dim].
            
        Returns:
            Style vector [B, latent_dim].
        """
        return self.layers(z)

# =============================================================================
# SYNTHESIS NETWORK
# =============================================================================

class SynthesisNetwork(nn.Module):
    """Generates feature maps from style vectors.
    
    Progressive upsampling from 4×4 to target resolution.
    
    Structure (for 512×512):
        Constant [B, C, 4, 4]
        ↓ StyledConv (4×4)
        ↓ UpsampleBlock (4→8)
        ↓ UpsampleBlock (8→16)
        ↓ UpsampleBlock (16→32)
        ↓ UpsampleBlock (32→64)
        ↓ UpsampleBlock (64→128)
        ↓ UpsampleBlock (128→256)
        ↓ UpsampleBlock (256→512)
        → features [B, min_channels, 512, 512]
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 512,
        base_channels: int = 512,
        min_channels: int = 32,  # Don't go below 32 channels
    ):
        """Initialize SynthesisNetwork.
        
        Args:
            latent_dim: Dimension of style vector.
            image_size: Target image resolution.
            base_channels: Channels at lowest resolution (4x4).
            min_channels: Minimum channels at highest resolution.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels
        self.min_channels = max(min_channels, MIN_FINAL_CHANNELS)
        
        # Calculate number of resolution doublings from 4x4
        self.num_blocks = int(math.log2(image_size)) - 2  # 4→512 = 7 blocks
        
        # Learnable constant input
        self.constant = nn.Parameter(
            torch.randn(1, base_channels, 4, 4) * 0.1
        )
        
        # Initial styled conv at 4×4
        channels_4x4 = base_channels
        self.conv_4x4 = StyledConv(base_channels, channels_4x4, 3, latent_dim)
        
        # Build upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        
        in_channels = channels_4x4
        for i in range(self.num_blocks):
            # Calculate output channels
            # Halve channels with each resolution doubling, but respect minimum
            ideal_channels = base_channels // (2 ** (i + 1))
            out_channels = max(self.min_channels, ideal_channels)
            
            self.upsample_blocks.append(
                UpsampleBlock(in_channels, out_channels, latent_dim)
            )
            
            in_channels = out_channels
        
        # Final output channels
        self.output_channels = in_channels
        
        logger.info(
            f"SynthesisNetwork: {self.num_blocks + 1} blocks, "
            f"4×4 → {image_size}×{image_size}, "
            f"output channels: {self.output_channels}"
        )
    
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Generate feature maps from style vectors.
        
        Args:
            w: Style vector [B, latent_dim].
            
        Returns:
            Feature map [B, C, H, W] where H=W=image_size.
        """
        batch_size = w.shape[0]
        
        # Expand constant to batch size
        x = self.constant.expand(batch_size, -1, -1, -1)
        
        # Initial 4×4 conv
        x = self.conv_4x4(x, w)
        
        # Progressive upsampling
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x, w)
        
        return x

# =============================================================================
# DUAL OUTPUT HEADS
# =============================================================================

class OutputHead(nn.Module):
    """Output head for generating RGB or mask.
    
    Converts features to output with proper initialization.
    
    Input: [B, in_channels, H, W]
    Output: [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize OutputHead.
        
        Args:
            in_channels: Input feature channels.
            out_channels: Output channels (3 for RGB, 1 for mask).
        """
        super().__init__()
        
        hidden_channels = max(in_channels // 2, 16)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        for module in self.layers:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output from features.
        
        Args:
            x: Feature map [B, C_in, H, W].
            
        Returns:
            Output [B, C_out, H, W].
        """
        return self.layers(x)

# =============================================================================
# MAIN GENERATOR
# =============================================================================

class StyleGANGenerator(nn.Module):
    """StyleGAN-inspired Generator for paired satellite RGB and mask generation.
    
    This generator produces both an RGB satellite image and a corresponding
    building mask from a random latent vector. The two outputs share the
    same synthesis network but have separate output heads.
    
    Architecture:
        z [B, 512] → MappingNetwork → w [B, 512]
        w → SynthesisNetwork → features [B, C, 512, 512]
        features → RGB head → rgb_logits [B, 3, 512, 512]
        features → Mask head → mask_logits [B, 1, 512, 512]
    
    The outputs are raw logits. Apply sigmoid to get:
        rgb: [0, 1] range
        mask_prob: [0, 1] probability
    
    Note: This is a simplified StyleGAN architecture, not full StyleGAN2.
    Uses AdaIN for style injection (simpler, more stable) with noise injection
    for stochastic detail.
    
    Attributes:
        config: ModelConfig instance.
        mapping: MappingNetwork.
        synthesis: SynthesisNetwork.
        rgb_head: Output head for RGB generation.
        mask_head: Output head for mask generation.
    
    Example:
        >>> generator = StyleGANGenerator(config.model)
        >>> z = torch.randn(4, 512)
        >>> rgb_logits, mask_logits = generator(z)
        >>> rgb = torch.sigmoid(rgb_logits)  # [0, 1]
        >>> mask_prob = torch.sigmoid(mask_logits)  # probability
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the generator.
        
        Args:
            config: ModelConfig instance. Uses defaults if None.
        """
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        # Mapping network: z → w
        self.mapping = MappingNetwork(
            latent_dim=config.latent_dim,
            num_layers=config.num_mapping_layers,
        )
        
        # Synthesis network: w → features
        # Note: min_channels is set to 32 for sufficient capacity
        self.synthesis = SynthesisNetwork(
            latent_dim=config.latent_dim,
            image_size=config.image_size,
            base_channels=config.base_channels,
            min_channels=32,  # Don't go below 32 channels
        )
        
        # Output heads: features → (RGB, mask)
        self.rgb_head = OutputHead(
            in_channels=self.synthesis.output_channels,
            out_channels=config.in_channels,  # 3 for RGB
        )
        
        self.mask_head = OutputHead(
            in_channels=self.synthesis.output_channels,
            out_channels=config.out_channels_mask,  # 1 for mask
        )
        
        logger.info(
            f"StyleGANGenerator initialized: "
            f"latent_dim={config.latent_dim}, "
            f"image_size={config.image_size}, "
            f"output_channels=({config.in_channels}, {config.out_channels_mask}), "
            f"final_features={self.synthesis.output_channels}"
        )
    
    def forward(
        self,
        z: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate paired RGB and mask from latent.
        
        Args:
            z: Latent vector [B, latent_dim]. Ignored if w is provided.
            w: Pre-computed style vector [B, latent_dim]. Optional.
            
        Returns:
            Tuple of (rgb_logits, mask_logits):
                rgb_logits: [B, 3, H, W] - apply sigmoid for [0, 1]
                mask_logits: [B, 1, H, W] - apply sigmoid for probability
        """
        # Get style vector
        if w is None:
            w = self.mapping(z)
        
        # Generate features
        features = self.synthesis(w)  # [B, C, H, W]
        
        # Generate outputs through separate heads
        rgb_logits = self.rgb_head(features)  # [B, 3, H, W]
        mask_logits = self.mask_head(features)  # [B, 1, H, W]
        
        return rgb_logits, mask_logits
    
    def generate(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate RGB and mask with sigmoid applied.
        
        This is a convenience method for inference.
        
        Args:
            z: Latent vector [B, latent_dim].
            
        Returns:
            Tuple of (rgb, mask_prob):
                rgb: [B, 3, H, W] in [0, 1]
                mask_prob: [B, 1, H, W] in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            rgb_logits, mask_logits = self(z)
            rgb = torch.sigmoid(rgb_logits)
            mask_prob = torch.sigmoid(mask_logits)
        return rgb, mask_prob
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component.
        
        Returns:
            Dictionary with parameter counts.
        """
        return {
            'mapping': sum(p.numel() for p in self.mapping.parameters()),
            'synthesis': sum(p.numel() for p in self.synthesis.parameters()),
            'rgb_head': sum(p.numel() for p in self.rgb_head.parameters()),
            'mask_head': sum(p.numel() for p in self.mask_head.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }

# Backward compatibility alias
StyleGAN2Generator = StyleGANGenerator
