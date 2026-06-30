"""PatchGAN discriminator for paired satellite RGB and mask discrimination.

This module implements a PatchGAN-style discriminator that takes concatenated
(RGB, mask) pairs and outputs patch-level real/fake predictions. Uses spectral
normalization for training stability.

Tensor shape flow:
    Input:
        rgb: [B, 3, 512, 512]
        mask: [B, 1, 512, 512]
        concatenated: [B, 4, 512, 512]
    
    Downsampling:
        [B, 64, 256, 256]
        [B, 128, 128, 128]
        [B, 256, 64, 64]
        [B, 512, 32, 32]
    
    Output:
        logits: [B, 1, 31, 31] (patch predictions, 70x70 receptive field)

Usage:
    >>> from synthetic.config import get_default_config
    >>> from synthetic.models.discriminator import PatchGANDiscriminator
    >>> 
    >>> config = get_default_config()
    >>> discriminator = PatchGANDiscriminator(config.discriminator)
    >>> 
    >>> rgb = torch.randn(4, 3, 512, 512)
    >>> mask = torch.randn(4, 1, 512, 512)
    >>> logits = discriminator(rgb, mask)
    >>> print(logits.shape)
    torch.Size([4, 1, 31, 31])
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DiscriminatorConfig

logger = logging.getLogger(__name__)

# =============================================================================
# HELPER MODULES
# =============================================================================

class DiscriminatorBlock(nn.Module):
    """Single downsampling block for discriminator.
    
    Structure:
        Conv2d (4x4, stride=2, padding=1) with optional spectral norm
        LeakyReLU
    
    Input: [B, in_channels, H, W]
    Output: [B, out_channels, H//2, W//2]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
    ):
        """Initialize DiscriminatorBlock.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=4, stride=2, padding=1
        )
        
        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(conv)
        else:
            self.conv = conv
        
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        # He initialization for LeakyReLU
        nn.init.kaiming_normal_(
            self.conv.weight, 
            a=0.2, 
            mode='fan_in', 
            nonlinearity='leaky_relu'
        )
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling block.
        
        Args:
            x: Input feature map [B, C_in, H, W].
            
        Returns:
            Output feature map [B, C_out, H//2, W//2].
        """
        return self.activation(self.conv(x))


class MinibatchStdDev(nn.Module):
    """Append a channel describing variation across the current minibatch.

    The discriminator can use this signal to identify batches in which the
    generator produces nearly identical samples. The layer has no trainable
    parameters and supports any batch size, including a final batch of one.
    """

    def __init__(self, group_size: int = 4, eps: float = 1e-8):
        super().__init__()
        if group_size < 1:
            raise ValueError("group_size must be at least 1")
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate one minibatch standard-deviation feature channel."""
        batch_size, channels, height, width = x.shape
        group_size = min(self.group_size, batch_size)
        if batch_size % group_size != 0:
            group_size = batch_size

        grouped = x.reshape(group_size, -1, channels, height, width)
        stddev = torch.sqrt(grouped.var(dim=0, unbiased=False) + self.eps)
        stddev = stddev.mean(dim=(1, 2, 3), keepdim=True)
        stddev = stddev.repeat(group_size, 1, height, width)

        return torch.cat([x, stddev], dim=1)

class FinalBlock(nn.Module):
    """Final classification block for discriminator.
    
    Structure:
        Conv2d (4x4, stride=1, padding=1)
        LeakyReLU
        Conv2d (4x4, stride=1, padding=1) → logits
    
    Input: [B, in_channels, H, W]
    Output: [B, 1, H-3, W-3] (patch logits)
    """
    
    def __init__(
        self,
        in_channels: int,
        use_spectral_norm: bool = True,
    ):
        """Initialize FinalBlock.
        
        Args:
            in_channels: Input channels.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        conv1 = nn.Conv2d(in_channels, in_channels, 4, stride=1, padding=1)
        conv2 = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1)
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(conv1)
            self.conv2 = nn.utils.spectral_norm(conv2)
        else:
            self.conv1 = conv1
            self.conv2 = conv2
        
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        nn.init.kaiming_normal_(
            self.conv1.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu'
        )
        nn.init.kaiming_normal_(
            self.conv2.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu'
        )
        if hasattr(self.conv1, 'bias') and self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if hasattr(self.conv2, 'bias') and self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final classification block.
        
        Args:
            x: Input feature map [B, C, H, W].
            
        Returns:
            Patch logits [B, 1, H-3, W-3].
        """
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x

# =============================================================================
# MAIN DISCRIMINATOR
# =============================================================================

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for paired (RGB, mask) discrimination.
    
    Takes concatenated RGB and mask tensors and outputs patch-level
    real/fake predictions. Uses spectral normalization for stability.
    
    Architecture:
        concat(rgb, mask): [B, 4, 512, 512]
            ↓ DiscriminatorBlock (512→256)
        [B, 64, 256, 256]
            ↓ DiscriminatorBlock
        [B, 128, 128, 128]
            ↓ DiscriminatorBlock
        [B, 256, 64, 64]
            ↓ DiscriminatorBlock
        [B, 512, 32, 32]
            ↓ FinalBlock
        [B, 1, 31, 31] (patch logits)
    
    Each output patch corresponds to a 70×70 receptive field in the input.
    The output can be interpreted as:
        - Positive values → more likely real
        - Negative values → more likely fake
    
    For GAN loss, use:
        - Non-saturating logistic loss
        - R1 gradient penalty on real samples
    
    Attributes:
        config: DiscriminatorConfig instance.
        blocks: List of downsampling blocks.
        final: Final classification block.
    
    Example:
        >>> discriminator = PatchGANDiscriminator(config.discriminator)
        >>> rgb = torch.randn(4, 3, 512, 512)
        >>> mask = torch.randn(4, 1, 512, 512)
        >>> logits = discriminator(rgb, mask)
        >>> # For GAN loss:
        >>> loss = F.softplus(-logits).mean()  # Generator loss
        >>> loss = F.softplus(logits).mean()   # Discriminator loss on fake
    """
    
    def __init__(self, config: Optional[DiscriminatorConfig] = None):
        """Initialize the discriminator.
        
        Args:
            config: DiscriminatorConfig instance. Uses defaults if None.
        """
        super().__init__()
        
        if config is None:
            config = DiscriminatorConfig()
        
        self.config = config
        
        # Build downsampling blocks
        # Input: [B, 4, 512, 512] (RGB + mask)
        # Output: [B, 512, 32, 32]
        
        blocks = []
        in_channels = config.in_channels  # 4 (RGB + mask)
        
        for i in range(config.num_layers):
            out_channels = config.base_channels * config.get_channel_multiplier(i)
            blocks.append(
                DiscriminatorBlock(
                    in_channels, out_channels, 
                    use_spectral_norm=config.use_spectral_norm
                )
            )
            in_channels = out_channels
        
        self.blocks = nn.ModuleList(blocks)

        self.minibatch_stddev = MinibatchStdDev(group_size=4)

        # Final classification block
        self.final = FinalBlock(in_channels + 1, config.use_spectral_norm)
        
        logger.info(
            f"PatchGANDiscriminator initialized: "
            f"in_channels={config.in_channels}, "
            f"base_channels={config.base_channels}, "
            f"num_layers={config.num_layers}, "
            f"spectral_norm={config.use_spectral_norm}"
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminate RGB-mask pairs.
        
        Args:
            rgb: RGB image tensor [B, 3, H, W].
            mask: Mask tensor [B, 1, H, W].
            
        Returns:
            Patch logits [B, 1, H', W'] where H', W' depend on input size.
            For 512x512 input: [B, 1, 31, 31]
            
        Raises:
            ValueError: If input shapes are incompatible.
        """
        # Validate shapes
        if rgb.dim() != 4 or mask.dim() != 4:
            raise ValueError(
                f"Expected 4D tensors (B, C, H, W), got "
                f"rgb.dim()={rgb.dim()}, mask.dim()={mask.dim()}"
            )
        
        if rgb.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: rgb={rgb.shape[0]}, mask={mask.shape[0]}"
            )
        
        if rgb.shape[2:] != mask.shape[2:]:
            raise ValueError(
                f"Spatial size mismatch: rgb={rgb.shape[2:]}, mask={mask.shape[2:]}"
            )
        
        # Concatenate RGB and mask
        # rgb: [B, 3, H, W], mask: [B, 1, H, W] → [B, 4, H, W]
        x = torch.cat([rgb, mask], dim=1)
        
        # Apply downsampling blocks
        for block in self.blocks:
            x = block(x)

        # Give the discriminator an explicit signal when a batch lacks variety.
        x = self.minibatch_stddev(x)
        
        # Apply final classification
        logits = self.final(x)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total parameters.
        
        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

# =============================================================================
# ALTERNATIVE: CONDITIONAL DISCRIMINATOR (OPTIONAL)
# =============================================================================

class ConditionalPatchGANDiscriminator(nn.Module):
    """Conditional discriminator that can condition on additional information.
    
    This is an optional extension that allows conditioning the discriminator
    on additional information (e.g., building count, coverage) if needed
    for future experiments.
    
    Structure is similar to PatchGANDiscriminator but with projection
    of conditioning information.
    """
    
    def __init__(
        self,
        config: Optional[DiscriminatorConfig] = None,
        condition_dim: int = 0,
    ):
        """Initialize conditional discriminator.
        
        Args:
            config: DiscriminatorConfig instance.
            condition_dim: Dimension of conditioning vector (0 to disable).
        """
        super().__init__()
        
        if config is None:
            config = DiscriminatorConfig()
        
        self.config = config
        self.condition_dim = condition_dim
        
        # Main discriminator (same as PatchGANDiscriminator)
        blocks = []
        in_channels = config.in_channels
        
        for i in range(config.num_layers):
            out_channels = config.base_channels * config.get_channel_multiplier(i)
            blocks.append(
                DiscriminatorBlock(
                    in_channels, out_channels,
                    use_spectral_norm=config.use_spectral_norm
                )
            )
            in_channels = out_channels
        
        self.blocks = nn.ModuleList(blocks)
        self.final = FinalBlock(in_channels, config.use_spectral_norm)
        
        # Condition projection (if used)
        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, in_channels)
        else:
            self.condition_proj = None
    
    def forward(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Discriminate with optional conditioning.
        
        Args:
            rgb: RGB image tensor [B, 3, H, W].
            mask: Mask tensor [B, 1, H, W].
            condition: Optional conditioning vector [B, condition_dim].
            
        Returns:
            Patch logits [B, 1, H', W'].
        """
        x = torch.cat([rgb, mask], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        # Apply condition if provided
        if condition is not None and self.condition_proj is not None:
            cond_proj = self.condition_proj(condition)
            # Add condition to spatial dimensions
            x = x + cond_proj.unsqueeze(2).unsqueeze(3)
        
        logits = self.final(x)
        return logits
