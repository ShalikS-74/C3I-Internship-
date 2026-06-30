"""Loss functions for StyleGAN training.

This module implements loss functions for adversarial training including:
- Non-saturating logistic loss (standard GAN loss)
- R1 gradient penalty (for discriminator regularization)
- Auxiliary losses for mask quality

Tensor shape conventions:
    logits: [B, 1, H, W] - discriminator output
    loss: scalar tensor

Usage:
    >>> from synthetic.models.losses import (
    ...     generator_loss, discriminator_loss, r1_penalty
    ... )
    >>> 
    >>> # Generator loss
    >>> loss_G = generator_loss(fake_logits)
    >>> 
    >>> # Discriminator loss
    >>> loss_D = discriminator_loss(real_logits, fake_logits)
    >>> 
    >>> # R1 regularization
    >>> r1 = r1_penalty(real_rgb, real_mask, discriminator)
"""

import torch
import torch.nn.functional as F
from typing import Callable, Tuple, Optional

# =============================================================================
# ADVERSARIAL LOSSES
# =============================================================================

def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss.
    
    Uses softplus for numerical stability:
        loss = softplus(-fake_logits).mean()
    
    This encourages the discriminator to output positive values for fake samples.
    
    Args:
        fake_logits: Discriminator output for generated samples [B, 1, H, W].
        
    Returns:
        Scalar generator loss.
    """
    return F.softplus(-fake_logits).mean()

def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Non-saturating discriminator loss.
    
    loss = softplus(-real_logits) + softplus(fake_logits)
    
    This encourages:
        - Real samples → positive logits
        - Fake samples → negative logits
    
    Args:
        real_logits: Discriminator output for real samples [B, 1, H, W].
        fake_logits: Discriminator output for fake samples [B, 1, H, W].
        
    Returns:
        Scalar discriminator loss.
    """
    real_loss = F.softplus(-real_logits)
    fake_loss = F.softplus(fake_logits)
    return (real_loss + fake_loss).mean()

def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator.
    
    loss = -fake_logits.mean()
    
    Args:
        fake_logits: Discriminator output for fake samples.
        
    Returns:
        Scalar loss.
    """
    return -fake_logits.mean()

def hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss for discriminator.
    
    loss = relu(1 - real_logits) + relu(1 + fake_logits)
    
    Args:
        real_logits: Discriminator output for real samples.
        fake_logits: Discriminator output for fake samples.
        
    Returns:
        Scalar loss.
    """
    real_loss = F.relu(1.0 - real_logits)
    fake_loss = F.relu(1.0 + fake_logits)
    return (real_loss + fake_loss).mean()

# =============================================================================
# GRADIENT PENALTY
# =============================================================================

def r1_penalty(
    real_rgb: torch.Tensor,
    real_mask: torch.Tensor,
    discriminator: Callable,
    gamma: float = 10.0,
) -> torch.Tensor:
    """R1 gradient penalty for discriminator regularization.
    
    Penalizes gradients of the discriminator w.r.t. real samples:
        penalty = gamma/2 * ||grad_D(real)||^2
    
    This prevents the discriminator from becoming too confident and
    improves training stability.
    
    Args:
        real_rgb: Real RGB images [B, 3, H, W]. Must require_grad.
        real_mask: Real masks [B, 1, H, W]. Must require_grad.
        discriminator: Discriminator function that takes (rgb, mask).
        gamma: Penalty coefficient. Default 10.0 from StyleGAN2.
        
    Returns:
        Scalar R1 penalty.
        
    Note:
        Input tensors should have requires_grad=True for gradient computation.
    """
    # Enable gradients for real samples
    real_rgb = real_rgb.requires_grad_(True)
    real_mask = real_mask.requires_grad_(True)
    
    # Get discriminator output
    real_logits = discriminator(real_rgb, real_mask)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=[real_rgb, real_mask],
        create_graph=True,
        retain_graph=True,
    )
    
    # Compute the squared gradient norm per sample, then average the batch.
    grad_rgb, grad_mask = gradients
    batch_size = real_rgb.shape[0]
    grad_norm_sq = grad_rgb.square().reshape(batch_size, -1).sum(dim=1)
    grad_norm_sq += grad_mask.square().reshape(batch_size, -1).sum(dim=1)
    penalty = (gamma / 2) * grad_norm_sq.mean()
    
    return penalty

def r1_penalty_v2(
    real_rgb: torch.Tensor,
    real_mask: torch.Tensor,
    discriminator: Callable,
    gamma: float = 10.0,
) -> torch.Tensor:
    """R1 gradient penalty (alternative implementation).
    
    This version computes the gradient penalty per sample in the batch,
    which can be more numerically stable for large batches.
    
    Args:
        real_rgb: Real RGB images [B, 3, H, W].
        real_mask: Real masks [B, 1, H, W].
        discriminator: Discriminator function.
        gamma: Penalty coefficient.
        
    Returns:
        Scalar R1 penalty (mean over batch).
    """
    real_rgb = real_rgb.requires_grad_(True)
    real_mask = real_mask.requires_grad_(True)
    
    real_logits = discriminator(real_rgb, real_mask)
    
    # Compute gradients per sample
    batch_size = real_rgb.shape[0]
    gradients = torch.autograd.grad(
        outputs=real_logits,
        inputs=[real_rgb, real_mask],
        grad_outputs=torch.ones_like(real_logits),
        create_graph=True,
    )
    
    grad_rgb, grad_mask = gradients
    
    # Sum of squared gradients per sample
    grad_norm_sq = (grad_rgb.view(batch_size, -1) ** 2).sum(dim=1)
    grad_norm_sq = grad_norm_sq + (grad_mask.view(batch_size, -1) ** 2).sum(dim=1)
    
    # Mean over batch
    penalty = (gamma / 2) * grad_norm_sq.mean()
    
    return penalty

# =============================================================================
# AUXILIARY LOSSES
# =============================================================================

def mask_consistency_loss(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy loss for mask consistency.
    
    This loss encourages the generated mask to match the target mask
    when training with paired data.
    
    Args:
        mask_logits: Generated mask logits [B, 1, H, W].
        target_mask: Target mask values [B, 1, H, W] in {0, 1}.
        weight: Loss weight.
        
    Returns:
        Scalar BCE loss.
    """
    bce = F.binary_cross_entropy_with_logits(
        mask_logits, target_mask, reduction='mean'
    )
    return weight * bce

def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice loss for mask quality.
    
    More robust to class imbalance than BCE.
    
    Args:
        pred: Predicted mask probabilities [B, 1, H, W] in [0, 1].
        target: Target mask [B, 1, H, W] in {0, 1}.
        smooth: Smoothing constant to avoid division by zero.
        
    Returns:
        Scalar Dice loss.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice

def perimeter_loss(
    mask: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Perimeter loss for mask boundary quality.
    
    Encourages the boundaries of generated masks to match targets.
    
    Args:
        mask: Generated mask probabilities [B, 1, H, W].
        target: Target mask [B, 1, H, W].
        kernel_size: Sobel kernel size.
        
    Returns:
        Scalar perimeter loss.
    """
    # Sobel kernels for edge detection
    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3, 3)
    
    # Compute edges
    mask_edges_x = F.conv2d(mask, sobel_x, padding=1)
    mask_edges_y = F.conv2d(mask, sobel_y, padding=1)
    mask_edges = torch.sqrt(mask_edges_x ** 2 + mask_edges_y ** 2 + 1e-8)
    
    target_edges_x = F.conv2d(target, sobel_x, padding=1)
    target_edges_y = F.conv2d(target, sobel_y, padding=1)
    target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-8)
    
    # L1 loss on edges
    return F.l1_loss(mask_edges, target_edges)

# =============================================================================
# COMPOSITE LOSS
# =============================================================================

class GANLoss:
    """Composite GAN loss with multiple components.
    
    Combines adversarial loss with optional auxiliary losses:
        - R1 gradient penalty
        - Mask consistency loss
        - Dice loss
        - Perimeter loss
    
    Example:
        >>> gan_loss = GANLoss(config.loss)
        >>> 
        >>> # Generator update
        >>> fake_rgb, fake_mask_logits = generator(z)
        >>> fake_logits = discriminator(fake_rgb, fake_mask)
        >>> loss_G = gan_loss.generator_loss(
        ...     fake_logits, fake_mask_logits, real_mask
        ... )
        >>> 
        >>> # Discriminator update
        >>> loss_D, r1 = gan_loss.discriminator_loss(
        ...     real_logits, fake_logits, real_rgb, real_mask, discriminator
        ... )
    """
    
    def __init__(
        self,
        r1_gamma: float = 10.0,
        mask_weight: float = 0.0,
        dice_weight: float = 0.0,
        perimeter_weight: float = 0.0,
        loss_type: str = "non_saturating",
    ):
        """Initialize GAN loss.
        
        Args:
            r1_gamma: R1 penalty coefficient.
            mask_weight: Weight for mask BCE loss.
            dice_weight: Weight for dice loss.
            perimeter_weight: Weight for perimeter loss.
            loss_type: "non_saturating" or "hinge".
        """
        self.r1_gamma = r1_gamma
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.perimeter_weight = perimeter_weight
        self.loss_type = loss_type
    
    def generator_loss(
        self,
        fake_logits: torch.Tensor,
        mask_logits: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute generator loss with optional auxiliary losses.
        
        Args:
            fake_logits: Discriminator output for fake samples.
            mask_logits: Generated mask logits (for auxiliary losses).
            target_mask: Target mask (for auxiliary losses).
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Adversarial loss
        if self.loss_type == "hinge":
            loss_adv = hinge_generator_loss(fake_logits)
        else:
            loss_adv = generator_loss(fake_logits)
        
        losses = {"adversarial": loss_adv.item()}
        total_loss = loss_adv
        
        # Auxiliary losses
        if mask_logits is not None and target_mask is not None:
            if self.mask_weight > 0:
                loss_mask = mask_consistency_loss(mask_logits, target_mask, self.mask_weight)
                total_loss = total_loss + loss_mask
                losses["mask_bce"] = loss_mask.item()
            
            if self.dice_weight > 0:
                mask_prob = torch.sigmoid(mask_logits)
                loss_dice = dice_loss(mask_prob, target_mask) * self.dice_weight
                total_loss = total_loss + loss_dice
                losses["dice"] = loss_dice.item()
            
            if self.perimeter_weight > 0:
                mask_prob = torch.sigmoid(mask_logits)
                loss_perim = perimeter_loss(mask_prob, target_mask) * self.perimeter_weight
                total_loss = total_loss + loss_perim
                losses["perimeter"] = loss_perim.item()
        
        losses["total"] = total_loss.item()
        return total_loss, losses
    
    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        real_rgb: Optional[torch.Tensor] = None,
        real_mask: Optional[torch.Tensor] = None,
        discriminator: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute discriminator loss with R1 penalty.
        
        Args:
            real_logits: Discriminator output for real samples.
            fake_logits: Discriminator output for fake samples.
            real_rgb: Real RGB images (for R1 penalty).
            real_mask: Real masks (for R1 penalty).
            discriminator: Discriminator function (for R1 penalty).
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Adversarial loss
        if self.loss_type == "hinge":
            loss_adv = hinge_discriminator_loss(real_logits, fake_logits)
        else:
            loss_adv = discriminator_loss(real_logits, fake_logits)
        
        losses = {"adversarial": loss_adv.item()}
        total_loss = loss_adv
        
        # R1 penalty
        if (self.r1_gamma > 0 and 
            real_rgb is not None and 
            real_mask is not None and 
            discriminator is not None):
            
            # Only apply R1 periodically to save compute
            r1 = r1_penalty(real_rgb, real_mask, discriminator, self.r1_gamma)
            total_loss = total_loss + r1
            losses["r1"] = r1.item()
        
        losses["total"] = total_loss.item()
        return total_loss, losses
