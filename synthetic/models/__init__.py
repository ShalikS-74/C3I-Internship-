"""Neural network models for synthetic satellite imagery generation.

This module provides the generator and discriminator architectures
for the StyleGAN-inspired paired image generation system.

Classes:
    StyleGANGenerator: Generator for paired RGB-mask synthesis.
    PatchGANDiscriminator: Discriminator for real/fake classification.

Loss functions:
    GANLoss: Composite loss with adversarial and auxiliary terms.
    generator_loss: Non-saturating generator loss.
    discriminator_loss: Non-saturating discriminator loss.
    r1_penalty: R1 gradient penalty for regularization.
"""

from .generator import (
    StyleGANGenerator,
    StyleGAN2Generator,  # Backward compatibility alias
    MappingNetwork,
    SynthesisNetwork,
)

from .discriminator import (
    PatchGANDiscriminator,
    ConditionalPatchGANDiscriminator,
)

from .losses import (
    GANLoss,
    generator_loss,
    discriminator_loss,
    r1_penalty,
    hinge_generator_loss,
    hinge_discriminator_loss,
)

__all__ = [
    # Generator
    "StyleGANGenerator",
    "StyleGAN2Generator",
    "MappingNetwork",
    "SynthesisNetwork",
    # Discriminator
    "PatchGANDiscriminator",
    "ConditionalPatchGANDiscriminator",
    # Losses
    "GANLoss",
    "generator_loss",
    "discriminator_loss",
    "r1_penalty",
    "hinge_generator_loss",
    "hinge_discriminator_loss",
]