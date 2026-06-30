"""Training loop for StyleGAN-inspired satellite imagery generation.

This module implements the training loop for the StyleGAN-inspired
generator with paired RGB-mask outputs. Includes:
- Alternating G/D updates
- R1 gradient penalty
- Checkpointing and logging
- Learning rate scheduling

Usage:
    >>> from synthetic.config import get_default_config
    >>> from synthetic.training.trainer import Trainer
    >>> 
    >>> config = get_default_config()
    >>> trainer = Trainer(config)
    >>> trainer.train(num_epochs=100)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback when the optional tensorboard package is absent."""

        def __init__(self, *_args, **_kwargs):
            pass

        def add_scalar(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

from ..config import SyntheticConfig
from ..models import StyleGANGenerator, PatchGANDiscriminator, GANLoss
from ..datasets import create_dataloader

logger = logging.getLogger(__name__)

def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with warmup and decay.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Training config with lr_warmup_steps and lr_decay_steps.
        
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step):
        # Warmup phase
        if step < config.lr_warmup_steps:
            return step / max(1, config.lr_warmup_steps)
        # Decay phase
        progress = (step - config.lr_warmup_steps) / max(1, config.lr_decay_steps)
        progress = min(1.0, progress)
        return 1.0 - (1.0 - config.lr_decay_factor) * progress
    
    return LambdaLR(optimizer, lr_lambda)

class Trainer:
    """Trainer for StyleGAN satellite imagery generation.
    
    Handles the complete training pipeline including:
    - Model initialization
    - Data loading
    - Training loop with alternating G/D updates
    - Checkpointing
    - Logging and visualization
    
    Attributes:
        config: SyntheticConfig instance.
        generator: StyleGANGenerator model.
        discriminator: PatchGANDiscriminator model.
        gan_loss: GANLoss instance.
        g_optimizer: Generator optimizer.
        d_optimizer: Discriminator optimizer.
        device: Training device.
        
    Example:
        >>> config = get_default_config()
        >>> trainer = Trainer(config)
        >>> trainer.train(num_epochs=100)
    """
    
    def __init__(
        self,
        config: SyntheticConfig,
        generator: Optional[StyleGANGenerator] = None,
        discriminator: Optional[PatchGANDiscriminator] = None,
        train_loader: Optional[DataLoader] = None,
    ):
        """Initialize the trainer.
        
        Args:
            config: SyntheticConfig instance.
            generator: Optional pre-built generator.
            discriminator: Optional pre-built discriminator.
            train_loader: Optional pre-built data loader.
        """
        self.config = config
        
        # Set device
        self.device = torch.device(config.device.device)
        logger.info(f"Training on device: {self.device}")
        
        # Initialize models
        if generator is None:
            self.generator = StyleGANGenerator(config.model)
        else:
            self.generator = generator
            
        if discriminator is None:
            self.discriminator = PatchGANDiscriminator(config.discriminator)
        else:
            self.discriminator = discriminator
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Log model sizes
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info(f"Generator parameters: {g_params:,}")
        logger.info(f"Discriminator parameters: {d_params:,}")
        
        # Initialize loss
        self.gan_loss = GANLoss(
            r1_gamma=config.training.r1_gamma,
            mask_weight=config.loss.lambda_mask,
            dice_weight=config.loss.lambda_boundary,
            perimeter_weight=0.0,
            loss_type=config.loss.loss_type,
        )
        
        # Initialize optimizers
        self.g_optimizer = Adam(
            self.generator.parameters(),
            lr=config.training.g_lr,
            betas=(config.training.beta1, config.training.beta2),
        )
        self.d_optimizer = Adam(
            self.discriminator.parameters(),
            lr=config.training.d_lr,
            betas=(config.training.beta1, config.training.beta2),
        )
        
        # Initialize schedulers
        self.g_scheduler = get_lr_scheduler(self.g_optimizer, config.training)
        self.d_scheduler = get_lr_scheduler(self.d_optimizer, config.training)
        
        # Initialize data loader
        if train_loader is None:
            self.train_loader = create_dataloader(
                config,
                use_augmentation=True,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.train_loader = train_loader
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directories
        self.output_dir = Path(config.paths.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.sample_dir = self.output_dir / "samples"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Use a private CPU RNG so monitoring is reproducible without changing
        # the random sequence used by training.
        sample_rng = torch.Generator(device="cpu")
        sample_rng.manual_seed(config.training.seed)
        self.fixed_sample_latents = torch.randn(
            16,
            config.model.latent_dim,
            generator=sample_rng,
        ).to(self.device)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))
        
        # R1 penalty frequency
        self.r1_every = config.training.r1_interval
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Run the training loop.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Dataset size: {len(self.train_loader.dataset)}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")
        
        history = {
            'g_loss': [],
            'd_loss': [],
            'lr': [],
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = self._train_epoch()
            
            # Log epoch metrics
            for key, value in epoch_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Print progress
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"G_loss: {epoch_metrics['g_loss']:.4f} | "
                f"D_loss: {epoch_metrics['d_loss']:.4f} | "
                f"LR: {epoch_metrics['lr']:.6f} | "
                f"Time: {elapsed / 60:.1f}m"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")
            
            # Generate samples
            if (epoch + 1) % self.config.experiment.save_samples_every == 0:
                self._generate_samples(f"epoch_{epoch + 1}")
        
        # Save final checkpoint
        self._save_checkpoint("final")
        
        logger.info(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")
        
        return history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of epoch metrics.
        """
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0
        
        for batch_idx, (real_rgb, real_mask) in enumerate(self.train_loader):
            real_rgb = real_rgb.to(self.device)
            real_mask = real_mask.to(self.device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.d_optimizer.zero_grad()
            
            # Get fake samples (detached for D update)
            with torch.no_grad():
                z = torch.randn(
                    real_rgb.shape[0], self.config.model.latent_dim,
                    device=self.device
                )
                fake_rgb_logits, fake_mask_logits = self.generator(z)
                fake_rgb = torch.sigmoid(fake_rgb_logits)
                fake_mask = torch.sigmoid(fake_mask_logits)
            
            # Discriminator predictions
            real_logits = self.discriminator(real_rgb, real_mask)
            fake_logits = self.discriminator(fake_rgb, fake_mask)
            
            # D loss with R1 penalty
            if self.global_step % self.r1_every == 0:
                d_loss, d_loss_dict = self.gan_loss.discriminator_loss(
                    real_logits, fake_logits,
                    real_rgb, real_mask, self.discriminator
                )
            else:
                d_loss, d_loss_dict = self.gan_loss.discriminator_loss(
                    real_logits, fake_logits
                )
            
            d_loss.backward()
            self.d_optimizer.step()
            self.d_scheduler.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            self.g_optimizer.zero_grad()
            
            # Generate samples
            z = torch.randn(
                real_rgb.shape[0], self.config.model.latent_dim,
                device=self.device
            )
            fake_rgb_logits, fake_mask_logits = self.generator(z)
            fake_rgb = torch.sigmoid(fake_rgb_logits)
            fake_mask_sigmoid = torch.sigmoid(fake_mask_logits)
            
            # G loss
            fake_logits = self.discriminator(fake_rgb, fake_mask_sigmoid)
            g_loss, g_loss_dict = self.gan_loss.generator_loss(fake_logits)
            
            g_loss.backward()
            self.g_optimizer.step()
            self.g_scheduler.step()
            
            # Accumulate losses
            total_g_loss += g_loss_dict['total']
            total_d_loss += d_loss_dict['total']
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('loss/generator', g_loss_dict['total'], self.global_step)
                self.writer.add_scalar('loss/discriminator', d_loss_dict['total'], self.global_step)
                self.writer.add_scalar('lr/g', self.g_optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('lr/d', self.d_optimizer.param_groups[0]['lr'], self.global_step)
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches,
            'lr': self.g_optimizer.param_groups[0]['lr'],
        }
    
    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint.
        
        Args:
            name: Checkpoint name.
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'fixed_sample_latents': self.fixed_sample_latents.cpu(),
            'config': self.config.to_dict(),
        }
        
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _generate_samples(self, name: str, num_samples: int = 16) -> None:
        """Generate a fixed-latent contact sheet for diversity monitoring.
        
        Args:
            name: Sample name prefix.
            num_samples: Number of fixed samples to include, up to 16.
        """
        if not 1 <= num_samples <= len(self.fixed_sample_latents):
            raise ValueError(
                f"num_samples must be between 1 and "
                f"{len(self.fixed_sample_latents)}, got {num_samples}"
            )

        was_training = self.generator.training
        self.generator.eval()

        try:
            rgb_batches = []
            mask_batches = []
            sample_batch_size = min(self.config.training.batch_size, num_samples)
            with torch.no_grad():
                for start in range(0, num_samples, sample_batch_size):
                    z = self.fixed_sample_latents[
                        start:min(start + sample_batch_size, num_samples)
                    ]
                    rgb_logits, mask_logits = self.generator(z)
                    rgb_batches.append(torch.sigmoid(rgb_logits).cpu())
                    mask_batches.append((torch.sigmoid(mask_logits) >= 0.5).cpu())

            rgb = torch.cat(rgb_batches)
            mask = torch.cat(mask_batches)

            tiles = []
            for index in range(num_samples):
                rgb_np = (
                    rgb[index].permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                mask_np = (
                    mask[index, 0].numpy().astype(np.uint8) * 255
                )
                mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
                tiles.append(np.concatenate([rgb_np, mask_rgb], axis=1))

            columns = 4
            tile_height, tile_width = tiles[0].shape[:2]
            rows = (num_samples + columns - 1) // columns
            grid = np.zeros(
                (rows * tile_height, columns * tile_width, 3),
                dtype=np.uint8,
            )
            for index, tile in enumerate(tiles):
                row, column = divmod(index, columns)
                grid[
                    row * tile_height:(row + 1) * tile_height,
                    column * tile_width:(column + 1) * tile_width,
                ] = tile

            sample_path = self.sample_dir / f"{name}_fixed_grid.png"
            if not cv2.imwrite(str(sample_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Failed to save sample grid: {sample_path}")
        finally:
            if was_training:
                self.generator.train()

        logger.info(f"Saved fixed-latent sample grid: {sample_path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        required_keys = {
            'generator_state_dict',
            'discriminator_state_dict',
            'g_optimizer_state_dict',
            'd_optimizer_state_dict',
            'epoch',
            'global_step',
        }
        missing_keys = required_keys - set(checkpoint)
        if missing_keys:
            raise KeyError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        if 'fixed_sample_latents' in checkpoint:
            fixed_latents = checkpoint['fixed_sample_latents']
            expected_shape = tuple(self.fixed_sample_latents.shape)
            if tuple(fixed_latents.shape) != expected_shape:
                raise ValueError(
                    "Checkpoint fixed_sample_latents shape mismatch: "
                    f"expected {expected_shape}, got {tuple(fixed_latents.shape)}"
                )
            self.fixed_sample_latents = fixed_latents.to(self.device)
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")

def train_from_config(config_path: str) -> None:
    """Train from a config file.
    
    Args:
        config_path: Path to config JSON file.
    """
    with open(config_path) as f:
        config_dict = json.load(f)
    
    config = SyntheticConfig.from_dict(config_dict)
    trainer = Trainer(config)
    trainer.train(num_epochs=config.training.num_epochs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the StyleGAN-inspired satellite generator")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.config:
        train_from_config(args.config)
    else:
        from ..config import get_default_config
        config = get_default_config()
        config.training.num_epochs = args.epochs
        trainer = Trainer(config)
        trainer.train(num_epochs=args.epochs)
