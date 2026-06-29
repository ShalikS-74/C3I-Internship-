"""Train the StyleGAN-inspired satellite imagery generator.

Main entry point for training the StyleGAN-inspired generator
for paired RGB-mask satellite imagery.

Usage:
    # Train with default config
    python -m synthetic.train_stylegan

    # Train with custom config
    python -m synthetic.train_stylegan --config config.json

    # Resume from checkpoint
    python -m synthetic.train_stylegan --resume outputs/checkpoints/epoch_50.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from .config import SyntheticConfig, get_default_config
from .training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train the StyleGAN-inspired satellite imagery generator"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            config = SyntheticConfig.from_dict(json.load(f))
    else:
        config = get_default_config()
    
    # Apply command line overrides
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.g_lr = args.lr
        config.training.d_lr = args.lr
    if args.device is not None:
        config.device.device = args.device
    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device.device}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"G LR: {config.training.g_lr}")
    logger.info(f"D LR: {config.training.d_lr}")
    logger.info(f"Image size: {config.model.image_size}")
    logger.info(f"Latent dim: {config.model.latent_dim}")
    logger.info(f"Output dir: {config.paths.output_dir}")
    logger.info("=" * 60)
    
    # Set random seed
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        logger.info(f"Random seed set to {config.training.seed}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Run training
    try:
        history = trainer.train(num_epochs=config.training.num_epochs)
        logger.info("Training completed successfully!")
        
        # Save final config
        config_path = Path(config.paths.output_dir) / "config.json"
        config.to_json(config_path)
        logger.info(f"Saved final config to {config_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        trainer._save_checkpoint("error")
        raise

if __name__ == "__main__":
    main()
