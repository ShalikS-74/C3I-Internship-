"""Generate synthetic satellite imagery dataset.

This script generates a synthetic dataset of (RGB, mask) pairs using
a trained StyleGAN generator. Applies quality filtering to ensure
generated samples are suitable for training augmentation.

Usage:
    # Generate from trained model
    python -m synthetic.generate_dataset \
        --checkpoint outputs/checkpoints/final.pt \
        --output_dir outputs/synthetic_dataset \
        --num_samples 10000

    # Generate with quality filtering
    python -m synthetic.generate_dataset \
        --checkpoint outputs/checkpoints/final.pt \
        --output_dir outputs/synthetic_dataset \
        --num_samples 10000 \
        --quality_filter
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .config import SyntheticConfig, get_default_config
from .models import StyleGANGenerator
from .evaluation.quality_filter import QualityFilter

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate synthetic dataset from trained generator.
    
    Handles:
    - Loading trained generator from checkpoint
    - Generating samples in batches
    - Quality filtering
    - Saving to disk in dataset format
    
    Attributes:
        generator: Trained StyleGANGenerator.
        config: SyntheticConfig instance.
        device: Generation device.
        quality_filter: Optional QualityFilter instance.
        
    Example:
        >>> generator = DatasetGenerator.from_checkpoint("checkpoint.pt")
        >>> generator.generate(
        ...     output_dir="synthetic_dataset",
        ...     num_samples=10000,
        ...     use_quality_filter=True
        ... )
    """
    
    def __init__(
        self,
        generator: StyleGANGenerator,
        config: SyntheticConfig,
        device: Optional[torch.device] = None,
        quality_filter: Optional[QualityFilter] = None,
    ):
        """Initialize dataset generator.
        
        Args:
            generator: Trained StyleGANGenerator model.
            config: SyntheticConfig instance.
            device: Generation device. Defaults to config.device.device.
            quality_filter: Optional QualityFilter for filtering samples.
        """
        self.generator = generator
        self.config = config
        self.device = device or torch.device(config.device.device)
        self.quality_filter = quality_filter
        
        self.generator.to(self.device)
        self.generator.eval()
        
        logger.info(f"DatasetGenerator initialized on {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[SyntheticConfig] = None,
        device: Optional[str] = None,
        quality_filter: Optional[QualityFilter] = None,
    ) -> 'DatasetGenerator':
        """Create generator from checkpoint file.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file.
            config: Optional config. Loaded from checkpoint if not provided.
            device: Device string (e.g., "cuda:0", "cpu").
            quality_filter: Optional QualityFilter.
            
        Returns:
            DatasetGenerator instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must contain a dictionary.")
        if 'generator_state_dict' not in checkpoint:
            raise KeyError("Checkpoint is missing 'generator_state_dict'.")
        
        # Load or use provided config
        if config is None:
            if 'config' in checkpoint:
                config = SyntheticConfig.from_dict(checkpoint['config'])
            else:
                config = get_default_config()
        
        # Set device
        if device:
            config.device.device = device
        target_device = torch.device(config.device.device)
        
        # Build generator
        generator = StyleGANGenerator(config.model)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        logger.info(f"Loaded generator from {checkpoint_path}")
        
        return cls(
            generator=generator,
            config=config,
            device=target_device,
            quality_filter=quality_filter,
        )
    
    def generate(
        self,
        output_dir: str,
        num_samples: int,
        batch_size: int = 16,
        use_quality_filter: bool = True,
        save_metadata: bool = True,
        seed: Optional[int] = None,
        latent_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate synthetic dataset.
        
        Args:
            output_dir: Output directory for generated samples.
            num_samples: Number of samples to generate (before filtering).
            batch_size: Batch size for generation.
            use_quality_filter: Whether to apply quality filtering.
            save_metadata: Whether to save generation metadata.
            seed: Random seed for reproducibility.
            latent_dim: Latent dimension (uses config if not provided).
            
        Returns:
            Generation statistics dictionary.
        """
        # Setup
        output_dir = Path(output_dir)
        image_dir = output_dir / "image"
        mask_dir = output_dir / "label"
        
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        latent_dim = latent_dim or self.config.model.latent_dim
        
        # Statistics
        total_generated = 0
        total_saved = 0
        rejected_samples = 0
        generation_times = []
        
        logger.info(f"Generating {num_samples} samples to {output_dir}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Quality filtering: {use_quality_filter}")
        
        start_time = time.time()
        
        with torch.no_grad():
            with tqdm(total=num_samples, desc="Generating") as pbar:
                while total_saved < num_samples:
                    # Generate batch
                    batch_start = time.time()
                    
                    z = torch.randn(batch_size, latent_dim, device=self.device)
                    rgb_logits, mask_logits = self.generator(z)
                    rgb = torch.sigmoid(rgb_logits)
                    mask = torch.sigmoid(mask_logits)
                    
                    # Convert to numpy
                    rgb_np = rgb.cpu().permute(0, 2, 3, 1).numpy()  # [B, H, W, 3]
                    mask_np = mask.cpu().squeeze(1).numpy()  # [B, H, W]
                    
                    generation_times.append(time.time() - batch_start)
                    
                    # Filter and save each sample
                    for i in range(batch_size):
                        total_generated += 1
                        
                        sample_rgb = rgb_np[i]
                        sample_mask = mask_np[i]
                        
                        # Quality filter
                        if use_quality_filter and self.quality_filter:
                            # Convert to tensor for filter
                            rgb_tensor = torch.from_numpy(sample_rgb).permute(2, 0, 1).unsqueeze(0)
                            mask_tensor = torch.from_numpy(sample_mask).unsqueeze(0).unsqueeze(0)
                            
                            passed, metrics = self.quality_filter.filter(rgb_tensor, mask_tensor)
                            
                            if not passed:
                                rejected_samples += 1
                                continue
                        
                        # Save sample
                        sample_id = f"{total_saved:06d}"
                        
                        # Save RGB (BGR for cv2)
                        rgb_bgr = (sample_rgb * 255).astype(np.uint8)[..., ::-1]
                        cv2.imwrite(str(image_dir / f"{sample_id}.png"), rgb_bgr)
                        
                        # Save mask
                        mask_uint8 = (sample_mask * 255).astype(np.uint8)
                        cv2.imwrite(str(mask_dir / f"{sample_id}.png"), mask_uint8)
                        
                        total_saved += 1
                        
                        if total_saved >= num_samples:
                            break
                    
                    pbar.update(min(batch_size, num_samples - total_saved + rejected_samples))
                    pbar.set_postfix({
                        'saved': total_saved,
                        'rejected': rejected_samples,
                        'pass_rate': f"{total_saved/(total_generated or 1):.1%}"
                    })
        
        elapsed = time.time() - start_time
        
        # Statistics
        stats = {
            'num_requested': num_samples,
            'num_generated': total_generated,
            'num_saved': total_saved,
            'num_rejected': rejected_samples,
            'pass_rate': total_saved / (total_generated or 1),
            'generation_time': elapsed,
            'samples_per_second': total_saved / elapsed,
            'avg_batch_time': np.mean(generation_times),
            'config': self.config.to_dict(),
        }
        
        # Save metadata
        if save_metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info(
            f"Generation complete: {total_saved} samples saved, "
            f"{rejected_samples} rejected ({rejected_samples/(total_generated or 1):.1%}), "
            f"{elapsed/60:.1f} minutes"
        )
        
        return stats

def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic satellite imagery dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to generator checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/synthetic_dataset",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_quality_filter", action="store_true",
        help="Disable quality filtering"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (uses checkpoint config if not provided)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = SyntheticConfig.from_dict(json.load(f))
    
    # Override device
    if args.device:
        if config is None:
            config = get_default_config()
        config.device.device = args.device
    
    # Create quality filter
    quality_filter = None
    if not args.no_quality_filter:
        quality_filter = QualityFilter(config.quality_filter if config else None)
    
    # Create generator from checkpoint
    generator = DatasetGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
        quality_filter=quality_filter,
    )
    
    # Generate dataset
    generator.generate(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_quality_filter=not args.no_quality_filter,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
