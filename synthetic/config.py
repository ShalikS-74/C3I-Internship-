"""Configuration module for StyleGAN synthetic data generation.

This module defines all hyperparameters, paths, and constants used throughout
the synthetic generation pipeline. All values are centralized here to ensure
configuration-driven design without hardcoded paths or magic numbers.

Tensor dimension conventions:
    - B: Batch size
    - C: Channels
    - H: Height
    - W: Width
    
Example flow:
    latent z: [B, 512]
        ↓ mapping network
    w: [B, 512]
        ↓ synthesis network
    features: [B, C, H, W] where H=W=512
        ↓ dual heads
    rgb: [B, 3, 512, 512]
    mask: [B, 1, 512, 512]
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import json

# =============================================================================
# CONSTANTS
# =============================================================================

# Supported image sizes (can be extended for debugging)
SUPPORTED_IMAGE_SIZES = {256, 512}

# Default dataset subdirectory patterns (can be overridden)
DEFAULT_DATASET_PATTERNS = {
    'image_dirs': [
        "1. The cropped image data and raster labels/train/image",
        "1. The cropped image data and raster labels/image",
        "image",
        "images",
    ],
    'mask_dirs': [
        "1. The cropped image data and raster labels/train/label",
        "1. The cropped image data and raster labels/label",
        "label",
        "masks",
    ],
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root directory.
        
    Raises:
        RuntimeError: If project root cannot be determined.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    if not (project_root / "dataset.py").exists():
        raise RuntimeError(
            f"Cannot find project root. Expected dataset.py at {project_root}. "
            f"Current file: {current_file}"
        )
    
    return project_root

@dataclass
class PathConfig:
    """Path configuration for synthetic data generation.
    
    All paths are relative to the project root and are resolved at runtime.
    This ensures portability across different systems.
    
    Important: Directories are NOT created during initialization to avoid
    side effects on import. Call ensure_directories() explicitly when needed.
    
    Attributes:
        project_root: Root directory of the project.
        synthetic_dir: Directory containing synthetic module.
        checkpoint_dir: Directory for saving model checkpoints.
        output_dir: Directory for generated synthetic datasets.
        log_dir: Directory for training logs.
        real_data_dir: Directory containing real satellite imagery.
        dataset_image_patterns: List of possible image subdirectory patterns.
        dataset_mask_patterns: List of possible mask subdirectory patterns.
    """
    
    project_root: Path = field(default_factory=get_project_root)
    synthetic_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    
    # Checkpoint paths
    checkpoint_dir: Path = field(
        default_factory=lambda: get_project_root() / "synthetic" / "checkpoints"
    )
    
    # Output paths
    output_dir: Path = field(
        default_factory=lambda: get_project_root() / "synthetic" / "outputs"
    )
    
    # Log paths
    log_dir: Path = field(
        default_factory=lambda: get_project_root() / "synthetic" / "logs"
    )
    
    # Real data path (can be overridden via CLI/config file)
    real_data_dir: Path = field(
        default_factory=lambda: get_project_root() / "dataset" / "Satellite dataset Ⅱ (East Asia)"
    )
    
    # Dataset subdirectory patterns (allow override without code changes)
    dataset_image_patterns: List[str] = field(
        default_factory=lambda: DEFAULT_DATASET_PATTERNS['image_dirs'].copy()
    )
    dataset_mask_patterns: List[str] = field(
        default_factory=lambda: DEFAULT_DATASET_PATTERNS['mask_dirs'].copy()
    )
    
    def __post_init__(self) -> None:
        """Convert paths to Path objects after initialization."""
        self.project_root = Path(self.project_root)
        self.synthetic_dir = Path(self.synthetic_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        self.real_data_dir = Path(self.real_data_dir)
    
    def ensure_directories(self) -> None:
        """Create directories if they don't exist.
        
        This method must be called explicitly to avoid side effects on import.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_real_image_dir(self) -> Path:
        """Get directory containing real satellite images.
        
        Returns:
            Path to image directory.
            
        Raises:
            FileNotFoundError: If image directory doesn't exist.
        """
        for pattern in self.dataset_image_patterns:
            path = self.real_data_dir / pattern
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"Cannot find image directory in {self.real_data_dir}. "
            f"Tried patterns: {self.dataset_image_patterns}"
        )
    
    def get_real_mask_dir(self) -> Path:
        """Get directory containing real building masks.
        
        Returns:
            Path to mask directory.
            
        Raises:
            FileNotFoundError: If mask directory doesn't exist.
        """
        for pattern in self.dataset_mask_patterns:
            path = self.real_data_dir / pattern
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"Cannot find mask directory in {self.real_data_dir}. "
            f"Tried patterns: {self.dataset_mask_patterns}"
        )

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Experiment tracking and organization configuration.
    
    This allows organizing outputs by experiment/run, making it easier
    to compare different training configurations.
    
    Output structure:
        outputs/
            {experiment_name}/
                {run_name}/
                    checkpoints/
                    logs/
                    samples/
    
    Attributes:
        experiment_name: Name of the experiment.
        run_name: Name of this specific run.
        save_predictions: Whether to save generated predictions.
        save_samples_every: Save sample visualizations every N epochs.
        track_metrics: Whether to track detailed metrics.
    """
    
    experiment_name: str = "stylegan_baseline"
    run_name: str = "run001"
    save_predictions: bool = True
    save_samples_every: int = 5
    track_metrics: bool = True
    
    def get_output_dir(self, base_dir: Path) -> Path:
        """Get the full output directory for this experiment run.
        
        Args:
            base_dir: Base output directory.
            
        Returns:
            Path to experiment run directory.
        """
        return base_dir / self.experiment_name / self.run_name
    
    def get_checkpoint_dir(self, base_dir: Path) -> Path:
        """Get checkpoint directory for this run.
        
        Args:
            base_dir: Base output directory.
            
        Returns:
            Path to checkpoint directory.
        """
        return self.get_output_dir(base_dir) / "checkpoints"
    
    def get_log_dir(self, base_dir: Path) -> Path:
        """Get log directory for this run.
        
        Args:
            base_dir: Base output directory.
            
        Returns:
            Path to log directory.
        """
        return self.get_output_dir(base_dir) / "logs"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Defines the StyleGAN-inspired generator with dual output heads
    for paired RGB and mask generation.
    
    Tensor shape progression through synthesis network:
        [B, 512] (latent w)
            ↓
        [B, 512, 4, 4] (initial features)
            ↓ upsample blocks
        [B, 256, 8, 8]
        [B, 128, 16, 16]
        [B, 64, 32, 32]
        [B, 32, 64, 64]
        [B, 16, 128, 128]
        [B, 8, 256, 256]
        [B, 4, 512, 512]
            ↓ dual heads
        rgb: [B, 3, 512, 512]
        mask: [B, 1, 512, 512]
    
    Attributes:
        latent_dim: Dimension of latent vector z and w.
        num_mapping_layers: Number of FC layers in mapping network.
        image_size: Output image resolution (H = W).
        in_channels: Number of input channels for RGB images.
        out_channels_mask: Number of output channels for mask.
        base_channels: Base number of channels in synthesis network.
        max_channels: Maximum channels at lowest resolution.
        min_channels: Minimum channels at highest resolution.
        use_style_mixing: Whether to use style mixing regularization.
        style_mixing_prob: Probability of applying style mixing.
        truncation_psi: Truncation parameter (1.0 = no truncation).
    """
    
    # Latent space
    latent_dim: int = 512
    num_mapping_layers: int = 8
    
    # Output dimensions
    image_size: int = 512
    in_channels: int = 3  # RGB
    out_channels_mask: int = 1  # Binary mask
    
    # Channel dimensions (per resolution level)
    base_channels: int = 512  # Channels at 4x4 resolution
    max_channels: int = 512   # Maximum channels
    min_channels: int = 4     # Minimum channels at 512x512
    
    # Architecture choices
    use_style_mixing: bool = True
    style_mixing_prob: float = 0.9
    truncation_psi: float = 1.0  # 1.0 = no truncation
    
    def get_channels_at_resolution(self, resolution: int) -> int:
        """Calculate number of channels at a given resolution.
        
        Args:
            resolution: Spatial resolution (H or W, assuming square).
            
        Returns:
            Number of channels at that resolution.
            
        Raises:
            ValueError: If resolution is not a power of 2 or out of range.
        """
        if resolution < 4 or resolution > self.image_size:
            raise ValueError(
                f"Resolution {resolution} out of range [4, {self.image_size}]"
            )
        
        # Check if power of 2
        if resolution & (resolution - 1) != 0:
            raise ValueError(f"Resolution {resolution} must be a power of 2")
        
        # Channel halving with each resolution doubling
        log_resolution = resolution.bit_length() - 1
        log_base = (4).bit_length() - 1  # log2(4) = 2
        
        channels = self.base_channels // (2 ** (log_resolution - log_base - 1))
        return max(self.min_channels, min(self.max_channels, channels))
    
    def get_resolution_blocks(self) -> List[Tuple[int, int]]:
        """Get list of (resolution, channels) for each synthesis block.
        
        Returns:
            List of (resolution, channels) tuples from 4x4 to image_size.
        """
        blocks = []
        res = 4
        while res <= self.image_size:
            channels = self.get_channels_at_resolution(res)
            blocks.append((res, channels))
            res *= 2
        return blocks

# =============================================================================
# DISCRIMINATOR CONFIGURATION
# =============================================================================

@dataclass
class DiscriminatorConfig:
    """Discriminator architecture configuration.
    
    PatchGAN-style discriminator for 70x70 receptive field.
    Takes concatenated (RGB, mask) as input: [B, 4, H, W]
    
    Tensor shape progression:
        input: [B, 4, 512, 512] (RGB + mask concatenated)
            ↓
        [B, 64, 256, 256]
        [B, 128, 128, 128]
        [B, 256, 64, 64]
        [B, 512, 32, 32]
        [B, 1, 31, 31] (patch outputs)
    
    Attributes:
        in_channels: Input channels (RGB + mask = 4).
        base_channels: Base channel count.
        num_layers: Number of downsampling layers.
        use_spectral_norm: Whether to use spectral normalization.
    """
    
    in_channels: int = 4  # RGB (3) + Mask (1)
    base_channels: int = 64
    num_layers: int = 4
    use_spectral_norm: bool = True
    
    def get_channel_multiplier(self, layer_idx: int) -> int:
        """Get channel multiplier for a given layer.
        
        Args:
            layer_idx: Index of the layer (0-indexed).
            
        Returns:
            Channel multiplier (1, 2, 4, 8, ...).
        """
        return min(2 ** layer_idx, 8)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters.
    
    Uses a StyleGAN-inspired training recipe with TTUR.
    Learning rates are conservative for initial training stability.
    
    Attributes:
        batch_size: Training batch size.
        num_epochs: Total training epochs.
        d_lr: Discriminator learning rate.
        g_lr: Generator learning rate (lower per TTUR).
        beta1: Adam beta1.
        beta2: Adam beta2.
        r1_gamma: R1 gradient penalty weight.
        r1_interval: Apply R1 every N iterations.
        path_reg_weight: Path length regularization weight.
        path_reg_interval: Apply path reg every N iterations.
        warmup_iterations: Warmup period before regularization.
        save_interval: Save checkpoint every N epochs.
        keep_last_n: Keep last N checkpoints.
        use_amp: Use automatic mixed precision.
        seed: Random seed for reproducibility.
    """
    
    # Batch and epochs
    batch_size: int = 4  # Conservative for 512x512
    num_epochs: int = 100
    
    # Learning rates (conservative for stability, TTUR)
    # Using 10x lower than initial draft for safety
    d_lr: float = 4e-4  # Discriminator learns faster (TTUR)
    g_lr: float = 2e-4  # Generator learns slower
    beta1: float = 0.0
    beta2: float = 0.999

    # Learning-rate schedule
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    lr_decay_factor: float = 0.1
    
    # Gradient penalty (R1)
    r1_gamma: float = 10.0
    r1_interval: int = 16
    
    # Path length regularization
    path_reg_weight: float = 2.0
    path_reg_interval: int = 4
    warmup_iterations: int = 5000
    
    # Checkpointing
    save_interval: int = 1  # Save every epoch
    keep_last_n: int = 5  # Keep last N checkpoints
    
    # Mixed precision
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def get_optimizers(
        self, 
        g_params: Any, 
        d_params: Any
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Create optimizers for generator and discriminator.
        
        Args:
            g_params: Generator parameters.
            d_params: Discriminator parameters.
            
        Returns:
            Tuple of (g_optimizer, d_optimizer).
        """
        g_optim = torch.optim.Adam(
            g_params,
            lr=self.g_lr,
            betas=(self.beta1, self.beta2)
        )
        d_optim = torch.optim.Adam(
            d_params,
            lr=self.d_lr,
            betas=(self.beta1, self.beta2)
        )
        return g_optim, d_optim

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================

@dataclass
class LossConfig:
    """Loss function weights and configuration.
    
    Total generator loss:
        L_G = λ_adv * L_adv_G 
            + λ_mask * L_mask_bce
            + λ_boundary * L_boundary_dice
            + λ_feature * L_feature_consistency
            + λ_perceptual * L_perceptual_vgg
    
    Attributes:
        lambda_adv: Adversarial loss weight.
        lambda_mask: Mask BCE loss weight.
        lambda_boundary: Boundary Dice loss weight.
        lambda_feature: Feature consistency loss weight.
        lambda_perceptual: Perceptual (VGG) loss weight.
        boundary_radius: Dilation radius for boundary computation.
        perceptual_layers: VGG layer indices for perceptual loss.
    """
    
    # Loss weights
    lambda_adv: float = 1.0
    lambda_mask: float = 10.0
    lambda_boundary: float = 1.0
    lambda_feature: float = 0.5
    lambda_perceptual: float = 10.0
    loss_type: str = "non_saturating"
    
    # Boundary loss parameters
    boundary_radius: int = 2  # For boundary IoU dilation
    
    # Perceptual loss layers (VGG layer indices)
    perceptual_layers: List[int] = field(default_factory=lambda: [4, 9, 16, 23, 30])

# =============================================================================
# QUALITY FILTER CONFIGURATION
# =============================================================================

@dataclass
class QualityFilterConfig:
    """Quality filtering parameters for synthetic samples.
    
    Samples are rejected if they fail ANY of these criteria:
        - Empty mask (no buildings)
        - Excessive coverage (>max_coverage% building)
        - Impossible building count
        - Disconnected artifacts (tiny buildings)
        - Poor RGB-mask alignment
    
    IMPORTANT: Thresholds should be derived from real data statistics.
    Use compute_real_data_statistics() to get data-driven values instead
    of using these hardcoded defaults.
    
    Attributes:
        min_building_pixels: Minimum pixels for non-empty mask.
        max_coverage: Maximum building coverage ratio.
        min_building_count: Minimum number of buildings.
        max_building_count: Maximum number of buildings.
        min_building_area: Minimum area per building (pixels).
        min_alignment_score: Minimum RGB-mask alignment.
        area_ks_threshold: KS test p-value threshold for area distribution.
        use_data_driven_thresholds: Whether to use computed statistics.
    """
    
    # Coverage limits (conservative defaults)
    min_building_pixels: int = 100
    max_coverage: float = 0.90
    
    # Building count limits
    min_building_count: int = 1
    max_building_count: int = 200  # Should be computed from real data
    
    # Building size limits
    min_building_area: int = 20
    
    # Alignment threshold
    min_alignment_score: float = 0.50
    
    # Distribution validation
    area_ks_threshold: float = 0.01
    
    # Flag to indicate if thresholds are data-driven
    use_data_driven_thresholds: bool = False
    
    def validate_config(self) -> None:
        """Validate configuration values.
        
        Raises:
            ValueError: If any value is out of valid range.
        """
        if self.min_building_pixels < 0:
            raise ValueError("min_building_pixels must be non-negative")
        if not 0 < self.max_coverage <= 1:
            raise ValueError("max_coverage must be in (0, 1]")
        if self.min_building_count < 0:
            raise ValueError("min_building_count must be non-negative")
        if self.max_building_count <= self.min_building_count:
            raise ValueError("max_building_count must be > min_building_count")
        if self.min_building_area < 1:
            raise ValueError("min_building_area must be >= 1")
        if not 0 <= self.min_alignment_score <= 1:
            raise ValueError("min_alignment_score must be in [0, 1]")
    
    def update_from_statistics(
        self,
        max_coverage_percentile: float = 99.0,
        max_count_percentile: float = 99.0,
    ) -> None:
        """Mark that thresholds were updated from data statistics.
        
        This is a placeholder for the actual statistics computation,
        which should be implemented in evaluation/quality_metrics.py.
        
        Args:
            max_coverage_percentile: Percentile for max_coverage threshold.
            max_count_percentile: Percentile for max_building_count threshold.
        """
        self.use_data_driven_thresholds = True
        # Actual values should be set by the caller

# =============================================================================
# GENERATION CONFIGURATION
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for synthetic dataset generation.
    
    Output structure:
        synthetic_dataset/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        ├── test/
        │   ├── images/
        │   └── masks/
        └── metadata.csv
    
    Attributes:
        train_count: Number of training samples to generate.
        val_count: Number of validation samples.
        test_count: Number of test samples.
        output_name: Name of output directory.
        seed: Random seed for reproducibility.
        batch_size: Generation batch size.
        generate_extra_ratio: Generate extra to account for rejections.
    """
    
    train_count: int = 5000
    val_count: int = 500
    test_count: int = 500
    
    output_name: str = "synthetic_dataset"
    seed: int = 42
    batch_size: int = 16
    generate_extra_ratio: float = 1.3  # Generate 30% extra for filtering
    
    def get_total_counts(self) -> Dict[str, int]:
        """Get total samples to generate per split (with extra for filtering).
        
        Returns:
            Dictionary with 'train', 'val', 'test' counts.
        """
        return {
            'train': int(self.train_count * self.generate_extra_ratio),
            'val': int(self.val_count * self.generate_extra_ratio),
            'test': int(self.test_count * self.generate_extra_ratio),
        }

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

@dataclass
class DeviceConfig:
    """Device configuration with automatic CUDA detection.
    
    Attributes:
        device: Target device ('cuda', 'cpu', or 'auto').
        num_workers: Number of data loader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
    """
    
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self) -> None:
        """Resolve device after initialization."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_device(self) -> torch.device:
        """Get the resolved PyTorch device.
        
        Returns:
            torch.device instance.
        """
        return torch.device(self.device)
    
    def is_cuda(self) -> bool:
        """Check if CUDA is available and selected.
        
        Returns:
            True if using CUDA.
        """
        return self.device == "cuda" and torch.cuda.is_available()

# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class SyntheticConfig:
    """Master configuration combining all sub-configurations.
    
    This is the main configuration class that should be used throughout
    the pipeline. All sub-configurations are accessible as attributes.
    
    Example:
        >>> config = get_default_config()
        >>> print(config.model.image_size)
        512
        >>> print(config.training.batch_size)
        4
    """
    
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    def validate(self) -> None:
        """Validate all sub-configurations.
        
        Raises:
            ValueError: If any configuration is invalid.
        """
        self.quality_filter.validate_config()
        
        # Validate image size is supported
        if self.model.image_size not in SUPPORTED_IMAGE_SIZES:
            raise ValueError(
                f"image_size {self.model.image_size} not supported. "
                f"Supported sizes: {SUPPORTED_IMAGE_SIZES}"
            )
    
    def ensure_directories(self) -> None:
        """Create necessary directories.
        
        Call this before training/generation to ensure directories exist.
        """
        self.paths.ensure_directories()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (complete serialization).
        
        Uses dataclasses.asdict for full serialization of all fields.
        
        Returns:
            Dictionary representation of all configurations.
        """
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: convert_paths(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_paths(value) for value in obj]
            if isinstance(obj, tuple):
                return [convert_paths(value) for value in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return convert_paths(asdict(self))
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize configuration to JSON.
        
        Args:
            path: Optional path to save JSON file.
            
        Returns:
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if path is not None:
            path = Path(path)
            path.write_text(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            SyntheticConfig instance.
        """
        # Convert string paths back to Path objects
        if 'paths' in data:
            paths_data = data['paths']
            for key in ['project_root', 'synthetic_dir', 'checkpoint_dir', 
                       'output_dir', 'log_dir', 'real_data_dir']:
                if key in paths_data and isinstance(paths_data[key], str):
                    paths_data[key] = Path(paths_data[key])
        
        return cls(
            paths=PathConfig(**data.get('paths', {})),
            experiment=ExperimentConfig(**data.get('experiment', {})),
            model=ModelConfig(**data.get('model', {})),
            discriminator=DiscriminatorConfig(**data.get('discriminator', {})),
            training=TrainingConfig(**data.get('training', {})),
            loss=LossConfig(**data.get('loss', {})),
            quality_filter=QualityFilterConfig(**data.get('quality_filter', {})),
            generation=GenerationConfig(**data.get('generation', {})),
            device=DeviceConfig(**data.get('device', {})),
        )

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_default_config() -> SyntheticConfig:
    """Get default configuration instance.
    
    This function creates a fresh configuration instance each time,
    avoiding the side effects of a module-level singleton.
    
    Returns:
        New SyntheticConfig instance with default values.
    """
    return SyntheticConfig()

def get_config(
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> SyntheticConfig:
    """Get configuration with optional overrides.
    
    Args:
        seed: Override random seed.
        batch_size: Override batch size.
        num_epochs: Override number of epochs.
        experiment_name: Override experiment name.
        run_name: Override run name.
        
    Returns:
        Configuration instance with overrides applied.
    """
    config = get_default_config()
    
    if seed is not None:
        config.training.seed = seed
        config.generation.seed = seed
    
    if batch_size is not None:
        config.training.batch_size = batch_size
        config.generation.batch_size = batch_size
    
    if num_epochs is not None:
        config.training.num_epochs = num_epochs
    
    if experiment_name is not None:
        config.experiment.experiment_name = experiment_name
    
    if run_name is not None:
        config.experiment.run_name = run_name
    
    config.validate()
    return config

def load_config(path: Path) -> SyntheticConfig:
    """Load configuration from JSON file.
    
    Args:
        path: Path to JSON configuration file.
        
    Returns:
        SyntheticConfig instance.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return SyntheticConfig.from_dict(data)
