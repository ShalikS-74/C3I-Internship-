"""
evaluate.py - Evaluation Script for Building Footprint Segmentation

Evaluates a trained model and exports per-sample metrics CSV for reliability analysis.

Reports aggregate metrics:
- Mean IoU
- Mean Dice
- Mean Count Error
- Mean Coverage Error

Exports CSV with per-sample metrics for correlation analysis in analyze_predictions.py.

Usage:
    python evaluate.py --model deeplabv3plus --checkpoint checkpoints/deeplabv3plus_resnet34_best.pth --data_dir data/val
    python evaluate.py --model pspnet_scse --checkpoint checkpoints/pspnet_scse_resnet34_best.pth --data_dir data/test --output_csv results/pspnet_scse_metrics.csv
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from model import get_model, load_checkpoint, SUPPORTED_MODELS


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute IoU (Jaccard Index) for a single sample.
    
    Args:
        pred: Binary prediction mask [H, W] with values in {0, 1}
        gt: Binary ground truth mask [H, W] with values in {0, 1}
    
    Returns:
        IoU score as float in [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        # Both empty - perfect match
        return 1.0
    
    return float(intersection) / float(union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice coefficient for a single sample.
    
    Args:
        pred: Binary prediction mask [H, W] with values in {0, 1}
        gt: Binary ground truth mask [H, W] with values in {0, 1}
    
    Returns:
        Dice score as float in [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    
    if total == 0:
        # Both empty - perfect match
        return 1.0
    
    return 2.0 * float(intersection) / float(total)


def count_buildings(mask: np.ndarray) -> int:
    """
    Count number of distinct building instances using connected components.
    
    Args:
        mask: Binary mask [H, W] with values in {0, 1}
    
    Returns:
        Number of connected components (buildings)
    """
    if mask.sum() == 0:
        return 0
    
    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        return num_features
    except ImportError:
        # Fallback: estimate count by assuming each 16x16 block is one building
        # This is a rough approximation if scipy is not available
        block_size = 16
        h, w = mask.shape
        count = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = mask[i:i+block_size, j:j+block_size]
                if block.sum() > block_size * block_size * 0.1:  # >10% filled
                    count += 1
        return count


def compute_count_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute relative building count error.
    
    Count Error = |pred_count - gt_count| / max(gt_count, 1)
    
    Using max(gt_count, 1) avoids division by zero when there are no buildings.
    
    Args:
        pred: Binary prediction mask [H, W]
        gt: Binary ground truth mask [H, W]
    
    Returns:
        Relative count error as float (lower is better, 0 = perfect)
    """
    pred_count = count_buildings(pred)
    gt_count = count_buildings(gt)
    
    return abs(pred_count - gt_count) / max(gt_count, 1)


def compute_coverage_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute relative building coverage error.
    
    Coverage = building_pixels / total_pixels
    Coverage Error = |pred_coverage - gt_coverage| / max(gt_coverage, 1e-6)
    
    Using max(gt_coverage, 1e-6) avoids division by zero.
    
    Args:
        pred: Binary prediction mask [H, W]
        gt: Binary ground truth mask [H, W]
    
    Returns:
        Relative coverage error as float (lower is better, 0 = perfect)
    """
    pred_coverage = pred.sum() / pred.size
    gt_coverage = gt.sum() / gt.size
    
    return abs(pred_coverage - gt_coverage) / max(gt_coverage, 1e-6)


# =============================================================================
# Dataset (same as train.py for consistency)
# =============================================================================

class BuildingFootprintDataset(torch.utils.data.Dataset):
    """Dataset for evaluation (same as training, no augmentation)."""
    
    VALID_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in self.VALID_IMAGE_EXTS
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def _get_mask_path(self, img_file: str) -> Optional[str]:
        """Find corresponding mask file for an image."""
        img_stem = os.path.splitext(img_file)[0]
        
        for ext in self.VALID_IMAGE_EXTS:
            mask_path = os.path.join(self.mask_dir, img_stem + ext)
            if os.path.exists(mask_path):
                return mask_path
        
        mask_path = os.path.join(self.mask_dir, img_stem + '.png')
        if os.path.exists(mask_path):
            return mask_path
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_file = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        image = self._load_image(img_path)
        
        # Load mask
        mask_path = self._get_mask_path(img_file)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_file}")
        mask = self._load_mask(mask_path)
        
        # Resize if needed
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(0)
        
        return image, mask, img_file
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load RGB image as [3, H, W] float32 tensor normalized to [0, 1]."""
        from PIL import Image
        import numpy as np
        
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """Load mask as [1, H, W] float32 tensor with values in {0, 1}."""
        from PIL import Image
        import numpy as np
        
        mask = Image.open(path).convert('L')
        arr = np.array(mask, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


def detect_dirs(base_dir: str) -> Tuple[str, str]:
    """Detect image and mask directories from base path."""
    # Convention 1: subdirectories
    if os.path.isdir(os.path.join(base_dir, 'images')):
        img_dir = os.path.join(base_dir, 'images')
        mask_dir = os.path.join(base_dir, 'masks')
        if os.path.isdir(mask_dir):
            return img_dir, mask_dir
    
    # Convention 2: base_dir is images, masks in sibling
    parent = os.path.dirname(base_dir)
    mask_dir = os.path.join(parent, 'masks')
    if os.path.isdir(mask_dir):
        return base_dir, mask_dir
    
    # Fallback
    return base_dir, base_dir


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained segmentation model
        dataloader: Evaluation dataloader
        device: Compute device
        threshold: Binarization threshold for predictions
    
    Returns:
        Tuple of:
        - aggregate_metrics: Dict with 'iou', 'dice', 'count_error', 'coverage_error'
        - per_sample_results: List of dicts with per-sample metrics and metadata
    """
    model.eval()
    
    # Accumulators
    iou_scores = []
    dice_scores = []
    count_errors = []
    coverage_errors = []
    per_sample_results = []
    
    for images, masks, filenames in dataloader:
        images = images.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        
        # Apply sigmoid and threshold to get binary predictions
        preds_sigmoid = torch.sigmoid(outputs)
        preds_binary = (preds_sigmoid > threshold).float()
        
        # Move to CPU for numpy conversion
        preds_np = preds_binary.cpu().numpy()
        masks_np = masks.numpy()
        
        # Compute metrics per sample
        batch_size = preds_np.shape[0]
        for i in range(batch_size):
            pred_mask = preds_np[i, 0]  # [H, W]
            gt_mask = masks_np[i, 0]    # [H, W]
            filename = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
            
            # Compute all 4 metrics
            sample_iou = compute_iou(pred_mask, gt_mask)
            sample_dice = compute_dice(pred_mask, gt_mask)
            sample_count_err = compute_count_error(pred_mask, gt_mask)
            sample_coverage_err = compute_coverage_error(pred_mask, gt_mask)
            
            # Additional info for CSV
            pred_count = count_buildings(pred_mask)
            gt_count = count_buildings(gt_mask)
            pred_coverage = pred_mask.sum() / pred_mask.size
            gt_coverage = gt_mask.sum() / gt_mask.size
            
            # Accumulate
            iou_scores.append(sample_iou)
            dice_scores.append(sample_dice)
            count_errors.append(sample_count_err)
            coverage_errors.append(sample_coverage_err)
            
            # Store per-sample result
            per_sample_results.append({
                'filename': filename,
                'iou': sample_iou,
                'dice': sample_dice,
                'count_error': sample_count_err,
                'coverage_error': sample_coverage_err,
                'pred_count': pred_count,
                'gt_count': gt_count,
                'pred_coverage': pred_coverage,
                'gt_coverage': gt_coverage,
            })
    
    # Compute aggregate metrics
    aggregate_metrics = {
        'iou': np.mean(iou_scores) if iou_scores else 0.0,
        'dice': np.mean(dice_scores) if dice_scores else 0.0,
        'count_error': np.mean(count_errors) if count_errors else 0.0,
        'coverage_error': np.mean(coverage_errors) if coverage_errors else 0.0,
    }
    
    return aggregate_metrics, per_sample_results


def save_metrics_csv(
    per_sample_results: List[Dict],
    filepath: str,
    model_name: str,
    encoder_name: str,
    checkpoint_path: str,
) -> None:
    """
    Save per-sample metrics to CSV for reliability analysis.
    
    CSV columns:
    - filename: Image filename
    - iou: IoU score
    - dice: Dice score
    - count_error: Relative building count error
    - coverage_error: Relative building coverage error
    - pred_count: Predicted number of buildings
    - gt_count: Ground truth number of buildings
    - pred_coverage: Predicted building coverage ratio
    - gt_coverage: Ground truth building coverage ratio
    
    Args:
        per_sample_results: List of per-sample metric dicts
        filepath: Output CSV path
        model_name: Model architecture name
        encoder_name: Encoder backbone name
        checkpoint_path: Path to evaluated checkpoint
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Define columns (order matters for readability)
    fieldnames = [
        'filename',
        'iou',
        'dice',
        'count_error',
        'coverage_error',
        'pred_count',
        'gt_count',
        'pred_coverage',
        'gt_coverage',
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_results)
    
    print(f"  Per-sample metrics saved to: {filepath}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Building Footprint Segmentation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model',
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help='Model architecture (auto-detected from checkpoint if not specified)'
    )
    model_group.add_argument(
        '--encoder',
        type=str,
        default=None,
        help='Encoder name (auto-detected from checkpoint if not specified)'
    )
    model_group.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Data directory (containing images/ and masks/ subdirs)'
    )
    data_group.add_argument(
        '--image_size',
        type=int,
        default=None,
        help='Image size (auto-detected from checkpoint if not specified)'
    )
    data_group.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Evaluation batch size'
    )
    data_group.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader worker processes'
    )
    data_group.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Binarization threshold for predictions'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path for per-sample metrics CSV (default: auto-generated in results/ dir)'
    )
    
    # System arguments
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute device'
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Building Footprint Segmentation - Evaluation")
    print("=" * 80)
    print(f"\nDevice: {device}")
    
    # ----------------------------------------------------------------------
    # Load checkpoint first to get metadata
    # ----------------------------------------------------------------------
    print(f"\n--- Loading Checkpoint ---")
    print(f"Path: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Create a temporary model to load checkpoint (we'll recreate if needed)
    # First, peek at checkpoint metadata
    checkpoint_raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extract metadata
    ckpt_model_name = checkpoint_raw.get('model_name', None)
    ckpt_encoder_name = checkpoint_raw.get('encoder_name', 'resnet34')
    ckpt_image_size = checkpoint_raw.get('image_size', 512)
    ckpt_epoch = checkpoint_raw.get('epoch', '?')
    ckpt_best_dice = checkpoint_raw.get('best_dice', '?')
    
    print(f"  Checkpoint model: {ckpt_model_name}")
    print(f"  Checkpoint encoder: {ckpt_encoder_name}")
    print(f"  Checkpoint image size: {ckpt_image_size}")
    print(f"  Checkpoint epoch: {ckpt_epoch}")
    print(f"  Checkpoint best Dice: {ckpt_best_dice}")
    
    # Resolve model/encoder/image_size: CLI > checkpoint > default
    model_name = args.model if args.model is not None else ckpt_model_name
    encoder_name = args.encoder if args.encoder is not None else ckpt_encoder_name
    image_size = args.image_size if args.image_size is not None else ckpt_image_size
    
    if model_name is None:
        raise ValueError(
            "Model name not specified and not found in checkpoint. "
            "Use --model to specify explicitly."
        )
    
    print(f"\n--- Model ---")
    print(f"Architecture: {model_name}")
    print(f"Encoder: {encoder_name}")
    print(f"Image size: {image_size}")
    
    # Warn if CLI args differ from checkpoint
    if args.model is not None and args.model != ckpt_model_name:
        print(f"  WARNING: CLI --model '{args.model}' != checkpoint '{ckpt_model_name}'")
    if args.encoder is not None and args.encoder != ckpt_encoder_name:
        print(f"  WARNING: CLI --encoder '{args.encoder}' != checkpoint '{ckpt_encoder_name}'")
    
    # ----------------------------------------------------------------------
    # Create model and load weights
    # ----------------------------------------------------------------------
    model = get_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=None,  # Don't load pretrained, we're loading checkpoint
        in_channels=3,
        classes=1,
    )
    
    # Load checkpoint weights
    load_checkpoint(model, args.checkpoint, strict=True)
    
    # Multi-GPU support
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------
    print(f"\n--- Data ---")
    img_dir, mask_dir = detect_dirs(args.data_dir)
    print(f"  Images: {img_dir}")
    print(f"  Masks: {mask_dir}")
    
    dataset = BuildingFootprintDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        image_size=image_size,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"  Samples: {len(dataset)} ({len(dataloader)} batches)")
    
    # ----------------------------------------------------------------------
    # Run evaluation
    # ----------------------------------------------------------------------
    print(f"\n--- Evaluation (threshold={args.threshold}) ---")
    
    aggregate_metrics, per_sample_results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
    )
    
    # ----------------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  Model:              {model_name}")
    print(f"  Encoder:            {encoder_name}")
    print(f"  Checkpoint:         {args.checkpoint}")
    print(f"  Samples evaluated:  {len(per_sample_results)}")
    print(f"  Threshold:          {args.threshold}")
    print()
    print(f"  Mean IoU:           {aggregate_metrics['iou']:.4f}")
    print(f"  Mean Dice:          {aggregate_metrics['dice']:.4f}")
    print(f"  Mean Count Error:   {aggregate_metrics['count_error']:.4f}")
    print(f"  Mean Coverage Error:{aggregate_metrics['coverage_error']:.4f}")
    print()
    
    # Additional statistics
    if len(per_sample_results) > 1:
        ious = [r['iou'] for r in per_sample_results]
        dices = [r['dice'] for r in per_sample_results]
        count_errs = [r['count_error'] for r in per_sample_results]
        cov_errs = [r['coverage_error'] for r in per_sample_results]
        
        print("  --- Distribution ---")
        print(f"  IoU:    min={min(ious):.4f}, max={max(ious):.4f}, std={np.std(ious):.4f}")
        print(f"  Dice:   min={min(dices):.4f}, max={max(dices):.4f}, std={np.std(dices):.4f}")
        print(f"  CntErr: min={min(count_errs):.4f}, max={max(count_errs):.4f}, std={np.std(count_errs):.4f}")
        print(f"  CovErr: min={min(cov_errs):.4f}, max={max(cov_errs):.4f}, std={np.std(cov_errs):.4f}")
    
    print("=" * 80)
    
    # ----------------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------------
    # Auto-generate CSV path if not specified
    if args.output_csv is None:
        csv_dir = 'results'
        csv_filename = f"{model_name}_{encoder_name}_metrics.csv"
        output_csv = os.path.join(csv_dir, csv_filename)
    else:
        output_csv = args.output_csv
    
    save_metrics_csv(
        per_sample_results=per_sample_results,
        filepath=output_csv,
        model_name=model_name,
        encoder_name=encoder_name,
        checkpoint_path=args.checkpoint,
    )
    
    # ----------------------------------------------------------------------
    # Return metrics for programmatic use
    # ----------------------------------------------------------------------
    return aggregate_metrics, per_sample_results


if __name__ == "__main__":
    main()