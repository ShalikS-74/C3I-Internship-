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
    # Using explicit dirs (matches train.py)
    python evaluate.py \
      --model deeplabv3plus \
      --checkpoint checkpoints/deeplabv3plus/best_dice.pth \
      --image-dir "/content/dataset/.../val/image" \
      --mask-dir "/content/dataset/.../val/label"

    # With custom output CSV
    python evaluate.py \
      --model pspnet_scse \
      --checkpoint checkpoints/pspnet_scse/best_dice.pth \
      --image-dir data/test/image \
      --mask-dir data/test/label \
      --output-csv results/pspnet_scse_metrics.csv
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Single sources of truth - DO NOT REIMPLEMENT THESE
from dataset import BuildingFootprintDataset, get_validation_transform
from metrics import (
    iou_score,
    dice_score,
    count_buildings,
    building_coverage_percentage,
    binary_mask_from_logits,
)
from model import get_model, load_checkpoint, SUPPORTED_MODELS


# =============================================================================
# Evaluation Logic
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
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
    
    iou_scores = []
    dice_scores = []
    count_errors = []
    coverage_errors = []
    per_sample_results = []
    
    for batch in dataloader:
        # dataset.py returns dicts
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]  # Keep on CPU for metrics.py compatibility
        filenames = batch["image_path"]
        
        # Forward pass
        outputs = model(images)
        
        # Compute metrics per sample
        batch_size = images.shape[0]
        for i in range(batch_size):
            sample_logits = outputs[i:i+1].cpu()
            sample_gt = masks[i:i+1]
            filename = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
            
            # 1. IoU and Dice using metrics.py (expects batched logits, so we pass [1,1,H,W])
            sample_iou = iou_score(sample_logits, sample_gt, threshold=threshold).item()
            sample_dice = dice_score(sample_logits, sample_gt, threshold=threshold).item()
            
            # 2. Count and Coverage using metrics.py
            # Convert logits to binary mask for counting
            pred_binary = binary_mask_from_logits(outputs[i].cpu(), threshold=threshold)
            gt_binary = masks[i]  # Already 0/1 from dataset.py
            
            pred_count = count_buildings(pred_binary)
            gt_count = count_buildings(gt_binary)
            
            # building_coverage_percentage returns 0-100, normalize to 0-1 for error calc
            pred_cov_ratio = building_coverage_percentage(pred_binary) / 100.0
            gt_cov_ratio = building_coverage_percentage(gt_binary) / 100.0
            
            # 3. Compute downstream errors
            sample_count_err = abs(pred_count - gt_count) / max(gt_count, 1)
            sample_coverage_err = abs(pred_cov_ratio - gt_cov_ratio) / max(gt_cov_ratio, 1e-6)
            
            # Accumulate
            iou_scores.append(sample_iou)
            dice_scores.append(sample_dice)
            count_errors.append(sample_count_err)
            coverage_errors.append(sample_coverage_err)
            
            # Store per-sample result
            per_sample_results.append({
                'filename': os.path.basename(filename),
                'iou': sample_iou,
                'dice': sample_dice,
                'count_error': sample_count_err,
                'coverage_error': sample_coverage_err,
                'pred_count': pred_count,
                'gt_count': gt_count,
                'pred_coverage': pred_cov_ratio,
                'gt_coverage': gt_cov_ratio,
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
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
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
    parser = argparse.ArgumentParser(
        description='Evaluate Building Footprint Segmentation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model', type=str, default=None, choices=SUPPORTED_MODELS,
        help='Model architecture (auto-detected from checkpoint if not specified)'
    )
    model_group.add_argument(
        '--encoder', type=str, default=None,
        help='Encoder name (auto-detected from checkpoint if not specified)'
    )
    model_group.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth file)'
    )
    
    # Data arguments (Matches train.py format)
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--image-dir', type=str, required=True,
        help='Image directory'
    )
    data_group.add_argument(
        '--mask-dir', type=str, required=True,
        help='Mask directory'
    )
    data_group.add_argument(
        '--image-size', type=int, default=None,
        help='Image size (auto-detected from checkpoint if not specified)'
    )
    data_group.add_argument(
        '--batch-size', type=int, default=8,
        help='Evaluation batch size'
    )
    data_group.add_argument(
        '--num-workers', type=int, default=4,
        help='DataLoader worker processes'
    )
    data_group.add_argument(
        '--threshold', type=float, default=0.5,
        help='Binarization threshold for predictions'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-csv', type=str, default=None,
        help='Path for per-sample metrics CSV (default: auto-generated in results/ dir)'
    )
    
    # System arguments
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument(
        '--device', type=str, default='cuda',
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
    
    checkpoint_raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
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
    
    # Resolve: CLI > checkpoint > default
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
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    load_checkpoint(model, args.checkpoint, strict=True)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ----------------------------------------------------------------------
    # Load data (Using dataset.py single source of truth)
    # ----------------------------------------------------------------------
    print(f"\n--- Data ---")
    print(f"  Images: {args.image_dir}")
    print(f"  Masks:  {args.mask_dir}")
    
    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=image_size,
        transform=get_validation_transform()
    )
    
    dataloader = DataLoader(
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
    
    return aggregate_metrics, per_sample_results


if __name__ == "__main__":
    main()