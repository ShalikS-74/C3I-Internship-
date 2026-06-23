"""
train.py - Training Script for Building Footprint Segmentation

Supports all 7 model variants via CLI --model argument.
Reports Train Loss, Train IoU, Train Dice, Val Loss, Val IoU, Val Dice per epoch.
Saves metrics to CSV via --log-path.

Usage:
    # Full training
    !python train.py \\
      --model deeplabv3plus \\
      --image-dir "/content/dataset/.../train/image" \\
      --mask-dir "/content/dataset/.../train/label" \\
      --val-image-dir "/content/dataset/.../val/image" \\
      --val-mask-dir "/content/dataset/.../val/label" \\
      --checkpoint-path checkpoints/deeplabv3plus/best_dice.pth \\
      --log-path checkpoints/deeplabv3plus/training_log.csv \\
      --image-size 512 --epochs 100 --batch-size 8

    # Smoke test (1 epoch, 100 samples)
    !python train.py \\
      --model pspnet_scse \\
      --image-dir "/content/dataset/.../train/image" \\
      --mask-dir "/content/dataset/.../train/label" \\
      --checkpoint-path checkpoints/pspnet_scse_smoke/best_dice.pth \\
      --log-path checkpoints/pspnet_scse_smoke/training_log.csv \\
      --image-size 512 --epochs 1 --batch-size 4 --max-samples 100
"""

import os
import csv
import time
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Single sources of truth - DO NOT REIMPLEMENT THESE
from dataset import BuildingFootprintDataset, get_training_transform, get_validation_transform
from metrics import iou_score, dice_score
from model import get_model, save_checkpoint, load_checkpoint, SUPPORTED_MODELS


# =============================================================================
# Loss Function: Dice + BCE
# =============================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice Loss and Binary Cross-Entropy Loss."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection) / (preds_flat.sum() + targets_flat.sum() + 1e-8)
        dice_loss = 1.0 - dice_coeff
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# =============================================================================
# CSV Logger
# =============================================================================

class MetricsLogger:
    """Writes epoch metrics to CSV."""
    
    FIELDNAMES = [
        'epoch', 'train_loss', 'train_iou', 'train_dice',
        'val_loss', 'val_iou', 'val_dice', 'lr', 'time_s'
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self._file = open(filepath, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict], lr: float, time_s: float):
        row = {
            'epoch': epoch,
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_iou': f"{train_metrics['iou']:.4f}",
            'train_dice': f"{train_metrics['dice']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}" if val_metrics else '',
            'val_iou': f"{val_metrics['iou']:.4f}" if val_metrics else '',
            'val_dice': f"{val_metrics['dice']:.4f}" if val_metrics else '',
            'lr': f"{lr:.2e}",
            'time_s': f"{time_s:.1f}",
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss, total_iou, total_dice, num_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        # dataset.py returns dicts: {"image": tensor, "mask": tensor, ...}
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Using metrics.py implementations (expects raw logits)
            total_iou += iou_score(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
        
        total_loss += loss.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'iou': total_iou / max(num_batches, 1),
        'dice': total_dice / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_iou, total_dice, num_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_iou += iou_score(outputs, masks).item()
        total_dice += dice_score(outputs, masks).item()
        total_loss += loss.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'iou': total_iou / max(num_batches, 1),
        'dice': total_dice / max(num_batches, 1),
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train Building Footprint Segmentation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument('--model', type=str, required=True, choices=SUPPORTED_MODELS,
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='Pretrained weights (use "None" for random init)')

    # Data
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Training image directory')
    parser.add_argument('--mask-dir', type=str, required=True,
                        help='Training mask directory')
    parser.add_argument('--val-image-dir', type=str, default=None,
                        help='Validation image directory (optional)')
    parser.add_argument('--val-mask-dir', type=str, default=None,
                        help='Validation mask directory (optional)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size (square)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training samples (for smoke testing)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='Limit validation samples')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help='LR scheduler patience')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='LR scheduler factor')

    # Outputs
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to save best model checkpoint')
    parser.add_argument('--log-path', type=str, default=None,
                        help='Path to save training metrics CSV')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Resolve encoder weights
    encoder_weights = args.encoder_weights
    if encoder_weights.lower() == 'none':
        encoder_weights = None

    # Resolve log path (default: same dir as checkpoint)
    if args.log_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        args.log_path = os.path.join(ckpt_dir, 'training_log.csv')

    print("=" * 70)
    print(f"Model: {args.model} | Encoder: {args.encoder}")
    print(f"Images: {args.image_dir}")
    print(f"Masks:  {args.mask_dir}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Log: {args.log_path}")
    print("=" * 70)

    # Create model
    model = get_model(
        model_name=args.model,
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # PyTorch 2.x safe (removed verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=1e-7
    )

    # Data - Using dataset.py single source of truth
    train_dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
        transform=get_training_transform()
    )
    
    # Fixed: drop_last=False to prevent silent data loss
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    val_loader = None
    if args.val_image_dir and args.val_mask_dir:
        val_dataset = BuildingFootprintDataset(
            image_dir=args.val_image_dir,
            mask_dir=args.val_mask_dir,
            image_size=args.image_size,
            max_samples=args.max_val_samples,
            transform=get_validation_transform()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_loader.dataset) if val_loader else 0} samples")

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(model, args.resume, strict=True)
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Logger
    logger = MetricsLogger(args.log_path)

    # Training Loop
    print("-" * 70)
    prev_lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device) if val_loader else None

        dt = time.time() - t0

        # Scheduler
        if val_m:
            scheduler.step(val_m['dice'])

        # Print
        val_str = ""
        if val_m:
            val_str = f" | Val Loss: {val_m['loss']:.4f} IoU: {val_m['iou']:.4f} Dice: {val_m['dice']:.4f}"
        print(f"Epoch [{epoch+1}/{args.epochs}] LR: {lr:.2e} | Train Loss: {train_m['loss']:.4f} IoU: {train_m['iou']:.4f} Dice: {train_m['dice']:.4f}{val_str} | {dt:.1f}s")

        # Log to CSV
        logger.log(epoch + 1, train_m, val_m, lr, dt)

        # Save best
        is_best = val_m and val_m['dice'] > best_dice
        if is_best:
            best_dice = val_m['dice']
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                args.model, args.encoder, args.image_size, args.checkpoint_path
            )
            print(f"  ★ Saved best (Dice: {best_dice:.4f})")
            
        # Manual LR notification (replaces verbose=True removed in PyTorch 2.x)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"  ⚡ LR reduced to {new_lr:.2e}")
            prev_lr = new_lr

    logger.close()
    print("=" * 70)
    print(f"Done. Best Dice: {best_dice:.4f}")
    print(f"Log: {args.log_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()