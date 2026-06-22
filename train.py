"""
train.py - Training Script for Building Footprint Segmentation

Supports all 7 model variants via CLI --model argument.
Reports Train Loss, Train IoU, Train Dice, Val Loss, Val IoU, Val Dice per epoch.

Usage:
    python train.py --model deeplabv3plus --train_dir data/train --val_dir data/val
    python train.py --model pspnet_scse --train_dir data/train --val_dir data/val --epochs 50
    python train.py --model unet_scse --resume checkpoints/unet_scse_resnet34_best.pth ...
"""

import os
import time
import argparse
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import get_model, save_checkpoint, load_checkpoint, SUPPORTED_MODELS


# =============================================================================
# Loss Function: Dice + BCE
# =============================================================================

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross-Entropy Loss.
    
    BCEWithLogitsLoss applies sigmoid internally, so we work with raw logits.
    Dice loss is computed on sigmoid-activated predictions.
    
    Args:
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss (works on raw logits, applies sigmoid internally)
        bce_loss = self.bce(preds, targets)
        
        # Dice loss (needs sigmoid)
        preds_sigmoid = torch.sigmoid(preds)
        
        # Flatten to [B*H*W]
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice coefficient
        intersection = (preds_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection) / (
            preds_flat.sum() + targets_flat.sum() + 1e-8
        )
        dice_loss = 1.0 - dice_coeff
        
        # Combined loss
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# =============================================================================
# Metrics
# =============================================================================

def iou_score(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate IoU (Jaccard Index) per sample, return mean over batch.
    
    Args:
        preds: Raw logits [B, 1, H, W]
        targets: Binary masks [B, 1, H, W] with values in {0, 1}
        threshold: Binarization threshold for predictions
    
    Returns:
        Scalar tensor with mean IoU over batch
    """
    # Apply sigmoid and threshold
    preds_sigmoid = torch.sigmoid(preds)
    preds_binary = (preds_sigmoid > threshold).float()
    
    # Flatten spatial dims: [B, H*W]
    preds_flat = preds_binary.view(preds_binary.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)
    
    # Per-sample IoU
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    
    iou_per_sample = (intersection + 1e-8) / (union + 1e-8)
    
    return iou_per_sample.mean()


def dice_score(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate Dice coefficient per sample, return mean over batch.
    
    Args:
        preds: Raw logits [B, 1, H, W]
        targets: Binary masks [B, 1, H, W] with values in {0, 1}
        threshold: Binarization threshold for predictions
    
    Returns:
        Scalar tensor with mean Dice over batch
    """
    # Apply sigmoid and threshold
    preds_sigmoid = torch.sigmoid(preds)
    preds_binary = (preds_sigmoid > threshold).float()
    
    # Flatten spatial dims: [B, H*W]
    preds_flat = preds_binary.view(preds_binary.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)
    
    # Per-sample Dice
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice_per_sample = (2.0 * intersection + 1e-8) / (
        preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-8
    )
    
    return dice_per_sample.mean()


# =============================================================================
# Dataset
# =============================================================================

class BuildingFootprintDataset(Dataset):
    """
    Dataset for building footprint segmentation.
    
    Expected directory structure:
        data_dir/
            images/
                img001.png
                img002.png
                ...
            masks/
                img001.png
                img002.png
                ...
    
    Images: RGB, any size (will be resized to image_size x image_size)
    Masks: Grayscale, will be binarized at threshold 127
    """
    
    VALID_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 512,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # Find all images
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
        
        # Verify at least some masks exist
        self._verify_masks()
    
    def _verify_masks(self):
        """Check that masks exist for images, warn if not."""
        missing = 0
        for img_file in self.image_files[:10]:  # Check first 10
            mask_path = self._get_mask_path(img_file)
            if mask_path is None:
                missing += 1
        if missing > 0:
            print(f"  Warning: {missing}/10 sample masks not found")
    
    def _get_mask_path(self, img_file: str) -> Optional[str]:
        """Find corresponding mask file for an image."""
        img_stem = os.path.splitext(img_file)[0]
        
        # Try same extension first
        for ext in self.VALID_IMAGE_EXTS:
            mask_path = os.path.join(self.mask_dir, img_stem + ext)
            if os.path.exists(mask_path):
                return mask_path
        
        # Try .png as default mask format
        mask_path = os.path.join(self.mask_dir, img_stem + '.png')
        if os.path.exists(mask_path):
            return mask_path
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        return image, mask
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load RGB image as [3, H, W] float32 tensor normalized to [0, 1]."""
        from PIL import Image
        import numpy as np
        
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # [3, H, W]
        return tensor
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """Load mask as [1, H, W] float32 tensor with values in {0, 1}."""
        from PIL import Image
        import numpy as np
        
        mask = Image.open(path).convert('L')
        arr = np.array(mask, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)  # Binarize
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        return tensor


def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    image_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Handles multiple directory structure conventions:
    1. train_dir/images/ and train_dir/masks/
    2. train_dir/ as image dir, train_dir/../masks/ as mask dir
    3. train_dir/ and val_dir/ as separate image/mask pairs
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        batch_size: Batch size
        image_size: Input image size (square)
        num_workers: DataLoader workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Detect directory structure
    train_img_dir, train_mask_dir = _detect_dirs(train_dir, 'train')
    val_img_dir, val_mask_dir = _detect_dirs(val_dir, 'val')
    
    print(f"  Train images: {train_img_dir}")
    print(f"  Train masks:  {train_mask_dir}")
    print(f"  Val images:   {val_img_dir}")
    print(f"  Val masks:    {val_mask_dir}")
    
    # Create datasets
    train_dataset = BuildingFootprintDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        image_size=image_size,
    )
    
    val_dataset = BuildingFootprintDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        image_size=image_size,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid partial batches affecting metrics
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


def _detect_dirs(base_dir: str, split_name: str) -> Tuple[str, str]:
    """
    Detect image and mask directories from base path.
    
    Convention 1: base_dir/images/ and base_dir/masks/
    Convention 2: base_dir is images, look for masks in sibling dir
    """
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
    
    # Fallback: assume base_dir contains both (will fail in dataset __init__)
    print(f"  Warning: Could not auto-detect mask directory for {base_dir}")
    return base_dir, base_dir


# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Returns:
        Dict with 'loss', 'iou', 'dice' averaged over batches
    """
    model.train()
    
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Compute metrics (no grad)
        with torch.no_grad():
            batch_iou = iou_score(outputs, masks)
            batch_dice = dice_score(outputs, masks)
        
        # Accumulate
        total_loss += loss.item()
        total_iou += batch_iou.item()
        total_dice += batch_dice.item()
        num_batches += 1
        
        # Progress logging (every 10 batches or last batch)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"    Batch [{batch_idx+1:4d}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"IoU: {batch_iou.item():.4f} "
                f"Dice: {batch_dice.item():.4f}"
            )
    
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
    """
    Validate model.
    
    Returns:
        Dict with 'loss', 'iou', 'dice' averaged over batches
    """
    model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        batch_iou = iou_score(outputs, masks)
        batch_dice = dice_score(outputs, masks)
        
        total_loss += loss.item()
        total_iou += batch_iou.item()
        total_dice += batch_dice.item()
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Building Footprint Segmentation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=SUPPORTED_MODELS,
        help='Model architecture to train'
    )
    model_group.add_argument(
        '--encoder', 
        type=str, 
        default='resnet34',
        help='Encoder backbone name'
    )
    model_group.add_argument(
        '--encoder_weights', 
        type=str, 
        default='imagenet',
        help='Pretrained encoder weights (use "None" for random init)'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--train_dir', 
        type=str, 
        required=True,
        help='Training data directory (containing images/ and masks/ subdirs)'
    )
    data_group.add_argument(
        '--val_dir', 
        type=str, 
        required=True,
        help='Validation data directory (containing images/ and masks/ subdirs)'
    )
    data_group.add_argument(
        '--image_size', 
        type=int, 
        default=512,
        help='Input image size (square)'
    )
    data_group.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='DataLoader worker processes'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Training batch size'
    )
    train_group.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--lr', 
        type=float, 
        default=1e-4,
        help='Initial learning rate'
    )
    train_group.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-4,
        help='Optimizer weight decay'
    )
    train_group.add_argument(
        '--scheduler_patience', 
        type=int, 
        default=5,
        help='LR scheduler patience (epochs without improvement)'
    )
    train_group.add_argument(
        '--scheduler_factor', 
        type=float, 
        default=0.5,
        help='LR scheduler reduction factor'
    )
    
    # Loss arguments
    loss_group = parser.add_argument_group('Loss')
    loss_group.add_argument(
        '--dice_weight', 
        type=float, 
        default=0.5,
        help='Weight for Dice loss component'
    )
    loss_group.add_argument(
        '--bce_weight', 
        type=float, 
        default=0.5,
        help='Weight for BCE loss component'
    )
    
    # Checkpoint arguments
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    ckpt_group.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint for resuming training'
    )
    ckpt_group.add_argument(
        '--save_every', 
        type=int, 
        default=10,
        help='Save periodic checkpoint every N epochs (0 to disable)'
    )
    
    # System arguments
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='Compute device'
    )
    sys_group.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Building Footprint Segmentation - Training")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Parse encoder weights (handle string "None")
    encoder_weights = args.encoder_weights
    if encoder_weights.lower() == 'none':
        encoder_weights = None
    
    # Create model
    print(f"\n--- Model ---")
    print(f"Architecture: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Encoder weights: {encoder_weights}")
    
    model = get_model(
        model_name=args.model,
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
    
    # Multi-GPU support
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} (trainable: {num_trainable:,})")
    
    # Loss function
    criterion = DiceBCELoss(
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
    )
    print(f"\n--- Loss ---")
    print(f"Dice+BCE (weights: {args.dice_weight}, {args.bce_weight})")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"\n--- Optimizer ---")
    print(f"Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize Dice
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=1e-7,
        verbose=True,
    )
    
    # Load data
    print(f"\n--- Data ---")
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val samples: {len(val_loader.dataset)} ({len(val_loader)} batches)")
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume is not None:
        print(f"\n--- Resuming from {args.resume} ---")
        checkpoint = load_checkpoint(model, args.resume, strict=True)
        
        # Verify model name matches
        ckpt_model_name = checkpoint.get('model_name', 'unknown')
        if ckpt_model_name != args.model:
            print(f"WARNING: Checkpoint model '{ckpt_model_name}' != requested '{args.model}'")
            print("This may cause errors if architectures differ.")
        
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best Dice so far: {best_dice:.4f}")
    
    # Checkpoint paths
    ckpt_prefix = f"{args.model}_{args.encoder}"
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{ckpt_prefix}_best.pth")
    last_ckpt_path = os.path.join(args.checkpoint_dir, f"{ckpt_prefix}_last.pth")
    
    # Training loop
    print("\n" + "=" * 80)
    print(f"Training: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] (lr: {current_lr:.2e})")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        
        # Validate
        print("  Validating...")
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step(val_metrics['dice'])
        
        # Print epoch summary
        print("-" * 40)
        print(
            f"  TRAIN - Loss: {train_metrics['loss']:.4f} "
            f"IoU: {train_metrics['iou']:.4f} "
            f"Dice: {train_metrics['dice']:.4f}"
        )
        print(
            f"  VAL   - Loss: {val_metrics['loss']:.4f} "
            f"IoU: {val_metrics['iou']:.4f} "
            f"Dice: {val_metrics['dice']:.4f}"
        )
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                model_name=args.model,
                encoder_name=args.encoder,
                image_size=args.image_size,
                filepath=best_ckpt_path,
            )
            print(f"  ★ Saved best checkpoint (Dice: {best_dice:.4f})")
        
        # Save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dice,
            model_name=args.model,
            encoder_name=args.encoder,
            image_size=args.image_size,
            filepath=last_ckpt_path,
        )
        
        # Save periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0 and not is_best:
            periodic_path = os.path.join(
                args.checkpoint_dir, 
                f"{ckpt_prefix}_epoch{epoch+1}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                model_name=args.model,
                encoder_name=args.encoder,
                image_size=args.image_size,
                filepath=periodic_path,
            )
            print(f"  Saved periodic checkpoint: epoch {epoch+1}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()