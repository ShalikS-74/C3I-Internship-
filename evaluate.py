"""
evaluate.py
Evaluation script for building footprint segmentation.

Usage:
    python evaluate.py --model pspnet      --checkpoint checkpoints/pspnet_best.pth      --image_dir data/images --mask_dir data/masks
    python evaluate.py --model pspnet_scse --checkpoint checkpoints/pspnet_scse_best.pth --image_dir data/images --mask_dir data/masks
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import get_model, load_checkpoint, SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Dataset — must match train.py exactly
# ---------------------------------------------------------------------------
class BuildingDataset(torch.utils.data.Dataset):
    """
    Expects:
        image_dir/  -> *.png / *.jpg / *.tif
        mask_dir/   -> same filenames as images
    Returns:
        image: [3, 512, 512] float, ImageNet-normalized
        mask:  [1, 512, 512] float, binary {0.0, 1.0}
    """

    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 512):
        import numpy as np
        self.np = np
        exts = ["*.png", "*.jpg", "*.tif", "*.tiff"]
        self.image_paths = []
        for ext in exts:
            self.image_paths += sorted(Path(image_dir).glob(ext))
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        image = torch.from_numpy(self.np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(self.np.array(mask)).unsqueeze(0).float()
        mask = (mask > 127).float()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image, mask, str(img_path.name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Computes per-batch Dice, IoU, Precision, Recall, F1.

    Args:
        logits:    [B, 1, H, W] raw logits
        targets:   [B, 1, H, W] float binary masks
        threshold: binarization threshold

    Returns:
        dict of metric name -> float (batch mean)
    """
    probs = torch.sigmoid(logits).detach()
    preds = (probs > threshold).float()

    p = preds.view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    tp = (p * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)

    smooth = 1.0
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = (2 * precision * recall) / (precision + recall + 1e-8)

    return {
        "dice":      dice.mean().item(),
        "iou":       iou.mean().item(),
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "f1":        f1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Run full evaluation over loader.

    Returns:
        aggregated metrics dict, list of per-image result dicts
    """
    model.eval()

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    per_image_results = []
    n_batches = 0

    for images, masks, filenames in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)

        # Guard against tuple output (aux_params safety)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # Verify output shape
        assert logits.shape == masks.shape, (
            f"Shape mismatch: logits={logits.shape}, masks={masks.shape}"
        )

        batch_metrics = compute_metrics(logits, masks, threshold=threshold)

        for k, v in batch_metrics.items():
            totals[k] += v
        n_batches += 1

        # Per-image metrics
        probs = torch.sigmoid(logits).detach()
        preds = (probs > threshold).float()

        for i, fname in enumerate(filenames):
            p = preds[i].view(1, -1)
            t = masks[i].view(1, -1)
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            smooth = 1.0
            img_dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            img_iou  = (tp + smooth) / (tp + fp + fn + smooth)
            per_image_results.append({
                "filename": fname,
                "dice": round(img_dice, 4),
                "iou":  round(img_iou, 4),
            })

    aggregated = {k: v / max(1, n_batches) for k, v in totals.items()}
    return aggregated, per_image_results


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate building segmentation model")
    parser.add_argument("--model",       type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--checkpoint",  type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image_dir",   type=str, default="data/images")
    parser.add_argument("--mask_dir",    type=str, default="data/masks")
    parser.add_argument("--output_dir",  type=str, default="results")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--image_size",  type=int, default=512)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder",     type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--save_per_image_csv", action="store_true",
                        help="Save per-image metrics to CSV")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"[Evaluate] Model:      {args.model}")
    print(f"[Evaluate] Checkpoint: {args.checkpoint}")
    print(f"[Evaluate] Device:     {device}")
    print(f"[Evaluate] Threshold:  {args.threshold}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    dataset = BuildingDataset(
        args.image_dir,
        args.mask_dir,
        image_size=args.image_size,
    )
    print(f"[Evaluate] Images found: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = get_model(
        args.model,
        encoder_name=args.encoder,
        encoder_weights=None,   # weights loaded from checkpoint
    ).to(device)

    load_checkpoint(model, args.checkpoint, strict=True)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Evaluate] Parameters: {total_params:.1f}M")

    # Run evaluation
    t0 = time.time()
    aggregated, per_image_results = evaluate(
        model, loader, device, threshold=args.threshold
    )
    elapsed = time.time() - t0

    # Print results
    print()
    print("=" * 60)
    print(f"[Evaluate] Results over {len(dataset)} images ({elapsed:.1f}s)")
    print("=" * 60)
    print(f"  Dice:      {aggregated['dice']:.4f}")
    print(f"  IoU:       {aggregated['iou']:.4f}")
    print(f"  Precision: {aggregated['precision']:.4f}")
    print(f"  Recall:    {aggregated['recall']:.4f}")
    print(f"  F1:        {aggregated['f1']:.4f}")
    print("=" * 60)

    # Save aggregated results
    import json
    summary = {
        "model":      args.model,
        "checkpoint": args.checkpoint,
        "n_images":   len(dataset),
        "threshold":  args.threshold,
        "metrics":    aggregated,
    }
    summary_path = os.path.join(args.output_dir, f"{args.model}_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Evaluate] Summary saved -> {summary_path}")

    # Save per-image CSV
    if args.save_per_image_csv:
        import csv
        csv_path = os.path.join(args.output_dir, f"{args.model}_per_image.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "dice", "iou"])
            writer.writeheader()
            writer.writerows(per_image_results)
        print(f"[Evaluate] Per-image CSV saved -> {csv_path}")

        # Print worst 10
        sorted_results = sorted(per_image_results, key=lambda x: x["dice"])
        print("\n[Evaluate] 10 worst predictions by Dice:")
        for r in sorted_results[:10]:
            print(f"  {r['filename']:<40} dice={r['dice']:.4f}  iou={r['iou']:.4f}")


if __name__ == "__main__":
    main()
