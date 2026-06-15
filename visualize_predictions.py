"""Visualize model predictions alongside ground-truth masks.

Loads a saved checkpoint, runs inference on a small set of dataset samples,
and saves a 3-column figure (RGB | Ground truth | Predicted mask) per sample
into the outputs/predictions/ directory.

Usage example:
python visualize_predictions.py \
  --image-dir "dataset/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image" \
  --mask-dir "dataset/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/label" \
  --checkpoint-path checkpoints/benchmark_500/best_dice.pth \
  --image-size 512 \
  --num-samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BuildingFootprintDataset
from model import get_unet_model
from metrics import binary_mask_from_logits, binary_mask_to_uint8


def tensor_to_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a CxHxW or HxWxC image tensor/array to HxWx3 float image in [0,1]."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    # CxHxW -> HxWxC
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    # single-channel -> replicate
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def clean_state_dict(state_dict: dict) -> dict:
    """Remove common DataParallel prefixes from a state dict if present."""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path: str | Path, device: torch.device, encoder_name: str = "resnet34") -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        encoder_name = checkpoint.get("encoder_name", encoder_name)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    model = get_unet_model(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize model predictions.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint-path", default="checkpoints/benchmark_500/best_dice.pth")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-dir", default="outputs/predictions/")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint_path} on {device}")
    model = load_model(args.checkpoint_path, device=device, encoder_name=args.encoder_name)

    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        max_samples=args.num_samples,
    )

    sample_count = min(args.num_samples, len(dataset))
    print(f"Visualizing {sample_count} samples from dataset (found {len(dataset)}).")

    with torch.no_grad():
        for idx in range(sample_count):
            sample = dataset[idx]
            image_tensor = sample["image"]
            mask_tensor = sample["mask"]

            image_batch = image_tensor.unsqueeze(0).to(device)
            logits = model(image_batch)

            pred_mask = binary_mask_from_logits(logits[0], threshold=args.threshold)
            gt_mask = binary_mask_to_uint8(mask_tensor)

            img = tensor_to_numpy_image(image_tensor)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title("Original image")
            axes[0].axis("off")

            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground truth mask")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Predicted mask")
            axes[2].axis("off")

            stem = Path(sample["image_path"]).stem
            out_path = save_dir / f"{stem}.png"
            plt.tight_layout()
            fig.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
            if args.show:
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
