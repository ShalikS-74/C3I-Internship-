"""Evaluation script for building segmentation, count, and coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BuildingFootprintDataset, split_dataset
from metrics import (
    binary_mask_from_logits,
    building_coverage_percentage,
    count_buildings,
    dice_score,
    iou_score,
)
from model import get_unet_model
from visualization import show_predictions


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
    encoder_name: str = "resnet34",
) -> torch.nn.Module:
    """Load a saved U-Net checkpoint for inference."""

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        encoder_name = checkpoint.get("encoder_name", encoder_name)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = get_unet_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    threshold: float = 0.5,
    min_area: int = 20,
) -> dict[str, float]:
    """Evaluate validation loss, segmentation metrics, count, and coverage."""

    device = torch.device(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    totals = {
        "loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "building_count": 0.0,
        "building_coverage_percent": 0.0,
    }
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            batch_size = images.size(0)

            totals["loss"] += loss.item() * batch_size
            totals["iou"] += iou_score(logits, masks, threshold=threshold).item() * batch_size
            totals["dice"] += dice_score(logits, masks, threshold=threshold).item() * batch_size

            for item_index in range(batch_size):
                pred_mask = binary_mask_from_logits(logits[item_index], threshold=threshold)
                totals["building_count"] += count_buildings(pred_mask, min_area=min_area)
                totals["building_coverage_percent"] += building_coverage_percentage(pred_mask)

            sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained building segmentation model.")
    parser.add_argument("--image-dir", required=True, help="Folder containing satellite images.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing binary building masks.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument("--num-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    _, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        encoder_name=args.encoder_name,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        threshold=args.threshold,
        min_area=args.min_area,
    )

    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Validation IoU: {metrics['iou']:.4f}")
    print(f"Validation Dice: {metrics['dice']:.4f}")
    print(f"Mean Building Count = {metrics['building_count']:.2f}")
    print(f"Mean Building Coverage = {metrics['building_coverage_percent']:.2f}%")

    if args.show_examples:
        show_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            num_samples=args.num_examples,
            threshold=args.threshold,
            min_area=args.min_area,
        )


if __name__ == "__main__":
    main()
