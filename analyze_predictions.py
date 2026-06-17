"""Analyze urban metrics from model predictions on a small sample set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dataset import BuildingFootprintDataset
from metrics import binary_mask_from_logits, dice_score, iou_score
from model import get_deeplabv3plus_model
from urban_metrics import compare_ground_truth_prediction, to_binary_mask


DEFAULT_DATA_ROOT = Path("dataset/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels")


def clean_state_dict(state_dict: dict) -> dict:
    """Remove a DataParallel prefix if the checkpoint was saved that way."""

    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state_dict.items()}

    return state_dict


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    encoder_name: str = "resnet34",
) -> torch.nn.Module:
    """Load the trained segmentation model for analysis."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        encoder_name = checkpoint.get("encoder_name", encoder_name)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = get_deeplabv3plus_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(clean_state_dict(state_dict))
    model.to(device)
    model.eval()

    return model


def analyze_predictions(
    model: torch.nn.Module,
    dataset: BuildingFootprintDataset,
    device: torch.device,
    num_samples: int = 10,
    threshold: float = 0.5,
    min_area: int = 1,
) -> list[dict[str, float | int | str]]:
    """Run predictions and compute segmentation plus urban metrics."""

    rows: list[dict[str, float | int | str]] = []
    sample_count = min(num_samples, len(dataset))

    with torch.no_grad():
        for index in range(sample_count):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            ground_truth = sample["mask"].unsqueeze(0).to(device)
            logits = model(image)

            predicted_mask = binary_mask_from_logits(logits[0], threshold=threshold)
            ground_truth_mask = to_binary_mask(sample["mask"])
            urban_metrics = compare_ground_truth_prediction(
                ground_truth_mask=ground_truth_mask,
                predicted_mask=predicted_mask,
                min_area=min_area,
            )

            row = {
                "image_id": Path(str(sample["image_path"])).stem,
                "iou": float(iou_score(logits, ground_truth, threshold=threshold).item()),
                "dice": float(dice_score(logits, ground_truth, threshold=threshold).item()),
                **urban_metrics,
            }
            rows.append(row)

    return rows


def print_analysis(rows: list[dict[str, float | int | str]]) -> None:
    """Print per-image analysis in a readable format."""

    for row in rows:
        print(f"Image ID: {row['image_id']}")
        print(f"IoU: {row['iou']:.4f}")
        print(f"Dice: {row['dice']:.4f}")
        print(f"Ground Truth Count: {row['ground_truth_count']}")
        print(f"Predicted Count: {row['predicted_count']}")
        print(f"Count Error: {row['count_error']}")
        print(f"Ground Truth Coverage: {row['ground_truth_coverage']:.2f}%")
        print(f"Predicted Coverage: {row['predicted_coverage']:.2f}%")
        print(f"Coverage Error: {row['coverage_error']:.2f}%")
        print()


def save_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    """Save analysis rows to CSV."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "iou",
        "dice",
        "ground_truth_count",
        "predicted_count",
        "count_error",
        "ground_truth_coverage",
        "predicted_coverage",
        "coverage_error",
    ]
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze urban metrics from trained model predictions.")
    parser.add_argument("--image-dir", default=str(DEFAULT_DATA_ROOT / "test/image"))
    parser.add_argument("--mask-dir", default=str(DEFAULT_DATA_ROOT / "test/label"))
    parser.add_argument("--checkpoint-path", default="checkpoints/benchmark_500/best_dice.pth")
    parser.add_argument("--output-csv", default="outputs/urban_metrics_analysis.csv")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--encoder-name", default="resnet34")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        max_samples=args.num_samples,
    )
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        encoder_name=args.encoder_name,
    )

    rows = analyze_predictions(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    print_analysis(rows)
    save_csv(rows, args.output_csv)
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
