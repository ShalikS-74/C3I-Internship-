"""Visualize anomaly cases where IoU and count reliability disagree."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BuildingFootprintDataset
from metrics import binary_mask_from_logits
from model import get_unet_model
from urban_metrics import to_binary_mask


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
    """Load the trained segmentation model."""

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
    model.load_state_dict(clean_state_dict(state_dict))
    model.to(device)
    model.eval()

    return model


def load_analysis_rows(csv_path: str | Path) -> list[dict[str, float | int | str]]:
    """Load urban metric analysis rows from CSV."""

    rows: list[dict[str, float | int | str]] = []
    with Path(csv_path).open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "image_id": row["image_id"],
                    "iou": float(row["iou"]),
                    "dice": float(row["dice"]),
                    "ground_truth_count": int(float(row["ground_truth_count"])),
                    "predicted_count": int(float(row["predicted_count"])),
                    "count_error": int(float(row["count_error"])),
                    "ground_truth_coverage": float(row["ground_truth_coverage"]),
                    "predicted_coverage": float(row["predicted_coverage"]),
                    "coverage_error": float(row["coverage_error"]),
                }
            )

    return rows


def select_group(
    rows: list[dict[str, float | int | str]],
    group_name: str,
    threshold_pairs: list[tuple[float, int]],
    min_cases: int,
    max_cases: int,
) -> tuple[list[dict[str, float | int | str]], tuple[float, int], bool]:
    """Select anomaly rows, relaxing thresholds until enough cases are found."""

    selected: list[dict[str, float | int | str]] = []
    used_threshold = threshold_pairs[-1]
    relaxed = False

    for index, (iou_threshold, count_threshold) in enumerate(threshold_pairs):
        if group_name == "high_iou_high_count_error":
            candidates = [
                row
                for row in rows
                if float(row["iou"]) >= iou_threshold and int(row["count_error"]) >= count_threshold
            ]
            candidates = sorted(candidates, key=lambda row: (int(row["count_error"]), float(row["iou"])), reverse=True)
        elif group_name == "low_iou_low_count_error":
            candidates = [
                row
                for row in rows
                if float(row["iou"]) <= iou_threshold and int(row["count_error"]) <= count_threshold
            ]
            candidates = sorted(candidates, key=lambda row: (int(row["count_error"]), -float(row["iou"])))
        else:
            raise ValueError(f"Unknown group: {group_name}")

        selected = candidates[:max_cases]
        used_threshold = (iou_threshold, count_threshold)
        relaxed = index > 0

        if len(selected) >= min_cases:
            break

    return selected, used_threshold, relaxed


def tensor_to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert CxHxW tensor image to HxWx3 image in [0, 1]."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def build_sample_index(dataset: BuildingFootprintDataset) -> dict[str, int]:
    """Map image filename stems to dataset indices."""

    return {image_path.stem: index for index, (image_path, _) in enumerate(dataset.pairs)}


def save_case_figure(
    row: dict[str, float | int | str],
    dataset: BuildingFootprintDataset,
    sample_index: dict[str, int],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    threshold: float,
) -> Path | None:
    """Save a visual comparison figure for one anomaly case."""

    image_id = str(row["image_id"])
    if image_id not in sample_index:
        print(f"Skipping {image_id}: not found in dataset image directory.")
        return None

    sample = dataset[sample_index[image_id]]
    image_tensor = sample["image"]
    gt_mask = to_binary_mask(sample["mask"])

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred_mask = binary_mask_from_logits(logits[0], threshold=threshold)

    image = tensor_to_numpy_image(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")

    fig.suptitle(
        f"Image ID: {image_id} | IoU: {float(row['iou']):.3f} | Dice: {float(row['dice']):.3f}\n"
        f"GT Count: {int(row['ground_truth_count'])} | Pred Count: {int(row['predicted_count'])} | "
        f"Count Error: {int(row['count_error'])} | Coverage Error: {float(row['coverage_error']):.2f}%",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_id}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def summarize_group(
    group_title: str,
    rows: list[dict[str, float | int | str]],
    used_threshold: tuple[float, int],
    relaxed: bool,
    threshold_text: str,
) -> list[str]:
    """Return report lines for one anomaly group."""

    lines = [
        f"{group_title}",
        f"Threshold used: {threshold_text.format(iou=used_threshold[0], count=used_threshold[1])}",
        f"Threshold relaxed: {'yes' if relaxed else 'no'}",
        f"Number of anomalies found: {len(rows)}",
    ]

    if rows:
        image_ids = [str(row["image_id"]) for row in rows]
        avg_iou = sum(float(row["iou"]) for row in rows) / len(rows)
        avg_count_error = sum(float(row["count_error"]) for row in rows) / len(rows)
        lines.extend(
            [
                f"Image IDs: {', '.join(image_ids)}",
                f"Average IoU: {avg_iou:.4f}",
                f"Average Count Error: {avg_count_error:.4f}",
            ]
        )
    else:
        lines.append("Image IDs: none")

    return lines


def write_report(
    output_dir: Path,
    high_rows: list[dict[str, float | int | str]],
    high_threshold: tuple[float, int],
    high_relaxed: bool,
    low_rows: list[dict[str, float | int | str]],
    low_threshold: tuple[float, int],
    low_relaxed: bool,
) -> Path:
    """Write a text report summarizing anomaly selections."""

    report_path = output_dir / "report.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "=== Anomaly Analysis Report ===",
        "",
        "Purpose:",
        "Visual inspection of cases where IoU and building count reliability disagree.",
        "",
    ]
    lines.extend(
        summarize_group(
            group_title="Group A: High IoU but High Count Error",
            rows=high_rows,
            used_threshold=high_threshold,
            relaxed=high_relaxed,
            threshold_text="IoU >= {iou:.2f}, Count Error >= {count}",
        )
    )
    lines.extend(["", "Observations:"])
    if high_rows:
        lines.append("These cases suggest that apparently acceptable overlap can still merge, split, or miss buildings in ways that affect counts.")
    else:
        lines.append("No clear high-IoU/high-count-error examples were found under the relaxed thresholds.")

    lines.extend(["", ""])
    lines.extend(
        summarize_group(
            group_title="Group B: Low IoU but Low Count Error",
            rows=low_rows,
            used_threshold=low_threshold,
            relaxed=low_relaxed,
            threshold_text="IoU <= {iou:.2f}, Count Error <= {count}",
        )
    )
    lines.extend(["", "Observations:"])
    if low_rows:
        lines.append("These cases suggest count estimates can remain reliable even when pixel-level overlap is poor.")
    else:
        lines.append("No clear low-IoU/low-count-error examples were found under the relaxed thresholds.")

    lines.extend(
        [
            "",
            "Interpretation:",
            "These figures are for qualitative inspection only. They do not change the model, metrics, or reliability calculations.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize anomaly cases for IoU vs count reliability.")
    parser.add_argument("--csv-path", default="outputs/urban_metrics_analysis.csv")
    parser.add_argument("--image-dir", default=str(DEFAULT_DATA_ROOT / "test/image"))
    parser.add_argument("--mask-dir", default=str(DEFAULT_DATA_ROOT / "test/label"))
    parser.add_argument("--checkpoint-path", default="checkpoints/benchmark_500/best_dice.pth")
    parser.add_argument("--output-dir", default="outputs/anomaly_analysis")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--min-cases", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    high_dir = output_dir / "high_iou_high_count_error"
    low_dir = output_dir / "low_iou_low_count_error"

    rows = load_analysis_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    high_rows, high_threshold, high_relaxed = select_group(
        rows=rows,
        group_name="high_iou_high_count_error",
        threshold_pairs=[
            (0.60, 3),
            (0.55, 3),
            (0.50, 3),
            (0.60, 2),
            (0.55, 2),
            (0.50, 2),
            (0.45, 2),
        ],
        min_cases=args.min_cases,
        max_cases=args.max_cases,
    )
    low_rows, low_threshold, low_relaxed = select_group(
        rows=rows,
        group_name="low_iou_low_count_error",
        threshold_pairs=[
            (0.30, 1),
            (0.35, 1),
            (0.40, 1),
            (0.30, 2),
            (0.35, 2),
            (0.40, 2),
            (0.45, 2),
        ],
        min_cases=args.min_cases,
        max_cases=args.max_cases,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        encoder_name=args.encoder_name,
    )
    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
    )
    sample_index = build_sample_index(dataset)

    print("Selected anomaly groups:")
    print(
        f"Group A: {len(high_rows)} cases, IoU >= {high_threshold[0]:.2f}, "
        f"Count Error >= {high_threshold[1]}, relaxed={high_relaxed}"
    )
    print(
        f"Group B: {len(low_rows)} cases, IoU <= {low_threshold[0]:.2f}, "
        f"Count Error <= {low_threshold[1]}, relaxed={low_relaxed}"
    )

    for row in high_rows:
        output_path = save_case_figure(row, dataset, sample_index, model, device, high_dir, args.threshold)
        if output_path is not None:
            print(f"Saved {output_path}")

    for row in low_rows:
        output_path = save_case_figure(row, dataset, sample_index, model, device, low_dir, args.threshold)
        if output_path is not None:
            print(f"Saved {output_path}")

    report_path = write_report(
        output_dir=output_dir,
        high_rows=high_rows,
        high_threshold=high_threshold,
        high_relaxed=high_relaxed,
        low_rows=low_rows,
        low_threshold=low_threshold,
        low_relaxed=low_relaxed,
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
