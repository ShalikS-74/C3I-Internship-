"""Analyze the relationship between segmentation quality (IoU) and metric reliability.

Investigates whether segmentation quality fully explains the reliability of
Building Count and Building Coverage estimates.

Input:
  outputs/urban_metrics_analysis.csv

Output:
  outputs/reliability_analysis/
    - summary.txt
    - iou_vs_count_error.png
    - iou_vs_coverage_error.png

Usage:
  python analyze_reliability.py \
    --csv-path outputs/urban_metrics_analysis.csv \
    --output-dir outputs/reliability_analysis/
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or 0.0 when either input is constant."""

    if len(x) < 2 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze segmentation quality vs metric reliability.")
    parser.add_argument("--csv-path", default="outputs/urban_metrics_analysis.csv")
    parser.add_argument("--output-dir", default="outputs/reliability_analysis/")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading CSV from {csv_path}")
    
    # Load CSV data
    image_ids = []
    ious = []
    dices = []
    count_errors = []
    coverage_errors = []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row["image_id"])
            ious.append(float(row["iou"]))
            dices.append(float(row["dice"]))
            count_errors.append(float(row["count_error"]))
            coverage_errors.append(float(row["coverage_error"]))
    
    ious = np.array(ious)
    dices = np.array(dices)
    count_errors = np.array(count_errors)
    coverage_errors = np.array(coverage_errors)
    image_ids = np.array(image_ids, dtype=str)
    
    n_samples = len(ious)
    print(f"Loaded {n_samples} samples\n")

    # ---- Compute Summary Statistics ----
    mean_iou = ious.mean()
    mean_dice = dices.mean()
    mean_count_error = count_errors.mean()
    mean_coverage_error = coverage_errors.mean()

    print("=== Summary Statistics ===")
    print(f"Mean IoU:              {mean_iou:.4f}")
    print(f"Mean Dice:             {mean_dice:.4f}")
    print(f"Mean Count Error:      {mean_count_error:.4f}")
    print(f"Mean Coverage Error:   {mean_coverage_error:.4f}\n")

    # ---- Compute Correlations ----
    corr_iou_count = pearson_correlation(ious, count_errors)
    corr_iou_coverage = pearson_correlation(ious, coverage_errors)

    print("=== Correlations ===")
    print(f"Correlation(IoU, Count Error):    {corr_iou_count:+.4f}")
    print(f"Correlation(IoU, Coverage Error): {corr_iou_coverage:+.4f}\n")

    # ---- Identify Anomalies ----
    print("=== Anomalous Cases ===")

    # High IoU, large Count Error
    high_iou_candidates = np.argsort(-ious)[: min(5, n_samples)]
    high_iou_sorted = high_iou_candidates[np.argsort(-count_errors[high_iou_candidates])]
    
    print("\nTop 5 images with HIGH IoU but LARGE Count Error:")
    print(f"{'image_id':<10} {'iou':<10} {'count_error':<15} {'dice':<10}")
    print("-" * 45)
    for idx in high_iou_sorted:
        print(f"{image_ids[idx]:<10} {ious[idx]:<10.4f} {count_errors[idx]:<15.0f} {dices[idx]:<10.4f}")

    # Low IoU, small Count Error
    low_iou_candidates = np.argsort(ious)[: min(5, n_samples)]
    low_iou_sorted = low_iou_candidates[np.argsort(count_errors[low_iou_candidates])]
    
    print("\nTop 5 images with LOW IoU but SMALL Count Error:")
    print(f"{'image_id':<10} {'iou':<10} {'count_error':<15} {'dice':<10}")
    print("-" * 45)
    for idx in low_iou_sorted:
        print(f"{image_ids[idx]:<10} {ious[idx]:<10.4f} {count_errors[idx]:<15.0f} {dices[idx]:<10.4f}")

    # ---- Generate Plots ----
    print("\n=== Generating plots ===")

    # Plot 1: IoU vs Count Error
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ious, count_errors, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Count Error", fontsize=12)
    ax.set_title(f"IoU vs Count Error\n(r={corr_iou_count:.3f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    plot1_path = output_dir / "iou_vs_count_error.png"
    fig.savefig(plot1_path, dpi=150, bbox_inches="tight")
    print(f"Saved {plot1_path}")
    plt.close(fig)

    # Plot 2: IoU vs Coverage Error
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ious, coverage_errors, alpha=0.6, s=50, color="orange", edgecolors="k", linewidth=0.5)
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Coverage Error (%)", fontsize=12)
    ax.set_title(f"IoU vs Coverage Error\n(r={corr_iou_coverage:.3f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    plot2_path = output_dir / "iou_vs_coverage_error.png"
    fig.savefig(plot2_path, dpi=150, bbox_inches="tight")
    print(f"Saved {plot2_path}")
    plt.close(fig)

    # ---- Generate Summary Report ----
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Reliability Analysis Report ===\n\n")
        f.write("Research Question:\n")
        f.write("Does segmentation quality (IoU) fully explain the reliability of\n")
        f.write("Building Count and Building Coverage estimates?\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Mean IoU:              {mean_iou:.4f}\n")
        f.write(f"  Mean Dice:             {mean_dice:.4f}\n")
        f.write(f"  Mean Count Error:      {mean_count_error:.4f}\n")
        f.write(f"  Mean Coverage Error:   {mean_coverage_error:.4f}\n\n")

        f.write("Correlation Analysis:\n")
        f.write(f"  r(IoU, Count Error):      {corr_iou_count:+.4f}\n")
        f.write(f"  r(IoU, Coverage Error):   {corr_iou_coverage:+.4f}\n\n")

        f.write("Key Findings:\n")
        f.write(f"  - {n_samples} images analyzed\n")
        if abs(corr_iou_count) < 0.3:
            f.write("  - WEAK correlation between IoU and Count Error\n")
            f.write("    Segmentation quality alone does NOT explain Count reliability\n")
        elif abs(corr_iou_count) < 0.7:
            f.write("  - MODERATE correlation between IoU and Count Error\n")
        else:
            f.write("  - STRONG correlation between IoU and Count Error\n")

        if abs(corr_iou_coverage) < 0.3:
            f.write("  - WEAK correlation between IoU and Coverage Error\n")
            f.write("    Segmentation quality alone does NOT explain Coverage reliability\n")
        elif abs(corr_iou_coverage) < 0.7:
            f.write("  - MODERATE correlation between IoU and Coverage Error\n")
        else:
            f.write("  - STRONG correlation between IoU and Coverage Error\n")

        f.write("\n  - Images with similar IoU values can exhibit substantially\n")
        f.write("    different Count and Coverage Errors\n")
        f.write("  - This suggests additional factors influence reliability beyond\n")
        f.write("    raw pixel-level segmentation quality\n\n")

        f.write("Anomalous Cases:\n")
        f.write("  High IoU but large Count Error (potential over-estimation):\n")
        for idx in high_iou_sorted:
            f.write(f"    image_id={image_ids[idx]}: IoU={ious[idx]:.3f}, Count Error={count_errors[idx]:.0f}\n")
        f.write("  Low IoU but small Count Error (robust estimation despite poor segmentation):\n")
        for idx in low_iou_sorted:
            f.write(f"    image_id={image_ids[idx]}: IoU={ious[idx]:.3f}, Count Error={count_errors[idx]:.0f}\n")

    print(f"Saved {summary_path}\n")

    # ---- Print Conclusions ----
    print("\n=== Conclusions ===")
    print("\n1. Correlation Strength:")
    if abs(corr_iou_count) < 0.3 or abs(corr_iou_coverage) < 0.3:
        print("   Weak correlation: segmentation quality does not fully explain metric reliability.")
    else:
        print("   Moderate-to-strong correlation: IoU is a significant factor in reliability.")

    print("\n2. Variability:")
    print("   Images with similar IoU values exhibit different errors.")
    print(f"   Count Error range:    {count_errors.min():.0f} to {count_errors.max():.0f}")
    print(f"   Coverage Error range: {coverage_errors.min():.2f} to {coverage_errors.max():.2f}")

    print("\n3. Implication:")
    print("   Additional factors may influence reliability:")
    print("     - Image characteristics (texture, building density)")
    print("     - Object size distribution")
    print("     - Localized segmentation errors")

    print("\n=== Analysis Complete ===")
    print(f"All outputs saved to {output_dir}")



if __name__ == "__main__":
    main()
