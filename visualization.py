"""Matplotlib visualization helpers for dataset samples and predictions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import (
    binary_mask_from_logits,
    binary_mask_to_uint8,
    building_coverage_percentage,
    count_buildings,
)


def _to_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)


def show_training_samples(dataset, num_samples: int = 3) -> None:
    """Display original images and ground truth masks before training."""

    sample_count = min(num_samples, len(dataset))
    fig, axes = plt.subplots(sample_count, 2, figsize=(8, 4 * sample_count), squeeze=False)

    for row in range(sample_count):
        sample = dataset[row]
        image = _to_numpy_image(sample["image"])
        mask = binary_mask_to_uint8(sample["mask"])

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Ground truth mask")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


def display_prediction(
    image: np.ndarray | torch.Tensor,
    ground_truth_mask: np.ndarray | torch.Tensor | None,
    predicted_mask: np.ndarray | torch.Tensor,
    title: str | None = None,
) -> None:
    """Display original image, optional ground truth, and predicted mask."""

    panels = [("Original image", _to_numpy_image(image))]
    if ground_truth_mask is not None:
        panels.append(("Ground truth mask", binary_mask_to_uint8(ground_truth_mask)))
    panels.append(("Predicted mask", binary_mask_to_uint8(predicted_mask)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    axes = np.atleast_1d(axes)

    for axis, (panel_title, panel_image) in zip(axes, panels):
        if panel_title == "Original image":
            axis.imshow(panel_image)
        else:
            axis.imshow(panel_image, cmap="gray")
        axis.set_title(panel_title)
        axis.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def show_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    num_samples: int = 3,
    threshold: float = 0.5,
    min_area: int = 20,
) -> None:
    """Run inference on validation samples and display predictions."""

    device = torch.device(device)
    model.eval()
    shown = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)

            for item_index in range(images.size(0)):
                pred_mask = binary_mask_from_logits(logits[item_index], threshold=threshold)
                building_count = count_buildings(pred_mask, min_area=min_area)
                coverage = building_coverage_percentage(pred_mask)

                display_prediction(
                    image=batch["image"][item_index],
                    ground_truth_mask=batch["mask"][item_index],
                    predicted_mask=pred_mask,
                    title=f"Building Count = {building_count} | Building Coverage = {coverage:.2f}%",
                )

                shown += 1
                if shown >= num_samples:
                    return
