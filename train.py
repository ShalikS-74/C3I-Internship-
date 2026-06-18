"""Training script for binary building footprint segmentation using FCN + scSE."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BuildingFootprintDataset, split_dataset
from metrics import dice_score, iou_score
from model import get_fcn_scse_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _epoch_average(total: dict[str, float], sample_count: int) -> dict[str, float]:
    return {key: value / max(sample_count, 1) for key, value in total.items()}


class DiceBCELoss(torch.nn.Module):
    """Combined Dice loss and BCEWithLogitsLoss for binary segmentation."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        targets = (targets > 0.5).float()
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        total = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (total + self.eps)).mean()

        return dice_loss + bce_loss


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    sample_count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            totals["loss"] += loss.item() * batch_size
            totals["iou"] += iou_score(logits, masks, threshold=threshold).item() * batch_size
            totals["dice"] += dice_score(logits, masks, threshold=threshold).item() * batch_size
            sample_count += batch_size

    return _epoch_average(totals, sample_count)


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks)

            totals["loss"] += loss.item() * batch_size
            totals["iou"] += iou_score(logits, masks, threshold=threshold).item() * batch_size
            totals["dice"] += dice_score(logits, masks, threshold=threshold).item() * batch_size
            sample_count += batch_size

    return _epoch_average(totals, sample_count)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    epoch: int,
    best_dice: float,
    encoder_name: str,
    image_size: int | None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "encoder_name": encoder_name,
            "image_size": image_size,
        },
        checkpoint_path,
    )


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    epochs: int = 10,
    lr: float = 1e-4,
    checkpoint_path: str | Path = "best_fcn_scse_buildings.pth",
    threshold: float = 0.5,
    encoder_name: str = "resnet34",
    image_size: int | None = 256,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """Train a model and save the checkpoint with the best validation Dice."""

    device = torch.device(device)
    model.to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_dice = -1.0
    checkpoint_dir = Path(checkpoint_path).parent
    best_checkpoint_path = checkpoint_dir / "best_dice.pth"
    last_checkpoint_path = checkpoint_dir / "last.pth"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
                best_dice=best_dice,
                encoder_name=encoder_name,
                image_size=image_size,
            )
            print(f"Saved best model to {best_checkpoint_path} with Val Dice: {best_dice:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=last_checkpoint_path,
            epoch=epoch,
            best_dice=best_dice,
            encoder_name=encoder_name,
            image_size=image_size,
        )

    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FCN + scSE building segmentation model.")
    parser.add_argument("--image-dir", required=True, help="Folder containing satellite images.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing binary building masks.")
    parser.add_argument("--checkpoint-path", default="best_fcn_scse_buildings.pth")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test subset size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = BuildingFootprintDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    print(f"Matched {len(dataset)} image/mask pairs.")
    for image_path, mask_path in dataset.pairs[:5]:
        print(f"Image: {image_path} | Mask: {mask_path}")

    train_dataset, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_fcn_scse_model(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        threshold=args.threshold,
        encoder_name=args.encoder_name,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
