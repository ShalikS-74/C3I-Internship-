"""Training script for binary building footprint segmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BuildingFootprintDataset, split_dataset
from metrics import dice_score, iou_score
from model import get_model, SUPPORTED_MODELS


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
        targets_bin = (targets > 0.5).float()
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets_bin).sum(dim=dims)
        total = probabilities.sum(dim=dims) + targets_bin.sum(dim=dims)
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

        # Guard against tuple output (aux_params safety)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

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

            # Guard against tuple output (aux_params safety)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

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
    model_name: str = "fcn_scse",
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
            "model_name": model_name,
        },
        checkpoint_path,
    )


def load_checkpoint_for_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[int, float]:
    """
    Resume training from a checkpoint saved by save_checkpoint().

    Returns:
        (start_epoch, best_dice) so the training loop can continue correctly.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # weights_only=False required — checkpoint contains non-tensor metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Strip DataParallel prefix if present
    state_dict = checkpoint["model_state_dict"]
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_dice = checkpoint.get("best_dice", -1.0)

    print(f"[Resume] Loaded checkpoint: {checkpoint_path}")
    print(f"[Resume] Resuming from epoch {start_epoch}, best_dice={best_dice:.4f}")

    return start_epoch, best_dice


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    epochs: int = 10,
    lr: float = 1e-4,
    checkpoint_path: str | Path = "best_model_buildings.pth",
    threshold: float = 0.5,
    encoder_name: str = "resnet34",
    image_size: int | None = 256,
    model_name: str = "fcn_scse",
    resume_from: str | Path | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """
    Train a model and save the checkpoint with the best validation Dice.

    Args:
        model:            Instantiated nn.Module (already on CPU; moved to device here).
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        device:           torch.device or string.
        epochs:           Total number of epochs to train.
        lr:               Adam learning rate.
        checkpoint_path:  Base path; best and last checkpoints saved alongside.
        threshold:        Binarization threshold for metrics.
        encoder_name:     Saved into checkpoint metadata.
        image_size:       Saved into checkpoint metadata.
        model_name:       Saved into checkpoint metadata (required for correct reload).
        resume_from:      Optional path to a checkpoint to resume training from.

    Returns:
        (trained model, history list of per-epoch metric dicts)
    """
    device = torch.device(device)
    model.to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_dice = -1.0
    start_epoch = 1

    checkpoint_dir = Path(checkpoint_path).parent
    best_checkpoint_path = checkpoint_dir / "best_dice.pth"
    last_checkpoint_path = checkpoint_dir / "last.pth"

    # Resume support
    if resume_from is not None:
        start_epoch, best_dice = load_checkpoint_for_resume(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_from,
            device=device,
        )

    for epoch in range(start_epoch, epochs + 1):
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
            f"Train IoU:  {train_metrics['iou']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Val IoU:    {val_metrics['iou']:.4f} | "
            f"Val Dice:   {val_metrics['dice']:.4f}"
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
                model_name=model_name,
            )
            print(f"  -> Saved best model: {best_checkpoint_path}  Val Dice: {best_dice:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=last_checkpoint_path,
            epoch=epoch,
            best_dice=best_dice,
            encoder_name=encoder_name,
            image_size=image_size,
            model_name=model_name,
        )

    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a building segmentation model.")
    parser.add_argument(
        "--model",
        default="fcn_scse",
        choices=SUPPORTED_MODELS,
        help="Model architecture to train.",
    )
    parser.add_argument("--image-dir", required=True, help="Folder containing satellite images.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing binary building masks.")
    parser.add_argument(
        "--checkpoint-path",
        default="best_model_buildings.pth",
        help="Base checkpoint path; best_dice.pth and last.pth saved in same folder.",
    )
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional: limit dataset size for smoke tests.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional: path to checkpoint to resume training from.",
    )
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
        print(f"  Image: {image_path} | Mask: {mask_path}")

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
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    model = get_model(
        model_name=args.model,
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
        model_name=args.model,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
