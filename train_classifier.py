"""
Train a CNN classifier with real COVID-19 radiography images and optional GAN-generated images.

Examples:
    python train_classifier.py --real_train_dir data/covid_radiography/train \
        --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test

    python train_classifier.py --real_train_dir data/covid_radiography/train \
        --test_dir data/covid_radiography/test --out_name baseline_covid_classifier.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader

from classifier import SimpleCNN
from data import ImagePathDataset, MedicalImageDataset, build_transforms, discover_class_names, set_seed
from utils import plot_confusion_matrix, plot_training_curves


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    class_names: list[str],
) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def train_classifier(
    real_train_dir: str = "data/covid_radiography/train",
    synthetic_train_dir: Optional[str] = "synthetic_covid_images",
    test_dir: str = "data/covid_radiography/test",
    output_dir: str = "checkpoints",
    out_name: str = "covid_classifier.pth",
    epochs: int = 20,
    batch_size: int = 64,
    image_size: int = 128,
    lr: float = 0.0002,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[str] = None,
) -> Path:
    """Train, validate, test, and save a classifier checkpoint."""

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("generated_samples")
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_class_names(real_train_dir, test_dir, synthetic_train_dir or "")
    train_transform = build_transforms(image_size=image_size, augment=True)
    eval_transform = build_transforms(image_size=image_size, augment=False)

    real_dataset = MedicalImageDataset(real_train_dir, transform=None, class_names=class_names)
    image_paths = list(real_dataset.image_paths)
    labels = list(real_dataset.labels)
    if synthetic_train_dir and Path(synthetic_train_dir).is_dir():
        synth_dataset = MedicalImageDataset(
            synthetic_train_dir, transform=None, class_names=class_names
        )
        if len(synth_dataset) > 0:
            image_paths.extend(synth_dataset.image_paths)
            labels.extend(synth_dataset.labels)

    if not image_paths:
        raise ValueError(f"No training images found in {real_train_dir}")

    val_size = max(1, int(len(image_paths) * val_fraction))
    train_size = len(image_paths) - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(image_paths), generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    train_dataset = ImagePathDataset(
        [image_paths[idx] for idx in train_indices],
        [labels[idx] for idx in train_indices],
        class_names=class_names,
        transform=train_transform,
    )
    val_dataset = ImagePathDataset(
        [image_paths[idx] for idx in val_indices],
        [labels[idx] for idx in val_indices],
        class_names=class_names,
        transform=eval_transform,
    )

    test_dataset = MedicalImageDataset(test_dir, transform=eval_transform, class_names=class_names)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    history: list[dict[str, float]] = []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_model(model, test_loader, device, class_names)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
        "num_classes": len(class_names),
        "history": history,
        "test_metrics": metrics,
        "used_synthetic": bool(synthetic_train_dir and Path(synthetic_train_dir).is_dir()),
    }
    checkpoint_path = output_path / out_name
    torch.save(checkpoint, checkpoint_path)
    if out_name == "covid_classifier.pth":
        torch.save(checkpoint, "covid_classifier.pth")

    with (output_path / f"{Path(out_name).stem}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_path / f"{Path(out_name).stem}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    plot_training_curves(history, figures_dir / f"{Path(out_name).stem}_curves.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        figures_dir / f"{Path(out_name).stem}_confusion_matrix.png",
    )

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved classifier checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train COVID-19 chest radiography classifier")
    parser.add_argument("--real_train_dir", default="data/covid_radiography/train")
    parser.add_argument("--synthetic_train_dir", default="synthetic_covid_images")
    parser.add_argument("--test_dir", default="data/covid_radiography/test")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--out_name", default="covid_classifier.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    synthetic_dir = args.synthetic_train_dir
    if synthetic_dir and synthetic_dir.lower() in {"none", "null", "baseline"}:
        synthetic_dir = None

    train_classifier(
        real_train_dir=args.real_train_dir,
        synthetic_train_dir=synthetic_dir,
        test_dir=args.test_dir,
        output_dir=args.out_dir,
        out_name=args.out_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
