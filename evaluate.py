"""Evaluate a saved classifier checkpoint on real COVID-19 radiography test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from classifier import SimpleCNN
from data import MedicalImageDataset, build_transforms, discover_class_names
from utils import plot_confusion_matrix


def evaluate_classifier(
    checkpoint_path: str = "checkpoints/covid_classifier.pth",
    test_dir: str = "data/covid_radiography/test",
    output_json: str = "checkpoints/evaluation_metrics.json",
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names") or discover_class_names(test_dir)
    image_size = checkpoint.get("image_size", 128)
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    dataset = MedicalImageDataset(
        test_dir,
        transform=build_transforms(image_size=image_size, augment=False),
        class_names=class_names,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels_all: list[int] = []
    preds_all: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds_all.extend(outputs.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(labels_all, preds_all),
        "classification_report": classification_report(
            labels_all,
            preds_all,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels_all, preds_all).tolist(),
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_path.with_suffix(".png"),
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/covid_classifier.pth")
    parser.add_argument("--test_dir", default="data/covid_radiography/test")
    parser.add_argument("--output_json", default="checkpoints/evaluation_metrics.json")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    evaluate_classifier(args.checkpoint, args.test_dir, args.output_json, args.batch_size)


if __name__ == "__main__":
    main()
