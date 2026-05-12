"""Visualization helpers for training curves, samples, and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def save_generated_images(images: torch.Tensor, filepath: str | os.PathLike[str], nrow: int = 8) -> None:
    """Save a batch of normalized images to disk as a grid."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=nrow, normalize=True)


def show_image_grid(images: torch.Tensor, nrow: int = 8, title: Optional[str] = None) -> None:
    """Display a batch of images as a grid using Matplotlib."""

    grid = make_grid(images, nrow=nrow, normalize=True)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(np_grid)
    plt.show()


def plot_training_curves(history: list[dict[str, float]], filepath: str | os.PathLike[str]) -> None:
    """Save train/validation loss and accuracy curves."""

    if not history:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss" in history[0]:
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    if "val_loss" in history[0]:
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    if "loss_d" in history[0]:
        axes[0].plot(epochs, [row["loss_d"] for row in history], label="discriminator")
    if "loss_g" in history[0]:
        axes[0].plot(epochs, [row["loss_g"] for row in history], label="generator")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "train_accuracy" in history[0]:
        axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    if "val_accuracy" in history[0]:
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    filepath: str | os.PathLike[str],
) -> None:
    """Save a confusion matrix heatmap."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = values.max() / 2.0 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                format(values[i, j], "d"),
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_confusion_matrix",
    "plot_training_curves",
    "save_generated_images",
    "show_image_grid",
]
