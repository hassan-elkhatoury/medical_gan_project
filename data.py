"""
Dataset utilities for HAM10000 / Skin Cancer MNIST experiments.

The project uses a folder layout after preparation:

    data/ham10000/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg

Images are resized by torchvision transforms at training time and normalized to
[-1, 1], which is the range expected by the DCGAN generator/discriminator.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


HAM10000_LABELS: dict[str, str] = {
    "akiec": "actinic_keratoses",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic_nevi",
    "vasc": "vascular_lesions",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible data splits and training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms(
    image_size: int = 128,
    augment: bool = False,
    normalize: bool = True,
) -> transforms.Compose:
    """Build the standard image transform pipeline."""

    steps: list[object] = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
            ]
        )
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(steps)


class SkinLesionDataset(Dataset):
    """Dataset for skin-lesion images stored in class-specific folders."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        if class_names is None:
            class_names = [
                p.name for p in sorted(self.data_dir.iterdir()) if p.is_dir()
            ]
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class ImagePathDataset(Dataset):
    """Dataset from explicit image paths and integer labels."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        class_names: list[str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class SingleClassDataset(Dataset):
    """View of a folder dataset restricted to one class label."""

    def __init__(self, base_dataset: SkinLesionDataset, class_index: int) -> None:
        self.base_dataset = base_dataset
        self.class_index = class_index
        self.indices = [
            idx for idx, label in enumerate(base_dataset.labels) if label == class_index
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.base_dataset[self.indices[idx]]


def discover_class_names(*roots: str | os.PathLike[str]) -> list[str]:
    """Return sorted class names from the first existing folder tree."""

    for root in roots:
        path = Path(root)
        if path.is_dir():
            names = [p.name for p in sorted(path.iterdir()) if p.is_dir()]
            if names:
                return names
    return sorted(HAM10000_LABELS.values())


def get_dataloaders(
    train_dir: str | os.PathLike[str],
    test_dir: str | os.PathLike[str],
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
    augment_train: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Return training and test dataloaders for prepared folder data."""

    class_names = discover_class_names(train_dir, test_dir)
    train_dataset = SkinLesionDataset(
        train_dir,
        transform=build_transforms(image_size=image_size, augment=augment_train),
        class_names=class_names,
    )
    test_dataset = SkinLesionDataset(
        test_dir,
        transform=build_transforms(image_size=image_size, augment=False),
        class_names=class_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    return train_loader, test_loader


def find_ham10000_image_dirs(raw_dir: str | os.PathLike[str]) -> list[Path]:
    """Find HAM10000 image directories under an extracted Kaggle dataset."""

    raw_path = Path(raw_dir)
    candidates = [
        raw_path / "HAM10000_images_part_1",
        raw_path / "HAM10000_images_part_2",
        raw_path / "ham10000_images_part_1",
        raw_path / "ham10000_images_part_2",
    ]
    dirs = [path for path in candidates if path.is_dir()]
    if dirs:
        return dirs

    return [
        path
        for path in raw_path.rglob("*")
        if path.is_dir() and any(child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())
    ]


def build_image_index(image_dirs: Iterable[Path]) -> dict[str, Path]:
    """Map HAM image ids to their image file paths."""

    index: dict[str, Path] = {}
    for image_dir in image_dirs:
        for path in image_dir.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                index[path.stem] = path
    return index


def prepare_ham10000(
    raw_dir: str | os.PathLike[str] = "data/ham10000_raw",
    output_dir: str | os.PathLike[str] = "data/ham10000",
    test_size: float = 0.2,
    seed: int = 42,
    resize: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Create train/test class folders from the extracted HAM10000 dataset."""

    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    metadata_path = raw_path / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        matches = list(raw_path.rglob("HAM10000_metadata.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Could not find HAM10000_metadata.csv under {raw_path}"
            )
        metadata_path = matches[0]

    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    required_columns = {"image_id", "dx"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["class_name"] = metadata["dx"].map(HAM10000_LABELS).fillna(metadata["dx"])

    train_df, test_df = train_test_split(
        metadata,
        test_size=test_size,
        random_state=seed,
        stratify=metadata["class_name"],
    )

    image_index = build_image_index(find_ham10000_image_dirs(raw_path))
    if not image_index:
        raise FileNotFoundError(f"No HAM10000 image files found under {raw_path}")

    def copy_split(split_name: str, split_df: pd.DataFrame) -> None:
        records = split_df[["image_id", "class_name"]].to_dict("records")
        for record in records:
            image_id = str(record["image_id"])
            class_name = str(record["class_name"])
            source = image_index.get(image_id)
            if source is None:
                raise FileNotFoundError(f"Image {image_id} listed in metadata is missing")

            target_dir = output_path / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name

            if resize is None:
                shutil.copy2(source, target)
            else:
                image = Image.open(source).convert("RGB")
                image = image.resize((resize, resize), Image.BILINEAR)
                image.save(target)

    copy_split("train", train_df)
    copy_split("test", test_df)

    class_names = sorted(metadata["class_name"].unique().tolist())
    pd.Series(class_names, name="class_name").to_csv(
        output_path / "classes.csv", index=False
    )
    metadata.to_csv(output_path / "metadata_prepared.csv", index=False)


__all__ = [
    "HAM10000_LABELS",
    "ImagePathDataset",
    "SkinLesionDataset",
    "SingleClassDataset",
    "build_transforms",
    "discover_class_names",
    "get_dataloaders",
    "prepare_ham10000",
    "set_seed",
]
