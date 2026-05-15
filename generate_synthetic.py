"""
Generate class-organized synthetic COVID-19 radiography images from a trained conditional DCGAN.

Example:
    python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth \
        --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from data import MedicalImageDataset, discover_class_names, set_seed
from gan import ConditionalGenerator


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Find the requested generator checkpoint or explain what is missing."""

    path = Path(checkpoint_path)
    if path.exists():
        return path

    candidates = [
        path.parent / "generator.pth",
        path.parent / "covid_dcgan.pth",
        path.parent / "lung_dcgan.pth",
    ]
    candidates.extend(sorted(path.parent.glob("*dcgan*.pth")))
    candidates.extend(sorted(path.parent.glob("*generator*.pth")))

    for candidate in candidates:
        if candidate.exists():
            print(f"Requested checkpoint not found: {path}")
            print(f"Using available generator checkpoint instead: {candidate}")
            return candidate

    available = sorted(path.parent.glob("*.pth")) if path.parent.exists() else []
    if available:
        available_text = "\n".join(f"  - {candidate}" for candidate in available)
    else:
        available_text = "  - no .pth files found"
    raise FileNotFoundError(
        "Generator checkpoint is missing.\n"
        f"Requested: {path}\n"
        f"Available in {path.parent}:\n{available_text}\n"
        "Run the GAN training cell again and make sure it finishes successfully."
    )


def load_generator(checkpoint_path: str | Path, device: str) -> tuple[ConditionalGenerator, dict]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "generator_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint with generator_state_dict")

    model = ConditionalGenerator(
        nz=checkpoint.get("nz", 100),
        ngf=checkpoint.get("ngf", 64),
        nc=3,
        num_classes=checkpoint.get("num_classes", len(checkpoint.get("class_names", []))),
        image_size=checkpoint.get("image_size", 128),
    ).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    return model, checkpoint


def class_counts(real_dir: str | Path, class_names: list[str]) -> dict[str, int]:
    dataset = MedicalImageDataset(real_dir, transform=None, class_names=class_names)
    counts = {name: 0 for name in class_names}
    for label in dataset.labels:
        counts[class_names[label]] += 1
    return counts


def generate_synthetic_images(
    checkpoint: str | Path = "checkpoints/covid_dcgan.pth",
    out_dir: str | Path = "synthetic_covid_images",
    samples_per_class: int | None = None,
    match_real_dir: str | Path | None = "data/covid_radiography/train",
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> None:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator, metadata = load_generator(checkpoint, device)
    class_names = metadata.get("class_names") or discover_class_names(match_real_dir or "")
    nz = metadata.get("nz", 100)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if samples_per_class is None:
        if match_real_dir is None:
            raise ValueError("Provide --samples_per_class or --match_real_dir")
        counts = class_counts(match_real_dir, class_names)
    else:
        counts = {name: samples_per_class for name in class_names}

    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            target_dir = output_root / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            total = counts[class_name]
            written = 0
            while written < total:
                current_batch = min(batch_size, total - written)
                labels = torch.full((current_batch,), class_idx, dtype=torch.long, device=device)
                noise = torch.randn(current_batch, nz, 1, 1, device=device)
                images = generator(noise, labels).cpu()
                for offset, image in enumerate(images):
                    save_image(
                        image,
                        target_dir / f"{class_name}_{written + offset:06d}.png",
                        normalize=True,
                    )
                written += current_batch
            print(f"Generated {total} images for {class_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic COVID-19 radiography images")
    parser.add_argument("--checkpoint", default="checkpoints/covid_dcgan.pth")
    parser.add_argument("--out_dir", default="synthetic_covid_images")
    parser.add_argument("--samples_per_class", type=int, default=None)
    parser.add_argument("--match_real_dir", default="data/covid_radiography/train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_images(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        samples_per_class=args.samples_per_class,
        match_real_dir=args.match_real_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
