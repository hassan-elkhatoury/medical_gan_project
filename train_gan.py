"""
Train a class-conditional DCGAN on prepared lung cancer histopathology folders.

Example:
    python train_gan.py --train_dir data/lung_cancer/train --epochs 50 --image_size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torchvision.utils import save_image

from data import MedicalImageDataset, build_transforms, discover_class_names, set_seed
from gan import ConditionalDiscriminator, ConditionalGenerator, weights_init


def train_gan(
    train_dir: str,
    output_dir: str = "checkpoints",
    out_name: str = "lung_dcgan.pth",
    sample_dir: str = "generated_samples",
    epochs: int = 50,
    batch_size: int = 64,
    image_size: int = 128,
    nz: int = 100,
    ngf: int = 64,
    ndf: int = 64,
    lr: float = 0.0002,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[str] = None,
) -> Path:
    """Train and save a conditional DCGAN checkpoint."""

    set_seed(seed)
    output_path = Path(output_dir)
    sample_path = Path(sample_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    class_names = discover_class_names(train_dir)
    dataset = MedicalImageDataset(
        train_dir,
        transform=build_transforms(image_size=image_size, augment=True),
        class_names=class_names,
    )
    if len(dataset) == 0:
        raise ValueError(f"No images found in {train_dir}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    num_classes = len(class_names)
    netG = ConditionalGenerator(
        nz=nz, ngf=ngf, nc=3, num_classes=num_classes, image_size=image_size
    ).to(device)
    netD = ConditionalDiscriminator(
        nc=3, ndf=ndf, num_classes=num_classes, image_size=image_size
    ).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_labels = (torch.arange(36, device=device) % num_classes).long()
    fixed_noise = torch.randn(fixed_labels.size(0), nz, 1, 1, device=device)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        netD.train()
        netG.train()
        loss_d_total = 0.0
        loss_g_total = 0.0
        batches = 0

        for real_images, labels in loader:
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch = real_images.size(0)
            real_targets = torch.ones(batch, device=device)
            fake_targets = torch.zeros(batch, device=device)

            netD.zero_grad(set_to_none=True)
            real_scores = netD(real_images, labels)
            loss_d_real = criterion(real_scores, real_targets)

            noise = torch.randn(batch, nz, 1, 1, device=device)
            fake_images = netG(noise, labels)
            fake_scores = netD(fake_images.detach(), labels)
            loss_d_fake = criterion(fake_scores, fake_targets)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            netG.zero_grad(set_to_none=True)
            gen_scores = netD(fake_images, labels)
            loss_g = criterion(gen_scores, real_targets)
            loss_g.backward()
            optimizerG.step()

            loss_d_total += loss_d.item()
            loss_g_total += loss_g.item()
            batches += 1

        avg_d = loss_d_total / max(1, batches)
        avg_g = loss_g_total / max(1, batches)
        history.append({"epoch": epoch, "loss_d": avg_d, "loss_g": avg_g})
        print(f"Epoch [{epoch}/{epochs}] Loss_D: {avg_d:.4f} Loss_G: {avg_g:.4f}")

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).cpu()
            save_image(
                fake,
                sample_path / f"gan_epoch_{epoch:03d}.png",
                normalize=True,
                nrow=6,
            )

        checkpoint = {
            "generator_state_dict": netG.state_dict(),
            "discriminator_state_dict": netD.state_dict(),
            "class_names": class_names,
            "conditional": True,
            "image_size": image_size,
            "nz": nz,
            "ngf": ngf,
            "ndf": ndf,
            "num_classes": num_classes,
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, output_path / out_name)
        torch.save(checkpoint, output_path / "generator.pth")

    with (output_path / "gan_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return output_path / out_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional DCGAN")
    parser.add_argument("--train_dir", default="data/lung_cancer/train")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--out_name", default="lung_dcgan.pth")
    parser.add_argument("--sample_dir", default="generated_samples")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128, choices=[64, 128])
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_gan(
        train_dir=args.train_dir,
        output_dir=args.out_dir,
        out_name=args.out_name,
        sample_dir=args.sample_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        nz=args.nz,
        ngf=args.ngf,
        ndf=args.ndf,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
