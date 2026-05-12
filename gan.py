"""
DCGAN models for dermoscopic image generation.

The conditional variants are used by default so a single GAN can generate
samples for all seven HAM10000 lesion classes.
"""

from __future__ import annotations

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    """Initialize convolution and batch-normalization layers for DCGAN."""

    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def _generator_layers(input_channels: int, ngf: int, nc: int, image_size: int) -> nn.Sequential:
    if image_size not in {64, 128}:
        raise ValueError("DCGAN generator supports image_size 64 or 128")

    layers: list[nn.Module] = [
        nn.ConvTranspose2d(input_channels, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
    ]
    if image_size == 128:
        layers.extend(
            [
                nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf // 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            ]
        )
    else:
        layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _discriminator_layers(input_channels: int, ndf: int, image_size: int) -> nn.Sequential:
    if image_size not in {64, 128}:
        raise ValueError("DCGAN discriminator supports image_size 64 or 128")

    layers: list[nn.Module] = [
        nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    if image_size == 128:
        layers.extend(
            [
                nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
    layers.extend(
        [
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        ]
    )
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Unconditional DCGAN generator."""

    def __init__(
        self,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 3,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.image_size = image_size
        self.main = _generator_layers(nz, ngf, nc, image_size)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.main(noise)


class Discriminator(nn.Module):
    """Unconditional DCGAN discriminator."""

    def __init__(self, nc: int = 3, ndf: int = 64, image_size: int = 128) -> None:
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.image_size = image_size
        self.main = _discriminator_layers(nc, ndf, image_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.main(image).view(-1)


class ConditionalGenerator(nn.Module):
    """Class-conditional DCGAN generator."""

    def __init__(
        self,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 3,
        num_classes: int = 7,
        embedding_dim: int = 50,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.main = _generator_layers(nz + embedding_dim, ngf, nc, image_size)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_embedding(labels).view(labels.size(0), self.embedding_dim, 1, 1)
        conditioned = torch.cat([noise, label_vec], dim=1)
        return self.main(conditioned)


class ConditionalDiscriminator(nn.Module):
    """Class-conditional DCGAN discriminator."""

    def __init__(
        self,
        nc: int = 3,
        ndf: int = 64,
        num_classes: int = 7,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        self.main = _discriminator_layers(nc + 1, ndf, image_size)

    def forward(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_embedding(labels).view(labels.size(0), 1, self.image_size, self.image_size)
        conditioned = torch.cat([image, label_map], dim=1)
        return self.main(conditioned).view(-1)


__all__ = [
    "ConditionalDiscriminator",
    "ConditionalGenerator",
    "Discriminator",
    "Generator",
    "weights_init",
]
