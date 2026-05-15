"""
classifier.py
-------------

This module defines a simple convolutional neural network (CNN) for
multi-class classification of COVID-19 chest X-ray images.  The architecture
consists of several convolutional blocks followed by fully connected
layers.  It is deliberately kept lightweight to allow faster training
on moderate hardware, but can be replaced with more sophisticated
architectures such as ResNet or EfficientNet as needed.
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A straightforward CNN for image classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  For the COVID-19 Radiography Database this should be 4.
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        # The number of input features to the first linear layer depends on
        # the input image size.  For 128×128 input and four 2×2 max pools
        # the spatial dimension is reduced to 8×8.  Adjust accordingly
        # if ``image_size`` in train scripts is changed.
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


__all__ = [
    "SimpleCNN",
]
