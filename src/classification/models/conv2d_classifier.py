"""
GeoConvNet-2D: Lightweight ResNet-18 variant for image classification.

Theoretical basis: 2D translation-equivariant convolution is the natural
G-equivariant operator for signals on (Z^2, +). See paper Section 3.
Dataset: CIFAR-10 (32x32 color images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """Standard residual block (He et al., 2016)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class GeoConvNet2D(nn.Module):
    """
    ResNet-18-style 2D CNN adapted for CIFAR-10.

    For images, the group is (Z^2, +): the convolution is 2D translation-
    equivariant. Hierarchical pooling via strided convolutions reduces the
    spatial domain at each level.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).
        in_channels: Input image channels (3 for RGB).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Stem — smaller kernel for 32×32 CIFAR (no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layer(in_c: int, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResBlock2D(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock2D(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image batch.
        Returns:
            logits: (B, num_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet2D(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"GeoConvNet-2D output shape: {out.shape}")  # (4, 10)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
