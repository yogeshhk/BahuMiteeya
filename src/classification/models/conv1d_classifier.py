"""
GeoConvNet-1D: Residual 1D CNN for time series classification.

Theoretical basis: 1D translation-equivariant convolution is the natural
G-equivariant operator for signals on (Z, +). See paper Section 3.
Datasets: UCR Time Series Archive (tested on ECG5000).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Residual block with two 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class GeoConvNet1D(nn.Module):
    """
    Translation-equivariant 1D CNN for time series classification.

    Architecture: three residual blocks with kernel sizes {8, 5, 3},
    global average pooling, fully-connected head.

    Args:
        in_channels: Number of input channels (1 for univariate TS).
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlock1D(in_channels, 64, kernel_size=8),
            ResBlock1D(64, 128, kernel_size=5),
            ResBlock1D(128, 128, kernel_size=3),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global average pool → (B, 128, 1)
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — batch of time series.
        Returns:
            logits: (B, num_classes)
        """
        return self.head(self.blocks(x))


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet1D(in_channels=1, num_classes=5)
    x = torch.randn(8, 1, 140)  # ECG5000: 140 timesteps
    out = model(x)
    print(f"GeoConvNet-1D output shape: {out.shape}")  # (8, 5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
