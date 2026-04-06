"""
GeoConvNet-1D: Dilated Residual Encoder-Decoder for 1D Motif Segmentation.

Task: per-timestep dense labeling — assign one of K semantic classes to
every timestep t in a signal, identifying where each motif class occurs.

Theoretical basis:
  1D translation-equivariant convolution (G = (Z,+)) is the natural
  operator for signals on the integers. Dilation exponentially expands
  the receptive field without losing temporal resolution, enabling
  detection of long-range repeated patterns (motifs). The decoder
  mirrors the encoder with transposed convolutions and skip connections,
  restoring full per-timestep label resolution.

Datasets:
  - UCR ECG5000 with motif-level segmentation labels (5-class)
  - Synthetic motif dataset (3 motif types in Gaussian noise)

Usage:
  model = GeoConvNet1D(in_channels=1, num_classes=5)
  x = torch.randn(B, 1, T)
  logits = model(x)  # (B, num_classes, T)  — per-timestep logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock1D(nn.Module):
    """
    Dilated residual block for sequence feature extraction.

    Uses two dilated 1D convolutions with the same dilation rate,
    preserving temporal resolution while expanding receptive field.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)  # residual; channels unchanged


class EncoderStage1D(nn.Module):
    """Encoder stage: optional downsampling conv + dilated residual block."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int, downsample: bool = True):
        super().__init__()
        self.down = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
            if downsample else
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
        )
        self.block = DilatedResBlock1D(out_ch, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return self.block(self.down(x))


class DecoderStage1D(nn.Module):
    """Decoder stage: upsample + concatenate skip + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align lengths in case of rounding
        if x.size(2) != skip.size(2):
            x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class GeoConvNet1D(nn.Module):
    """
    Dilated 1D encoder-decoder for per-timestep motif segmentation.

    Encoder: 4 stages with exponentially increasing dilation {1,2,4,8}
             + stride-2 downsampling at each stage.
    Decoder: 4 symmetric upsampling stages with skip connections.
    Head:    1x1 conv → per-timestep K-class logits.

    Receptive field (per encoder stage, dilation d, kernel k=3):
      RF_stage = (k-1)*d + 1 = 2*d + 1
      Total RF ≈ 75 timesteps across 4 stages.

    Args:
        in_channels:  Input channels (1 for univariate time series).
        num_classes:  Number of motif classes including background.
        base_ch:      Base channel width (default 64).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_ch: int = 64):
        super().__init__()
        C = base_ch

        # Stem — no downsampling, no dilation
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Encoder: (channels, dilation, downsample)
        self.enc1 = EncoderStage1D(C,    C,    dilation=1,  downsample=True)   # T/2
        self.enc2 = EncoderStage1D(C,    C*2,  dilation=2,  downsample=True)   # T/4
        self.enc3 = EncoderStage1D(C*2,  C*4,  dilation=4,  downsample=True)   # T/8
        self.enc4 = EncoderStage1D(C*4,  C*8,  dilation=8,  downsample=True)   # T/16

        # Decoder
        self.dec3 = DecoderStage1D(C*8, C*4, C*4)   # T/8
        self.dec2 = DecoderStage1D(C*4, C*2, C*2)   # T/4
        self.dec1 = DecoderStage1D(C*2, C,   C)     # T/2
        self.dec0 = DecoderStage1D(C,   C,   C)     # T

        # Per-timestep segmentation head
        self.head = nn.Conv1d(C, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) — batch of time series.
        Returns:
            logits: (B, num_classes, T) — per-timestep class logits.
        """
        s  = self.stem(x)   # (B, C, T)
        e1 = self.enc1(s)   # (B, C,   T/2)
        e2 = self.enc2(e1)  # (B, 2C,  T/4)
        e3 = self.enc3(e2)  # (B, 4C,  T/8)
        e4 = self.enc4(e3)  # (B, 8C,  T/16)

        d3 = self.dec3(e4, e3)  # (B, 4C,  T/8)
        d2 = self.dec2(d3, e2)  # (B, 2C,  T/4)
        d1 = self.dec1(d2, e1)  # (B, C,   T/2)
        d0 = self.dec0(d1, s)   # (B, C,   T)

        return self.head(d0)    # (B, num_classes, T)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet1D(in_channels=1, num_classes=5)
    T = 140
    x = torch.randn(4, 1, T)
    logits = model(x)
    print(f"GeoConvNet-1D segmentation output: {logits.shape}")   # (4, 5, 140)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Compute predicted segmentation mask
    mask = logits.argmax(dim=1)
    print(f"Predicted mask shape: {mask.shape}")                   # (4, 140)
