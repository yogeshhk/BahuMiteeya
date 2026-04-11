"""
GeoConvNet-2D: Convolutional Summariser for 2D Images.

Task: produce a compact summary image (H/4 × W/4) from a full-resolution
input image, preserving semantic content (class-discriminative structure)
while reducing spatial resolution by 16x in total area.

Theoretical basis:
  2D translation-equivariant convolution (G = (Z^2,+)) with strided
  downsampling produces a spatially coarser representation.  The summariser
  encodes a (H×W) image into a (H/4 × W/4) summary via two stride-2
  residual stages, then decodes back to the original resolution with
  symmetric skip connections (U-Net style).  The summary image is a
  semantically rich but spatially compressed version of the input.

Dataset: CIFAR-10 (32×32 color images, 10 classes)
Default: 4x per-dimension compression → 32×32 → 8×8 summary image

Training objective:
  L = alpha * CrossEntropy(logits, y) + (1-alpha) * MSE(recon, x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Standard residual block (He et al., 2016)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class DecoderStage2D(nn.Module):
    """Bilinear upsample × 2 + skip concatenation + residual conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = ResBlock2D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# GeoConvNet-2D: Summariser
# ---------------------------------------------------------------------------

class GeoConvNet2DSummariser(nn.Module):
    """
    Translation-equivariant 2D convolutional summariser.

    Compresses an (H×W) image to an (H/4 × W/4) summary image via two
    stride-2 residual encoder stages, then reconstructs the original via
    a symmetric bilinear U-Net decoder.

    Architecture:
        stem   (H,   C)      ← 3×3 conv, no downsampling
        enc1   (H/2, C)      ← stride-2 ResBlock
        enc2   (H/4, 2C)     ← stride-2 ResBlock  [SUMMARY FEATURES]
        ──────────────────────────────────────────────────
        summary_head: (H/4, 2C) → (H/4, 3)   [3-channel summary image]
        ──────────────────────────────────────────────────
        dec1   (H/2, C)      ← upsample + skip(enc1)
        dec0   (H,   C)      ← upsample + skip(stem)
        recon_head: (H, C) → (H, 3)            [reconstructed image]
        ──────────────────────────────────────────────────
        cls_head: GAP(enc2) → Dropout → Linear → (K,)

    Args:
        num_classes:  Number of downstream classification classes.
        in_channels:  Input image channels (3 for RGB).
        base_ch:      Base channel width.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 base_ch: int = 64):
        super().__init__()
        C = base_ch

        # Stem — no downsampling (adapted for 32×32 CIFAR)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Encoder: 2 stride-2 stages → 4x per-dimension, 16x area compression
        self.enc1 = ResBlock2D(C,   C,   stride=2)   # H/2
        self.enc2 = ResBlock2D(C,   C*2, stride=2)   # H/4  [summary level]

        # Summary head: project rich features → 3-channel summary image
        self.summary_head = nn.Conv2d(C*2, in_channels, kernel_size=1)

        # Reconstruction decoder (U-Net style, symmetric to encoder)
        self.dec1 = DecoderStage2D(C*2, C,   C)    # H/4 → H/2 + skip(enc1)
        self.dec0 = DecoderStage2D(C,   C,   C)    # H/2 → H   + skip(stem)
        self.recon_head = nn.Conv2d(C, in_channels, kernel_size=1)

        # Downstream classification head (on compressed features)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(C*2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — batch of images.
        Returns:
            summary: (B, 3, H//4, W//4) — compressed representative image.
            recon:   (B, 3, H, W)       — reconstruction of x from summary features.
            logits:  (B, num_classes)   — downstream classification logits.
        """
        s  = self.stem(x)      # (B, C,  H,   W)
        e1 = self.enc1(s)      # (B, C,  H/2, W/2)
        e2 = self.enc2(e1)     # (B, 2C, H/4, W/4)  ← summary features

        summary = self.summary_head(e2)          # (B, 3, H/4, W/4)

        d1   = self.dec1(e2, e1)                 # (B, C,  H/2, W/2)
        d0   = self.dec0(d1, s)                  # (B, C,  H,   W)
        recon = self.recon_head(d0)              # (B, 3, H,   W)

        logits = self.cls_head(e2)               # (B, K)

        return summary, recon, logits

    def summarise(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the compressed summary image. No grad tracking."""
        with torch.no_grad():
            summary, _, _ = self.forward(x)
        return summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet2DSummariser(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    summary, recon, logits = model(x)
    print(f"Input shape:   {x.shape}")            # (4, 3, 32, 32)
    print(f"Summary shape: {summary.shape}")      # (4, 3, 8, 8)
    print(f"Recon shape:   {recon.shape}")        # (4, 3, 32, 32)
    print(f"Logits shape:  {logits.shape}")       # (4, 10)
    ratio = (x.shape[2] * x.shape[3]) / (summary.shape[2] * summary.shape[3])
    print(f"Area compression: {ratio:.0f}x")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
