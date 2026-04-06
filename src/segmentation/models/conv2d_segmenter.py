"""
GeoConvNet-2D: U-Net with ResNet-18 Encoder for Semantic Segmentation.

Task: per-pixel dense labeling — assign one of K semantic classes to
every pixel (i,j) in an image (semantic segmentation / object detection).

Theoretical basis:
  2D translation-equivariant convolution (G = (Z^2,+)) is the natural
  operator for images. The encoder builds a 4-level feature pyramid at
  strides {2,4,8,16}; the decoder restores full spatial resolution via
  bilinear upsampling + skip connections, enabling per-pixel prediction
  of arbitrary object boundaries.

Dataset:
  PASCAL VOC 2012 — 21-class semantic segmentation (20 objects + background).

Usage:
  model = GeoConvNet2D(num_classes=21)
  x = torch.randn(B, 3, H, W)
  logits = model(x)  # (B, num_classes, H, W) — per-pixel logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ResNet-18 encoder blocks (from scratch — no torchvision dependency needed
# but weights can be loaded from torchvision.models.resnet18 if desired)
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Standard ResNet residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def make_layer(in_ch, out_ch, blocks, stride):
    layers = [ResBlock2D(in_ch, out_ch, stride=stride)]
    for _ in range(1, blocks):
        layers.append(ResBlock2D(out_ch, out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Decoder block: upsample × 2 + skip concat + conv
# ---------------------------------------------------------------------------

class DecoderBlock2D(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# GeoConvNet-2D: full U-Net segmentation model
# ---------------------------------------------------------------------------

class GeoConvNet2D(nn.Module):
    """
    U-Net with ResNet-18 encoder for per-pixel semantic segmentation.

    Encoder stages (ResNet-18):
      stem  → stride 2  : (B, 64,  H/2,  W/2)
      layer1 → stride 1 : (B, 64,  H/2,  W/2)   skip s1
      layer2 → stride 2 : (B, 128, H/4,  W/4)   skip s2
      layer3 → stride 2 : (B, 256, H/8,  W/8)   skip s3
      layer4 → stride 2 : (B, 512, H/16, W/16)  bottleneck

    Decoder: bilinear upsampling × 2 + skip concat at each stage.
    Head: 1×1 conv → per-pixel K-class logits, then bilinear to input size.

    Args:
        num_classes: Number of semantic classes (21 for PASCAL VOC).
        in_channels: Input image channels (3 for RGB).
    """

    def __init__(self, num_classes: int = 21, in_channels: int = 3):
        super().__init__()

        # --- Encoder (ResNet-18 skeleton) ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool   = nn.MaxPool2d(3, stride=2, padding=1)   # H/4, W/4
        self.layer1 = make_layer(64,  64,  blocks=2, stride=1)  # H/4
        self.layer2 = make_layer(64,  128, blocks=2, stride=2)  # H/8
        self.layer3 = make_layer(128, 256, blocks=2, stride=2)  # H/16
        self.layer4 = make_layer(256, 512, blocks=2, stride=2)  # H/32

        # --- Decoder ---
        self.dec4 = DecoderBlock2D(512, 256, 256)   # H/16
        self.dec3 = DecoderBlock2D(256, 128, 128)   # H/8
        self.dec2 = DecoderBlock2D(128,  64,  64)   # H/4
        self.dec1 = DecoderBlock2D( 64,  64,  64)   # H/2

        # --- Segmentation head ---
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — batch of RGB images.
        Returns:
            logits: (B, num_classes, H, W) — per-pixel class logits.
        """
        H, W = x.shape[-2:]

        # Encoder
        s0 = self.stem(x)           # (B, 64, H/2, W/2)
        s0p = self.pool(s0)         # (B, 64, H/4, W/4)
        s1 = self.layer1(s0p)       # (B, 64, H/4, W/4)   skip
        s2 = self.layer2(s1)        # (B,128, H/8, W/8)   skip
        s3 = self.layer3(s2)        # (B,256,H/16,W/16)   skip
        s4 = self.layer4(s3)        # (B,512,H/32,W/32)   bottleneck

        # Decoder
        d4 = self.dec4(s4, s3)      # (B,256,H/16,W/16)
        d3 = self.dec3(d4, s2)      # (B,128, H/8, W/8)
        d2 = self.dec2(d3, s1)      # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, s0)      # (B, 64, H/2, W/2)

        # Head + upsample to input resolution
        out = self.head(d1)         # (B, K,  H/2, W/2)
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet2D(num_classes=21)
    x = torch.randn(2, 3, 480, 480)
    logits = model(x)
    print(f"GeoConvNet-2D segmentation output: {logits.shape}")  # (2, 21, 480, 480)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mask = logits.argmax(dim=1)
    print(f"Predicted mask shape: {mask.shape}")                  # (2, 480, 480)
