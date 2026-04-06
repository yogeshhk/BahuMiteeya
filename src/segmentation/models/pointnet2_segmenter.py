"""
GeoConvNet-3D-PC: PointNet++ Encoder + Feature-Propagation Decoder
                  for Per-Point Part Segmentation.

Task: per-point dense labeling — assign one of K part labels to every
point p_i in the point cloud (e.g., chair: seat, back, leg, arm).

Theoretical basis:
  Permutation-invariant set abstraction (G = S_n) is the natural
  G-equivariant operator for unordered point sets. The feature
  propagation (FP) decoder restores per-point resolution via
  inverse-distance weighted interpolation from coarser abstractions —
  the 3D analog of bilinear upsampling in U-Net.

Dataset:
  ShapeNetPart — 16,881 shapes, 16 categories, 50 part labels.
  Evaluation: instance-averaged mean IoU over part classes.

Usage:
  model = GeoConvNet3DPCSeg(num_classes=50)
  # data: torch_geometric Data with .pos (N,3), .batch (N,), .x (N,C) or None
  logits = model(data)  # (N_total, num_classes) — per-point logits

Requirements:
  pip install torch-geometric torch-scatter torch-sparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import PointNetConv, fps, radius, knn_interpolate
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARNING] torch_geometric not found. Install: pip install torch-geometric")


# ---------------------------------------------------------------------------
# Shared MLP builder
# ---------------------------------------------------------------------------

def mlp(channels: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(channels) - 1):
        layers += [nn.Linear(channels[i], channels[i+1], bias=False),
                   nn.BatchNorm1d(channels[i+1]),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Set Abstraction (encoder)
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    """
    PointNet++ set abstraction layer.
    Samples npoint centroids, aggregates ball-query neighbors,
    and extracts per-centroid features with a shared MLP.
    """

    def __init__(self, ratio: float, r: float, nn_channels: list[int]):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(local_nn=mlp(nn_channels), global_nn=None)

    def forward(self, x, pos, batch):
        idx   = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_out = self.conv((x, None if x is None else x[idx]),
                          (pos, pos[idx]), edge_index)
        return x_out, pos[idx], batch[idx]


# ---------------------------------------------------------------------------
# Feature Propagation (decoder)
# ---------------------------------------------------------------------------

class FeaturePropagation(nn.Module):
    """
    PointNet++ feature propagation (FP) layer.
    Interpolates features from coarser set to finer set via k-NN
    inverse-distance weighting, then applies a shared MLP.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = mlp([in_channels, out_channels, out_channels])

    def forward(self, x_coarse, pos_coarse, batch_coarse,
                x_fine,   pos_fine,   batch_fine):
        """
        Interpolate x_coarse → pos_fine resolution, concat with x_fine skip.
        """
        # knn_interpolate: interpolates features from coarse to fine positions
        x_interp = knn_interpolate(x_coarse, pos_coarse, pos_fine,
                                   batch_coarse, batch_fine, k=self.k)
        if x_fine is not None:
            x_out = torch.cat([x_interp, x_fine], dim=1)
        else:
            x_out = x_interp
        return self.mlp(x_out)


# ---------------------------------------------------------------------------
# GeoConvNet-3D-PC: Part Segmentation
# ---------------------------------------------------------------------------

class GeoConvNet3DPCSeg(nn.Module):
    """
    PointNet++ MSG encoder + 3-level FP decoder for per-point part segmentation.

    Encoder (3 SA levels):
      SA1: ratio=0.5,  r=0.2  →  N/2  points,  128-dim features
      SA2: ratio=0.25, r=0.4  →  N/8  points,  256-dim features
      SA3: ratio=0.125,r=0.8  →  N/64 points,  512-dim features

    Decoder (3 FP levels, mirroring encoder):
      FP3: 512+256 → 256-dim  (N/8  points)
      FP2: 256+128 → 128-dim  (N/2  points)
      FP1: 128+0   → 128-dim  (N    points)

    Head: per-point MLP → K-class logits.

    Args:
        num_classes:  Number of part labels (50 for ShapeNetPart).
        in_channels:  Per-point input feature channels (0 = xyz only).
    """

    def __init__(self, num_classes: int = 50, in_channels: int = 0):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")

        c = in_channels  # 0 → PointNetConv adds 3 (pos offset) internally

        # Encoder — ratios are per-sample (PyG fps applies them independently per shape):
        # SA1: 0.50 × N   → N/2  points  (e.g. 1024 → 512)
        # SA2: 0.25 × N/2 → N/8  points  (e.g. 512  → 128)
        # SA3: 0.25 × N/8 → N/32 points  (e.g. 128  → 32)  — FP decoder restores full N
        self.sa1 = SetAbstraction(ratio=0.50, r=0.2, nn_channels=[c+3, 64,  64,  128])
        self.sa2 = SetAbstraction(ratio=0.25, r=0.4, nn_channels=[128+3, 128, 128, 256])
        self.sa3 = SetAbstraction(ratio=0.25, r=0.8, nn_channels=[256+3, 256, 256, 512])

        # Decoder (FP): in_channels = coarse_feat + skip_feat
        self.fp3 = FeaturePropagation(in_channels=512 + 256, out_channels=256)
        self.fp2 = FeaturePropagation(in_channels=256 + 128, out_channels=128)
        self.fp1 = FeaturePropagation(in_channels=128 + 0,   out_channels=128)

        # Per-point segmentation head
        self.head = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric Data with:
                  .pos   (N_total, 3)  — point coordinates
                  .batch (N_total,)    — batch index per point
                  .x     (N_total, C) or None
        Returns:
            logits: (N_total, num_classes) — per-point part logits.
        """
        x0, pos0, batch0 = data.x, data.pos, data.batch

        # Encode
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)

        # Decode — feature propagation back to original resolution
        x2_up = self.fp3(x3, pos3, batch3, x2, pos2, batch2)   # N/8 points
        x1_up = self.fp2(x2_up, pos2, batch2, x1, pos1, batch1) # N/2 points
        x0_up = self.fp1(x1_up, pos1, batch1, None, pos0, batch0) # N points

        return self.head(x0_up)  # (N_total, num_classes)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print("Install torch-geometric to run smoke test.")
    else:
        B, N = 2, 1024
        pos   = torch.randn(B * N, 3)
        batch = torch.repeat_interleave(torch.arange(B), N)
        data  = Data(pos=pos, batch=batch, x=None)

        model  = GeoConvNet3DPCSeg(num_classes=50)
        logits = model(data)
        print(f"GeoConvNet-3D-PC segmentation output: {logits.shape}")  # (2048, 50)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        mask = logits.argmax(dim=1)
        print(f"Predicted per-point part label shape: {mask.shape}")    # (2048,)
