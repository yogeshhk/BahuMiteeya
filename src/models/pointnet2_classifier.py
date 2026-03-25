"""
GeoConvNet-3D-PC: PointNet++ MSG encoder for 3D point cloud classification
and part segmentation.

Theoretical basis: Permutation-invariant set abstraction is the G-equivariant
operator for point clouds under the symmetry group S_n x R^3.
See paper Section 3.

Datasets:
  - Classification: ModelNet40  (torch_geometric.datasets.ModelNet)
  - Segmentation:   ShapeNetPart (torch_geometric.datasets.ShapeNet)

Requirements:
  pip install torch-geometric torch-scatter torch-sparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, knn_interpolate
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARNING] torch_geometric not found. Install with: pip install torch-geometric")


# ---------------------------------------------------------------------------
# Shared MLP helper
# ---------------------------------------------------------------------------

def build_mlp(channels: list[int], last_act: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(channels) - 1):
        layers += [nn.Linear(channels[i], channels[i + 1], bias=False),
                   nn.BatchNorm1d(channels[i + 1])]
        if i < len(channels) - 2 or last_act:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Set Abstraction Module (PointNet++ style)
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    """
    Single-scale set abstraction layer.

    Samples npoint centroids, aggregates neighbors within radius r,
    and extracts local features with a shared MLP.
    """

    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp_channels: list[int]):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")
        self.npoint = npoint
        self.radius_val = radius
        self.nsample = nsample
        self.conv = PointNetConv(
            local_nn=build_mlp([in_channels + 3] + mlp_channels),
            global_nn=None,
        )

    def forward(self, x, pos, batch):
        # Farthest point sampling
        idx = fps(pos, batch, ratio=self.npoint / pos.size(0))
        row, col = radius(pos, pos[idx], self.radius_val,
                          batch, batch[idx], max_num_neighbors=self.nsample)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x_out = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos_out, batch_out = pos[idx], batch[idx]
        return x_out, pos_out, batch_out


# ---------------------------------------------------------------------------
# GeoConvNet-3D-PC: Classification
# ---------------------------------------------------------------------------

class GeoConvNet3DPC(nn.Module):
    """
    PointNet++ MSG-inspired point cloud classifier.

    Three set abstraction levels followed by global max pooling
    and a fully-connected classification head.

    Args:
        num_classes: Number of output classes (40 for ModelNet40).
        in_channels: Per-point input feature channels (0 for xyz only).
    """

    def __init__(self, num_classes: int = 40, in_channels: int = 0):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")

        c0 = in_channels  # 0 means xyz only; SA modules add 3 from pos concat

        self.sa1 = SetAbstraction(512, 0.2, 32, c0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(1,   1e6, 128, 256, [256, 512, 1024])

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric Data with .pos (N,3) and .batch (N,).
        Returns:
            logits: (B, num_classes)
        """
        x, pos, batch = data.x, data.pos, data.batch

        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)

        # After sa3: one global feature per shape → global max pool is trivial
        x = global_max_pool(x, batch)
        return self.head(x)


# ---------------------------------------------------------------------------
# Smoke test (requires torch_geometric)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print("Install torch-geometric to run smoke test.")
    else:
        from torch_geometric.data import Batch

        B, N = 4, 1024
        pos = torch.randn(B * N, 3)
        batch = torch.repeat_interleave(torch.arange(B), N)
        data = Data(pos=pos, batch=batch, x=None)

        model = GeoConvNet3DPC(num_classes=40)
        out = model(data)
        print(f"GeoConvNet-3D-PC output shape: {out.shape}")  # (4, 40)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
