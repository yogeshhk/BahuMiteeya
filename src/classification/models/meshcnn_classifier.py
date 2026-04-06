"""
GeoConvNet-3D-Mesh: Edge-based convolutional network for 3D mesh classification.

Theoretical basis: Edge convolution on a triangular mesh is a gauge-equivariant
operator—invariant to local coordinate frame choice—making it the natural
G-equivariant operator for manifold signals. See paper Section 3.

Based on: Hanocka et al., "MeshCNN: A Network with an Edge", SIGGRAPH/TOG 2019.
Dataset: SHREC16 (downloaded via the official MeshCNN repo).

This is a self-contained PyTorch re-implementation of the core MeshCNN
convolution and pooling operations. For full training use:
  python train.py --domain mesh

Input features per edge (5-dimensional, rotation/translation/scale invariant):
  [dihedral_angle, inner_angle_1, inner_angle_2, ratio_1, ratio_2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MeshConv: convolution on mesh edges
# ---------------------------------------------------------------------------

class MeshConv(nn.Module):
    """
    Convolution over mesh edges.

    Each edge e has exactly 4 neighbors (the 4 edges of its two incident
    triangles). The operation is:
        f'(e) = MLP([f(e), sort_sym(f(neighbors))])
    where sort_sym applies symmetric (order-invariant) aggregation over
    the two face-pairs.

    Args:
        in_channels:  Input edge feature dimension.
        out_channels: Output edge feature dimension.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 5 inputs: center edge + 4 neighbors → but symmetrized to 2 groups
        # Following MeshCNN: input is (in_channels * 5)
        self.conv = nn.Sequential(
            nn.Linear(in_channels * 5, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            (E, C)    — edge features
            neighbor_idx: (E, 4)   — indices of the 4 neighbor edges for each edge
        Returns:
            x_out: (E, out_channels)
        """
        E, C = x.shape
        # Gather neighbor features: (E, 4, C)
        neighbors = x[neighbor_idx.view(-1)].view(E, 4, C)

        # Symmetric pooling over the two face-pairs:
        # faces (a,b) and (c,d) — sort within each pair for order invariance
        pair1 = torch.sort(neighbors[:, :2, :], dim=1).values  # (E, 2, C)
        pair2 = torch.sort(neighbors[:, 2:, :], dim=1).values  # (E, 2, C)

        # Concatenate: center + pair1_0 + pair1_1 + pair2_0 + pair2_1
        feats = torch.cat([x, pair1[:, 0], pair1[:, 1],
                              pair2[:, 0], pair2[:, 1]], dim=1)  # (E, 5*C)
        return self.conv(feats)


# ---------------------------------------------------------------------------
# MeshPool: learnable edge collapse
# ---------------------------------------------------------------------------

class MeshPool(nn.Module):
    """
    Mesh pooling via edge collapse.

    Scores edges by their feature L2 norm and collapses the lowest-scoring edges.

    This is a simplified version of MeshCNN pooling: it uses L2-norm of the
    learned features as a fixed importance proxy (lower norm = less informative
    = collapse first), rather than the full learnable edge-importance scoring in
    the original paper (Hanocka et al., SIGGRAPH 2019). The topology update is
    also simplified: collapsed edges are dropped and neighbor indices are
    remapped rather than properly merging face-pairs. This is sufficient for
    classification but differs from the full MeshCNN implementation.

    Args:
        target_edges: Number of edges to retain after pooling.
    """

    def __init__(self, target_edges: int):
        super().__init__()
        self.target_edges = target_edges

    def forward(self, x: torch.Tensor, neighbor_idx: torch.Tensor):
        """
        Args:
            x:            (E, C)
            neighbor_idx: (E, 4)
        Returns:
            x_pooled:     (target_edges, C)  — pooled edge features
            neighbor_idx_pooled: (target_edges, 4) — updated neighbor indices
        """
        E = x.size(0)
        if E <= self.target_edges:
            return x, neighbor_idx

        # Score = L2 norm of features (lower = less informative = collapse first)
        scores = x.norm(dim=1)
        keep_idx = scores.argsort(descending=True)[:self.target_edges]
        keep_idx_sorted = keep_idx.sort().values

        # Simple subset: keep high-scoring edges, remap neighbor indices
        x_pooled = x[keep_idx_sorted]

        # Remap neighbor indices (edges not kept → clamped to valid range)
        remap = torch.full((E,), -1, dtype=torch.long, device=x.device)
        remap[keep_idx_sorted] = torch.arange(self.target_edges, device=x.device)
        neighbors_remapped = remap[neighbor_idx[keep_idx_sorted].clamp(0, E - 1)]
        # Fill invalid (-1) neighbors with self-loop
        self_idx = torch.arange(self.target_edges, device=x.device).unsqueeze(1).expand_as(neighbors_remapped)
        neighbors_remapped = torch.where(neighbors_remapped < 0, self_idx, neighbors_remapped)

        return x_pooled, neighbors_remapped


# ---------------------------------------------------------------------------
# GeoConvNet-3D-Mesh: Classification
# ---------------------------------------------------------------------------

class GeoConvNet3DMesh(nn.Module):
    """
    Edge-based mesh classification network.

    Architecture:
        MeshConv(5 → 64) → MeshPool(1500) →
        MeshConv(64 → 128) → MeshPool(750) →
        MeshConv(128 → 256) → MeshPool(375) →
        MeshConv(256 → 512) → GlobalMaxPool → FC(num_classes)

    Args:
        num_classes: Number of output classes (30 for SHREC16).
        input_features: Number of input edge features (5 for standard MeshCNN).
    """

    def __init__(self, num_classes: int = 30, input_features: int = 5):
        super().__init__()
        self.conv1 = MeshConv(input_features, 64)
        self.pool1 = MeshPool(1500)
        self.conv2 = MeshConv(64, 128)
        self.pool2 = MeshPool(750)
        self.conv3 = MeshConv(128, 256)
        self.pool3 = MeshPool(375)
        self.conv4 = MeshConv(256, 512)

        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            (E, 5)  — per-edge feature vectors
            neighbor_idx: (E, 4)  — 4-neighbor indices per edge
        Returns:
            logits: (1, num_classes)  — single mesh classification output
        """
        x = self.conv1(x, neighbor_idx)
        x, neighbor_idx = self.pool1(x, neighbor_idx)

        x = self.conv2(x, neighbor_idx)
        x, neighbor_idx = self.pool2(x, neighbor_idx)

        x = self.conv3(x, neighbor_idx)
        x, neighbor_idx = self.pool3(x, neighbor_idx)

        x = self.conv4(x, neighbor_idx)

        # Global max pool over all remaining edges
        g = x.max(dim=0, keepdim=True).values  # (1, 512)
        return self.head(g)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    E = 2000  # number of edges in a sample mesh
    x = torch.randn(E, 5)
    # Random neighbor indices (in practice loaded from mesh data)
    neighbor_idx = torch.randint(0, E, (E, 4))

    model = GeoConvNet3DMesh(num_classes=30)
    out = model(x, neighbor_idx)
    print(f"GeoConvNet-3D-Mesh output shape: {out.shape}")  # (1, 30)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
