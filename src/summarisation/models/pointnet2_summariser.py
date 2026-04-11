"""
GeoConvNet-3D-PC: PointNet++ Summariser for 3D Point Clouds.

Task: produce a sparse representative point cloud (summary) from a dense
input point cloud, preserving the geometric structure (shape semantics)
while reducing point count by 8x.

Theoretical basis:
  Farthest point sampling (FPS) under permutation symmetry (G = S_n) is
  the natural G-equivariant coarsening operator for point clouds.  The
  summariser uses two set-abstraction levels to select M = N/8 maximally
  spread representative points (the summary), then uses feature-propagation
  layers to decode back to the original N-point resolution for reconstruction.
  The summary is evaluated by:
    (a) downstream classification accuracy on the M-point cloud, and
    (b) Chamfer distance between reconstructed and original point positions.

Dataset: ModelNet40 (1024 points, 40 classes)
Default: 8x compression → 1024 → 128 representative points

Training objective:
  L = alpha * CrossEntropy(logits, y) + (1-alpha) * ChamferDist(recon, pos)

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
    print("[WARNING] torch_geometric not found. "
          "Install with: pip install torch-geometric")


# ---------------------------------------------------------------------------
# Chamfer distance (symmetric, per-batch)
# ---------------------------------------------------------------------------

def chamfer_distance_batch(recon_pos: torch.Tensor, orig_pos: torch.Tensor,
                           batch_recon: torch.Tensor,
                           batch_orig: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer distance averaged over the batch.

    Args:
        recon_pos:   (N_total, 3) reconstructed point positions.
        orig_pos:    (N_total, 3) original point positions.
        batch_recon: (N_total,)   batch index for each reconstructed point.
        batch_orig:  (N_total,)   batch index for each original point.
    Returns:
        Scalar mean Chamfer distance across all samples in the batch.
    """
    B = batch_orig.max().item() + 1
    total = 0.0
    for b in range(B):
        r = recon_pos[batch_recon == b]   # (n, 3)
        o = orig_pos[batch_orig   == b]   # (m, 3)
        if r.size(0) == 0 or o.size(0) == 0:
            continue
        # Pairwise squared distances: (n, m)
        rr = (r ** 2).sum(1, keepdim=True)       # (n, 1)
        oo = (o ** 2).sum(1, keepdim=True)       # (m, 1)
        dist2 = rr + oo.T - 2.0 * (r @ o.T)     # (n, m)
        d1 = dist2.min(dim=1).values.mean()      # min over o for each r
        d2 = dist2.min(dim=0).values.mean()      # min over r for each o
        total += (d1 + d2)
    return torch.tensor(total / B, device=recon_pos.device,
                        requires_grad=True)


# ---------------------------------------------------------------------------
# Shared MLP helper
# ---------------------------------------------------------------------------

def mlp(channels: list) -> nn.Sequential:
    layers = []
    for i in range(len(channels) - 1):
        layers += [nn.Linear(channels[i], channels[i + 1], bias=False),
                   nn.BatchNorm1d(channels[i + 1]),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Set Abstraction (encoder)
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    """
    PointNet++ set abstraction layer.

    Selects centroids via FPS, aggregates ball-query neighbors,
    and extracts per-centroid features with a shared MLP.

    Args:
        ratio:       FPS sampling ratio per sample (batch-size-independent).
        r:           Ball-query radius.
        nn_channels: MLP channel list (first = in_channels + 3).
    """

    def __init__(self, ratio: float, r: float, nn_channels: list):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")
        self.ratio = ratio
        self.r     = r
        self.conv  = PointNetConv(local_nn=mlp(nn_channels), global_nn=None)

    def forward(self, x, pos, batch):
        idx        = fps(pos, batch, ratio=self.ratio)
        row, col   = radius(pos, pos[idx], self.r, batch, batch[idx],
                            max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst      = None if x is None else x[idx]
        x_out      = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x_out, pos[idx], batch[idx]


# ---------------------------------------------------------------------------
# Feature Propagation (decoder)
# ---------------------------------------------------------------------------

class FeaturePropagation(nn.Module):
    """
    PointNet++ feature propagation layer.

    Interpolates features from a coarser set to a finer set via k-NN
    inverse-distance weighting, then applies a shared MLP.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.k   = k
        self.mlp = mlp([in_channels, out_channels, out_channels])

    def forward(self, x_coarse, pos_coarse, batch_coarse,
                x_fine, pos_fine, batch_fine):
        x_interp = knn_interpolate(x_coarse, pos_coarse, pos_fine,
                                   batch_coarse, batch_fine, k=self.k)
        if x_fine is not None:
            x_out = torch.cat([x_interp, x_fine], dim=1)
        else:
            x_out = x_interp
        return self.mlp(x_out)


# ---------------------------------------------------------------------------
# GeoConvNet-3D-PC: Summariser
# ---------------------------------------------------------------------------

class GeoConvNet3DPCSummariser(nn.Module):
    """
    Permutation-equivariant 3D point cloud summariser.

    Compresses N points to M = N/8 representative points (the summary)
    via two set-abstraction levels, then decodes back to N points for
    reconstruction.

    Encoder (2 SA levels):
        SA1: ratio=0.50  →  N/2 points,  128-dim features
        SA2: ratio=0.25  →  M=N/8 points, 256-dim features  [SUMMARY LEVEL]

    Summary:
        The M FPS-selected 3D positions (pos2) constitute the summary point cloud.

    Decoder (2 FP levels):
        FP2: M → N/2,   in=256+128, out=128
        FP1: N/2 → N,   in=128+0,   out=64
        recon_head: Linear(64→3) → reconstructed xyz

    Classification head:
        GlobalMaxPool(features_M) → FC(256→K)

    Args:
        num_classes:  Number of downstream classification classes.
        in_channels:  Per-point input feature channels (0 = xyz only).
    """

    def __init__(self, num_classes: int = 40, in_channels: int = 0):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")

        c = in_channels   # PointNetConv adds 3 (xyz offset) internally

        # Encoder
        self.sa1 = SetAbstraction(ratio=0.50, r=0.2,
                                  nn_channels=[c+3, 64, 64, 128])
        self.sa2 = SetAbstraction(ratio=0.25, r=0.4,
                                  nn_channels=[128+3, 128, 128, 256])

        # Reconstruction decoder
        self.fp2 = FeaturePropagation(in_channels=256 + 128, out_channels=128)
        self.fp1 = FeaturePropagation(in_channels=128,       out_channels=64)

        # Reconstruction head: dense features → xyz coordinates
        self.recon_head = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Downstream classification head (on summary features)
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        """
        Args:
            data: torch_geometric Data with:
                  .pos   (N_total, 3)  — original point positions
                  .batch (N_total,)    — batch index per point
                  .x     (N_total, C) or None
        Returns:
            summary_pos: (M_total, 3)   — M representative point positions.
            recon_pos:   (N_total, 3)   — reconstructed N point positions.
            logits:      (B, num_classes) — downstream classification logits.
        """
        x0, pos0, batch0 = data.x, data.pos, data.batch

        # Encode
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)   # N/2 points
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)   # M=N/8 points [summary]

        # Summary: the M representative point positions
        summary_pos = pos2    # (M_total, 3)

        # Decode: feature propagation back to N-point resolution
        x1_up = self.fp2(x2, pos2, batch2, x1, pos1, batch1)  # N/2 points
        x0_up = self.fp1(x1_up, pos1, batch1, None, pos0, batch0)  # N points

        # Reconstruct xyz
        recon_pos = self.recon_head(x0_up)   # (N_total, 3)

        # Classify from summary features (global max pool over M points)
        from torch_geometric.nn import global_max_pool
        g = global_max_pool(x2, batch2)      # (B, 256)
        logits = self.cls_head(g)            # (B, K)

        return summary_pos, recon_pos, logits

    def summarise(self, data):
        """Return only the summary point positions and batch index."""
        with torch.no_grad():
            x0, pos0, batch0 = data.x, data.pos, data.batch
            x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
            x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        return pos2, batch2


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print("Install torch-geometric to run smoke test.")
    else:
        from torch_geometric.data import Batch

        B, N = 4, 1024
        pos   = torch.randn(B * N, 3)
        batch = torch.repeat_interleave(torch.arange(B), N)
        data  = Data(pos=pos, batch=batch, x=None)

        model = GeoConvNet3DPCSummariser(num_classes=40)
        summary_pos, recon_pos, logits = model(data)

        M = summary_pos.size(0)
        print(f"Input points:   {B * N}")            # 4096
        print(f"Summary points: {M}  ({M//(B)}  per sample)")  # 512 total → 128/sample
        print(f"Recon pos:      {recon_pos.shape}")   # (4096, 3)
        print(f"Logits shape:   {logits.shape}")      # (4, 40)
        print(f"Compression:    {B*N // M}x")
        print(f"Parameters:     {sum(p.numel() for p in model.parameters()):,}")
