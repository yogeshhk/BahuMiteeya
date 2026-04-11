"""
GeoConvNet-3D-Mesh: MeshCNN Summariser for 3D Polygonal Meshes.

Task: produce a coarser representative mesh (summary) from a detailed input
mesh, preserving geometric structure (shape semantics) while reducing edge
count by 8x via learned edge collapse.

Theoretical basis:
  Edge convolution on a triangular mesh (G = gauge symmetry group) is the
  natural G-equivariant operator for signals on 2-manifolds.  Symmetrization
  over the two face-pairs of each edge enforces gauge invariance.  Learned
  edge collapse (MeshPool) progressively coarsens the mesh — exactly the
  domain-appropriate summarisation operator.  The summary is a coarser mesh
  with E/8 edges, represented by 5-dimensional intrinsic edge features
  (rotation/translation/scale invariant).  The decoder un-collapses the
  summary back to the original E-edge resolution for reconstruction.

Dataset: SHREC16 (30 classes)
Default: 8x compression — E=~2000 edges → E'=~250 edges

Training objective:
  L = alpha * CrossEntropy(logits, y) + (1-alpha) * MSE(recon_feats, x)

Based on:
  Hanocka et al., "MeshCNN: A Network with an Edge", SIGGRAPH/TOG 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MeshConv: gauge-invariant edge convolution
# ---------------------------------------------------------------------------

class MeshConv(nn.Module):
    """
    Convolution over mesh edges with gauge-invariant symmetrization.

    Each edge has exactly 4 neighbors (edges of its two incident triangles).
    Features from the two face-pairs are sorted to achieve gauge invariance:
        f'(e) = MLP([f(e), sort(f(a),f(b)), sort(f(c),f(d))])

    Args:
        in_ch:  Input edge feature dimension.
        out_ch: Output edge feature dimension.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 5, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (E, in_ch)
            nb: (E, 4)    — indices of 4 neighbor edges
        Returns:
            (E, out_ch)
        """
        E, C = x.shape
        n    = x[nb.clamp(0, E - 1).view(-1)].view(E, 4, C)
        p1   = torch.sort(n[:, :2, :], dim=1).values   # (E, 2, C) face-pair 1
        p2   = torch.sort(n[:, 2:, :], dim=1).values   # (E, 2, C) face-pair 2
        feat = torch.cat([x, p1[:, 0], p1[:, 1],
                             p2[:, 0], p2[:, 1]], dim=1)   # (E, 5C)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# MeshPool: learnable edge collapse with bookkeeping
# ---------------------------------------------------------------------------

class MeshPool(nn.Module):
    """
    Mesh pooling via score-based edge collapse.

    Scores edges by feature L2 norm; retains the top-scoring edges.
    Stores keep_idx so the decoder (MeshUnpool) can broadcast features
    back to the pre-collapse resolution.

    Args:
        target: Target edge count after pooling.
    """

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def forward(self, x: torch.Tensor, nb: torch.Tensor):
        """
        Returns:
            x_pool:   (target, C)
            nb_pool:  (target, 4) — remapped neighbor indices
            keep_idx: (target,)   — original indices of retained edges
        """
        E = x.size(0)
        if E <= self.target:
            return x, nb, torch.arange(E, device=x.device)

        keep_idx = x.norm(dim=1).argsort(descending=True)[:self.target]
        keep_idx = keep_idx.sort().values

        x_pool   = x[keep_idx]

        remap = torch.full((E,), -1, dtype=torch.long, device=x.device)
        remap[keep_idx] = torch.arange(self.target, device=x.device)
        nb_remapped = remap[nb[keep_idx].clamp(0, E - 1)]
        self_idx    = torch.arange(self.target, device=x.device
                                   ).unsqueeze(1).expand_as(nb_remapped)
        nb_pool     = torch.where(nb_remapped < 0, self_idx, nb_remapped)

        return x_pool, nb_pool, keep_idx


# ---------------------------------------------------------------------------
# MeshUnpool: broadcast coarse features back to fine resolution
# ---------------------------------------------------------------------------

class MeshUnpool(nn.Module):
    """
    Broadcast coarse edge features back to the pre-collapse (fine) resolution.

    Kept edges copy their feature directly.  Collapsed (missing) edges
    receive the feature of their nearest kept edge by index distance.
    """

    def forward(self, x_coarse: torch.Tensor, keep_idx: torch.Tensor,
                E_fine: int) -> torch.Tensor:
        """
        Args:
            x_coarse: (E_coarse, C)
            keep_idx: (E_coarse,) — indices in the fine edge set
            E_fine:   int — fine edge count
        Returns:
            x_fine: (E_fine, C)
        """
        x_fine = torch.zeros(E_fine, x_coarse.size(1),
                             dtype=x_coarse.dtype, device=x_coarse.device)
        x_fine[keep_idx] = x_coarse
        mask = torch.ones(E_fine, dtype=torch.bool, device=x_coarse.device)
        mask[keep_idx] = False
        if mask.any():
            missing_idx = torch.where(mask)[0]
            kept_exp    = keep_idx.float().unsqueeze(0)      # (1, E_coarse)
            miss_exp    = missing_idx.float().unsqueeze(1)   # (missing, 1)
            nn_idx      = (kept_exp - miss_exp).abs().argmin(dim=1)
            x_fine[missing_idx] = x_coarse[nn_idx]
        return x_fine


# ---------------------------------------------------------------------------
# GeoConvNet-3D-Mesh: Summariser
# ---------------------------------------------------------------------------

class GeoConvNet3DMeshSummariser(nn.Module):
    """
    Gauge-equivariant mesh summariser.

    Coarsens a mesh from E edges to E/8 via two MeshPool stages, producing
    a summary mesh with 5-dimensional intrinsic edge features.  Decodes
    back to E-edge resolution for reconstruction via MeshUnpool + MeshConv.

    Encoder:
        MeshConv(5→64)   →  MeshPool(E/2)   →  (E/2, 64)
        MeshConv(64→128) →  MeshPool(E/8)   →  (E/8, 128)  [SUMMARY LEVEL]

    Summary:
        summary_head: (E/8, 128) → (E/8, 5)   [5-D intrinsic edge features]

    Decoder:
        MeshUnpool(E/8→E/2) + skip(E/2, 64) → MeshConv(192→64)
        MeshUnpool(E/2→E)   + skip(E, 5)    → MeshConv(69→32)
        recon_head: Linear(32→5)             [reconstructed edge features]

    Classification head:
        GlobalMaxPool(summary_features) → FC(128→K)

    Args:
        num_classes:     Number of downstream classification classes.
        input_features:  Edge feature dimension (5 for standard MeshCNN).
    """

    def __init__(self, num_classes: int = 30, input_features: int = 5):
        super().__init__()

        # Encoder
        self.econv1 = MeshConv(input_features, 64)
        self.pool1  = MeshPool(target=0)  # target set dynamically (E/2)

        self.econv2 = MeshConv(64, 128)
        self.pool2  = MeshPool(target=0)  # target = E/8  [summary level]

        # Summary head: feature → intrinsic edge-feature space
        self.summary_head = nn.Linear(128, input_features)

        # Decoder
        self.unpool2 = MeshUnpool()
        # After unpool2 (E/8→E/2): concat with econv1 skip (E/2, 64) → MeshConv(128+64=192→64)
        self.dconv2  = MeshConv(128 + 64, 64)

        self.unpool1 = MeshUnpool()
        # After unpool1 (E/2→E): concat with raw input (E, 5) → MeshConv(64+5=69→32)
        self.dconv1  = MeshConv(64 + input_features, 32)

        # Reconstruction head
        self.recon_head = nn.Linear(32, input_features)

        # Downstream classification head
        # Note: no BatchNorm here — mesh processing is always per-sample
        # (batch size = 1), so BN would fail during training mode.
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

        self._input_features = input_features

    def forward(self, x: torch.Tensor, nb: torch.Tensor):
        """
        Args:
            x:  (E, 5)  — per-edge 5D intrinsic feature vectors
            nb: (E, 4)  — 4-neighbor edge indices
        Returns:
            summary_feats: (E//8, 5)       — coarse-mesh edge features (summary).
            recon_feats:   (E, 5)          — reconstructed fine-mesh edge features.
            logits:        (1, num_classes) — downstream classification logits.
        """
        E = x.size(0)
        t1 = max(E // 2, 1)
        t2 = max(E // 8, 1)

        # Encode
        e1 = self.econv1(x, nb)                             # (E,   64)
        e1p, nb1, keep1 = self._pool(e1, nb, t1)            # (E/2, 64)

        e2 = self.econv2(e1p, nb1)                          # (E/2, 128)
        e2p, nb2, keep2 = self._pool(e2, nb1, t2)           # (E/8, 128) ← SUMMARY

        # Summary: project to intrinsic edge-feature space
        summary_feats = self.summary_head(e2p)              # (E/8, 5)

        # Decode
        d2 = self.unpool2(e2p, keep2, e2.size(0))           # (E/2, 128)
        d2 = self.dconv2(torch.cat([d2, e1p], dim=1), nb1)  # (E/2, 64)

        d1 = self.unpool1(d2, keep1, e1.size(0))            # (E,   64)
        d1 = self.dconv1(torch.cat([d1, x], dim=1), nb)     # (E,   32)

        recon_feats = self.recon_head(d1)                   # (E, 5)

        # Classification: global max pool over summary features
        g = e2p.max(dim=0, keepdim=True).values             # (1, 128)
        logits = self.cls_head(g)                           # (1, K)

        return summary_feats, recon_feats, logits

    @staticmethod
    def _pool(x: torch.Tensor, nb: torch.Tensor, target: int):
        """Convenience wrapper for dynamic target MeshPool."""
        E = x.size(0)
        if E <= target:
            return x, nb, torch.arange(E, device=x.device)
        keep_idx = x.norm(dim=1).argsort(descending=True)[:target].sort().values
        x_pool   = x[keep_idx]
        remap = torch.full((E,), -1, dtype=torch.long, device=x.device)
        remap[keep_idx] = torch.arange(target, device=x.device)
        nb_r = remap[nb[keep_idx].clamp(0, E - 1)]
        si   = torch.arange(target, device=x.device).unsqueeze(1).expand_as(nb_r)
        return x_pool, torch.where(nb_r < 0, si, nb_r), keep_idx

    def summarise(self, x: torch.Tensor, nb: torch.Tensor):
        """Return only the coarse-mesh summary (feats + neighbor indices)."""
        with torch.no_grad():
            summary_feats, _, _ = self.forward(x, nb)
        return summary_feats


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    E  = 2000
    x  = torch.randn(E, 5)
    nb = torch.randint(0, E, (E, 4))

    model = GeoConvNet3DMeshSummariser(num_classes=30)
    summary_feats, recon_feats, logits = model(x, nb)

    print(f"Input edges:    {E}")                              # 2000
    print(f"Summary edges:  {summary_feats.size(0)}")         # ~250
    print(f"Recon feats:    {recon_feats.shape}")             # (2000, 5)
    print(f"Logits shape:   {logits.shape}")                  # (1, 30)
    print(f"Compression:    ~{E // summary_feats.size(0)}x")
    print(f"Parameters:     {sum(p.numel() for p in model.parameters()):,}")
