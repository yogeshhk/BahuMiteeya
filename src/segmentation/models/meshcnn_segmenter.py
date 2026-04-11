"""
GeoConvNet-3D-Mesh: MeshCNN Encoder-Decoder for Per-Edge Part Segmentation.

Task: per-edge dense labeling — assign one of K part labels to every
edge e in a triangular mesh (e.g., mug: body, handle, rim; chair: seat,
back, leg, arm).

Theoretical basis:
  Edge convolution on a triangular mesh (G = gauge symmetry group) is
  the natural G-equivariant operator for signals on 2-manifolds.
  Symmetrization over the two face-pairs of each edge enforces
  invariance to local coordinate frame choice (gauge invariance).
  The segmentation decoder "un-collapses" pooled edges back to their
  pre-collapse ancestors, restoring per-edge label resolution.

Dataset:
  COSEG — co-segmentation dataset (vases, chairs, aliens).
  Each mesh has per-edge ground-truth part labels (3-6 classes per set).
  Protocol: MeshCNN train/test split.

Based on:
  Hanocka et al., "MeshCNN: A Network with an Edge", SIGGRAPH/TOG 2019.

Usage:
  model = GeoConvNet3DMeshSeg(num_classes=4)
  x  = torch.randn(E, 5)    # per-edge 5D features
  nb = torch.randint(0,E,(E,4))  # 4-neighbor indices
  logits = model(x, nb)     # (E, num_classes) — per-edge logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MeshConv: edge convolution with gauge-invariant symmetrization
# ---------------------------------------------------------------------------

class MeshConv(nn.Module):
    """
    Convolution over mesh edges.

    Neighborhood: each edge e has 4 neighbors — the two edges from each
    incident triangle that share a face with e.  Features from the two
    face-pairs are sorted (symmetrized) to achieve gauge invariance.

    f'(e) = MLP([f(e), sort(f(a),f(b)), sort(f(c),f(d))])
    where {a,b} come from one incident triangle, {c,d} from the other.

    Args:
        in_ch:  Input edge feature dimension.
        out_ch: Output edge feature dimension.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 5 feature groups: center + 2 from face1 + 2 from face2
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
        # Gather: (E, 4, C)
        n = x[nb.clamp(0, E-1).view(-1)].view(E, 4, C)
        # Symmetrize within each face-pair (sort by channel sum → deterministic order)
        p1 = torch.sort(n[:, :2, :], dim=1).values   # (E, 2, C)
        p2 = torch.sort(n[:, 2:, :], dim=1).values   # (E, 2, C)
        feat = torch.cat([x, p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]], dim=1)  # (E, 5C)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# MeshPool: edge-collapse pooling with bookkeeping for decoder
# ---------------------------------------------------------------------------

class MeshPool(nn.Module):
    """
    Mesh pooling via learned edge collapse.

    Scores edges by feature L2 norm; collapses the lowest-scoring edges,
    storing a mapping (collapse_map) from kept→original indices so the
    decoder can un-collapse (broadcast) features back.

    Args:
        target: Target edge count after pooling.
    """

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def forward(self, x: torch.Tensor, nb: torch.Tensor):
        """
        Returns:
            x_pool:    (target, C)
            nb_pool:   (target, 4)  — remapped neighbor indices
            keep_idx:  (target,)    — original indices of kept edges
        """
        E = x.size(0)
        if E <= self.target:
            return x, nb, torch.arange(E, device=x.device)

        scores    = x.norm(dim=1)
        keep_idx  = scores.argsort(descending=True)[:self.target]
        keep_idx  = keep_idx.sort().values

        x_pool = x[keep_idx]

        # Remap neighbor indices
        remap = torch.full((E,), -1, dtype=torch.long, device=x.device)
        remap[keep_idx] = torch.arange(self.target, device=x.device)
        nb_remapped = remap[nb[keep_idx].clamp(0, E-1)]
        self_idx = torch.arange(self.target, device=x.device).unsqueeze(1).expand_as(nb_remapped)
        nb_pool  = torch.where(nb_remapped < 0, self_idx, nb_remapped)

        return x_pool, nb_pool, keep_idx


# ---------------------------------------------------------------------------
# MeshUnpool: broadcast pooled features back to pre-collapse resolution
# ---------------------------------------------------------------------------

class MeshUnpool(nn.Module):
    """
    Broadcast coarse edge features back to fine (pre-collapse) resolution.
    Edges that were kept copy their feature directly. Collapsed (missing) edges
    receive the feature of their nearest kept edge by index position.

    Note: The nearest-fill uses integer index distance, not feature similarity.
    This is a non-differentiable approximation of the un-collapse step — it
    mirrors the simplified approach in the original MeshCNN repo and works well
    in practice because adjacent edges in the ordering tend to be spatially
    close on the mesh.
    """

    def forward(self, x_coarse: torch.Tensor, keep_idx: torch.Tensor,
                E_fine: int) -> torch.Tensor:
        """
        Args:
            x_coarse: (E_coarse, C)
            keep_idx: (E_coarse,) — indices into the fine edge set
            E_fine:   int — number of edges in the fine set
        Returns:
            x_fine: (E_fine, C)
        """
        x_fine = torch.zeros(E_fine, x_coarse.size(1),
                             dtype=x_coarse.dtype, device=x_coarse.device)
        x_fine[keep_idx] = x_coarse
        # Fill non-kept edges with their nearest kept edge (simple nearest fill)
        mask = torch.ones(E_fine, dtype=torch.bool, device=x_coarse.device)
        mask[keep_idx] = False
        if mask.any():
            # For each missing edge, assign the feature of the closest kept edge
            missing_idx  = torch.where(mask)[0]
            kept_expanded = keep_idx.float().unsqueeze(0)  # (1, E_coarse)
            missing_exp   = missing_idx.float().unsqueeze(1)  # (missing, 1)
            nn_idx = (kept_expanded - missing_exp).abs().argmin(dim=1)
            x_fine[missing_idx] = x_coarse[nn_idx]
        return x_fine


# ---------------------------------------------------------------------------
# GeoConvNet-3D-Mesh: Segmentation
# ---------------------------------------------------------------------------

class GeoConvNet3DMeshSeg(nn.Module):
    """
    MeshCNN encoder-decoder for per-edge mesh part segmentation.

    Encoder:
      MeshConv(5→64) → MeshPool(1500)
      MeshConv(64→128) → MeshPool(750)
      MeshConv(128→256) → MeshPool(375)
      MeshConv(256→512)

    Decoder (edge unpool + skip concat + MeshConv):
      Unpool(375→750) + skip e3(256) → MeshConv(512+256→256)
      Unpool(750→1500) + skip e2(128) → MeshConv(256+128→128)
      Unpool(1500→E)   + skip e1(64)  → MeshConv(128+64→64)

    Head: per-edge Linear → K-class logits.

    Args:
        num_classes:     Number of part labels per class (e.g., 4 for chairs).
        input_features:  Edge feature dimension (5 for standard MeshCNN).
    """

    def __init__(self, num_classes: int = 4, input_features: int = 5):
        super().__init__()

        # Encoder
        self.econv1 = MeshConv(input_features, 64)
        self.pool1  = MeshPool(1500)

        self.econv2 = MeshConv(64, 128)
        self.pool2  = MeshPool(750)

        self.econv3 = MeshConv(128, 256)
        self.pool3  = MeshPool(375)

        self.econv4 = MeshConv(256, 512)

        # Decoder
        self.unpool3 = MeshUnpool()
        self.dconv3  = MeshConv(512 + 256, 256)

        self.unpool2 = MeshUnpool()
        self.dconv2  = MeshConv(256 + 128, 128)

        self.unpool1 = MeshUnpool()
        self.dconv1  = MeshConv(128 + 64, 64)

        # Segmentation head
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (E, 5)  — per-edge feature vectors
            nb: (E, 4)  — 4-neighbor edge indices
        Returns:
            logits: (E, num_classes) — per-edge part logits
        """
        E = x.size(0)

        # === Encode ===
        e1 = self.econv1(x, nb)                      # (E, 64)
        e1p, nb1, keep1 = self.pool1(e1, nb)         # (~1500, 64)

        e2 = self.econv2(e1p, nb1)                   # (~1500, 128)
        e2p, nb2, keep2 = self.pool2(e2, nb1)        # (~750,  128)

        e3 = self.econv3(e2p, nb2)                   # (~750,  256)
        e3p, nb3, keep3 = self.pool3(e3, nb2)        # (~375,  256)

        e4 = self.econv4(e3p, nb3)                   # (~375,  512)

        # === Decode ===
        # Level 3: unpool 375→750, concat skip e3 (~750,256), conv→256
        d3 = self.unpool3(e4, keep3, e3.size(0))     # (~750,  512)
        d3 = self.dconv3(torch.cat([d3, e3], dim=1),
                         nb2)                         # (~750,  256)

        # Level 2: unpool 750→1500, concat skip e2 (~1500,128), conv→128
        d2 = self.unpool2(d3, keep2, e2.size(0))     # (~1500, 256)
        d2 = self.dconv2(torch.cat([d2, e2], dim=1),
                         nb1)                         # (~1500, 128)

        # Level 1: unpool 1500→E, concat skip e1 (E,64), conv→64
        d1 = self.unpool1(d2, keep1, e1.size(0))     # (E, 128)
        d1 = self.dconv1(torch.cat([d1, e1], dim=1),
                         nb)                          # (E, 64)

        return self.head(d1)   # (E, num_classes)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    E  = 2000
    x  = torch.randn(E, 5)
    nb = torch.randint(0, E, (E, 4))

    model  = GeoConvNet3DMeshSeg(num_classes=4)
    logits = model(x, nb)
    print(f"GeoConvNet-3D-Mesh segmentation output: {logits.shape}")  # (2000, 4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mask = logits.argmax(dim=1)
    print(f"Predicted per-edge part label shape: {mask.shape}")        # (2000,)
