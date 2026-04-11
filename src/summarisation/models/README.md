# Summarisation Models

Each model is a domain-specific **encoder–summary–decoder** network producing three outputs:

| Output | Shape | Purpose |
|--------|-------|---------|
| `summary` | Smaller than input | The compressed representation (same format as input) |
| `recon` | Same shape as input | Reconstruction from summary features (for recon loss) |
| `logits` | `(B, K)` | Downstream classification (for task-preservation loss) |

---

## Model Overview

| File | Class | Domain | Input → Summary | Compression |
|------|-------|--------|-----------------|-------------|
| `conv1d_summariser.py` | `GeoConvNet1DSummariser` | 1D time series | `(B,1,T)` → `(B,1,T/4)` | 4× temporal |
| `conv2d_summariser.py` | `GeoConvNet2DSummariser` | 2D images | `(B,3,H,W)` → `(B,3,H/4,W/4)` | 16× spatial area |
| `pointnet2_summariser.py` | `GeoConvNet3DPCSummariser` | 3D point clouds | `N pts` → `N/8 pts` | 8× points |
| `meshcnn_summariser.py` | `GeoConvNet3DMeshSummariser` | 3D meshes | `E edges` → `E/8 edges` | 8× edges |

---

## Architecture Pattern

All four summarisers share the same three-level design:

```
Input
  └─ Encoder (G-equivariant, strided downsampling)
       └─ Summary features  ──► summary_head ──► Summary output
            ├─ cls_head ──────────────────────► logits (B, K)
            └─ Decoder (symmetric, skip connections)
                 └─ recon_head ────────────────► Reconstruction
```

The encoder is identical in spirit to the classification encoder — G-equivariant convolutions adapted to the domain's symmetry group. The decoder is identical in spirit to the segmentation decoder — symmetric upsampling with skip connections.  **Summarisation sits precisely between classification and segmentation** in the GeoConvNet framework.

---

## Symmetry Groups

| Domain | Group | Summarisation operator |
|--------|-------|----------------------|
| 1D | `(Z, +)` translations | Strided 1D conv (temporal downsampling) |
| 2D | `(Z², +)` translations | Strided 2D conv (spatial downsampling) |
| 3D-PC | `S_n` permutations | FPS + Set Abstraction (point subsampling) |
| 3D-Mesh | Gauge symmetry | MeshPool edge collapse (mesh decimation) |

---

## Quick Smoke Tests

```bash
python models/conv1d_summariser.py
python models/conv2d_summariser.py
python models/pointnet2_summariser.py   # requires torch-geometric
python models/meshcnn_summariser.py
```
