# Models — Classification

Each file implements one domain's G-equivariant encoder. All models output `(B, num_classes)` logits and have a `__main__` smoke test block.

---

## Domain → Model mapping

| Domain | File | Class | Input | Key design |
|--------|------|-------|-------|------------|
| 1D time series | `conv1d_classifier.py` | `GeoConvNet1D` | `(B, 1, T)` | 3 residual 1D-conv blocks, kernel sizes {8,5,3}, global avg pool |
| 2D images | `conv2d_classifier.py` | `GeoConvNet2D` | `(B, 3, H, W)` | ResNet-18 variant, adapted for 32×32 CIFAR (no maxpool in stem) |
| 3D point cloud | `pointnet2_classifier.py` | `GeoConvNet3DPC` | PyG `Data` | PointNet++ MSG: 3 set-abstraction levels + global max pool |
| 3D mesh | `meshcnn_classifier.py` | `GeoConvNet3DMesh` | `(E,5)` edge feats + `(E,4)` neighbor idx | Edge-based MeshCNN: 4 MeshConv layers + 3 MeshPool levels |

---

## Theoretical grounding (per Bronstein et al., 2021)

Each model is the canonical G-equivariant operator for its symmetry group:

- **1D** → group `(Z, +)` — translation equivariance → 1D convolution
- **2D** → group `(Z², +)` — translation equivariance → 2D convolution
- **3D-PC** → group `S_n` — permutation invariance → set abstraction
- **3D-Mesh** → gauge symmetry — local frame invariance → sorted edge-pair aggregation

---

## Adding a new domain

1. Create `models/conv_<domain>_classifier.py`.
2. Implement a class that outputs `(B, num_classes)` logits.
3. Add a `__main__` smoke test (see existing files for pattern).
4. Register it in `train.py` and `evaluate.py` under a new `--domain` choice.
5. Add a data loader in `datasets/loaders.py`.

---

## Smoke testing

```bash
# From src/classification/
python models/conv1d_classifier.py
python models/conv2d_classifier.py
python models/pointnet2_classifier.py   # requires torch-geometric
python models/meshcnn_classifier.py
```
