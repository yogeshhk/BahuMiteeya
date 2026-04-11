# Models — Segmentation

Each file implements one domain's G-equivariant encoder-decoder. All models output per-element logits (one per timestep / pixel / point / edge) and have a `__main__` smoke test block.

---

## Domain → Model mapping

| Domain | File | Class | Input | Output | Key design |
|--------|------|-------|-------|--------|------------|
| 1D time series | `conv1d_segmenter.py` | `GeoConvNet1D` | `(B, 1, T)` | `(B, K, T)` | Dilated 1D U-Net: dilation {1,2,4,8} encoder + transposed-conv decoder with skip connections |
| 2D images | `conv2d_segmenter.py` | `GeoConvNet2D` | `(B, 3, H, W)` | `(B, K, H, W)` | U-Net with ResNet-18 encoder; bilinear decoder with skip connections |
| 3D point cloud | `pointnet2_segmenter.py` | `GeoConvNet3DPCSeg` | PyG `Data` | `(N_total, K)` | PointNet++ encoder (3 SA levels) + Feature Propagation decoder (kNN interpolation) |
| 3D mesh | `meshcnn_segmenter.py` | `GeoConvNet3DMeshSeg` | `(E,5)` edge feats + `(E,4)` neighbor idx | `(E, K)` | MeshCNN encoder-decoder: MeshPool stores `keep_idx`; MeshUnpool broadcasts features back |

---

## Theoretical grounding (per Bronstein et al., 2021)

The encoder-decoder pattern is domain-universal: encode with a G-equivariant operator (downsampling), decode by inverting the downsampling operation while injecting skip features:

- **1D** — transposed 1D-conv restores temporal resolution
- **2D** — bilinear upsampling (F.interpolate) restores spatial resolution
- **3D-PC** — kNN inverse-distance weighted interpolation (FeaturePropagation) restores point resolution
- **3D-Mesh** — MeshUnpool nearest-fill broadcasts coarse edge features to fine edge resolution

---

## Adding a new domain

1. Create `models/<domain>_segmenter.py`.
2. Implement a class that outputs `(N_elements, num_classes)` per-element logits.
3. Add a `__main__` smoke test (see existing files for pattern).
4. Register it in `train.py` and `evaluate.py` under a new `--domain` choice.
5. Add a data loader in `datasets/loaders.py`.

---

## Smoke testing

```bash
# From src/segmentation/
python models/conv1d_segmenter.py
python models/conv2d_segmenter.py
python models/pointnet2_segmenter.py   # requires torch-geometric
python models/meshcnn_segmenter.py
```
