# GeoConvNet — Summarisation

**Graph/signal summarisation** across 1D, 2D, and 3D geometric domains.

Summarisation is the task of transforming a large structured signal into a smaller, semantically equivalent one — a graph with fewer nodes and edges, a point cloud with fewer points, an image with fewer pixels, or a time series with fewer timesteps — while preserving the properties that matter downstream (classification accuracy) and retaining enough information to approximately reconstruct the original (reconstruction fidelity).

This sub-project is the third in the GeoConvNet series, complementing:
- `src/classification/` — global label assignment (N → 1)
- `src/segmentation/`   — per-element labeling (N → N)
- `src/summarisation/`  — structural compression (N → M, M ≪ N) ← **this sub-project**

---

## Compression Ratios

| Domain | Dataset | Input size | Summary size | Compression |
|--------|---------|------------|--------------|-------------|
| 1D | UCR ECG5000 | 140 timesteps | 35 timesteps | 4× |
| 2D | CIFAR-10 | 32×32 px | 8×8 px | 16× area |
| 3D-PC | ModelNet40 | 1,024 points | 128 points | 8× |
| 3D-Mesh | SHREC16 | ~2,000 edges | ~250 edges | 8× |

---

## Training

All commands must be run from within `src/summarisation/`.

```bash
cd src/summarisation

# 1D time series
python train.py --domain 1d --data_dir data/ucr/ECG5000

# 2D images
python train.py --domain 2d --data_dir data/cifar10 --batch 128

# 3D point clouds
python train.py --domain 3dpc --data_dir data/modelnet40 --epochs 200

# 3D meshes
python train.py --domain mesh --data_dir data/shrec16 --epochs 200
```

The `--alpha` argument controls the loss balance:
```bash
# Equal weight (default: alpha=0.5)
python train.py --domain 1d --data_dir data/ucr/ECG5000 --alpha 0.5

# Reconstruction-focused (alpha=0.2 → 20% cls, 80% recon)
python train.py --domain 1d --data_dir data/ucr/ECG5000 --alpha 0.2

# Task-preservation-focused (alpha=0.8 → 80% cls, 20% recon)
python train.py --domain 1d --data_dir data/ucr/ECG5000 --alpha 0.8
```

Checkpoints are saved to `checkpoints/best_<domain>.pt`.

---

## Evaluation

```bash
python evaluate.py --domain 1d   --checkpoint checkpoints/best_1d.pt   --data_dir data/ucr/ECG5000
python evaluate.py --domain 2d   --checkpoint checkpoints/best_2d.pt   --data_dir data/cifar10
python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40
python evaluate.py --domain mesh --checkpoint checkpoints/best_mesh.pt  --data_dir data/shrec16
```

Reports:
- **Downstream accuracy** — classification accuracy of a head trained on the summaries
- **Reconstruction quality** — MSE (1D/2D/Mesh) or Chamfer distance (3D-PC)
- **Compression ratio** — |input| / |summary| element count

---

## Smoke-testing a model

```bash
python models/conv1d_summariser.py
python models/conv2d_summariser.py
python models/pointnet2_summariser.py
python models/meshcnn_summariser.py
```

---

## Architecture

Each summariser follows the pattern:

```
Input signal
  → G-equivariant encoder (same symmetry as classification/segmentation)
      → Summary features (bottleneck)
          ├─ summary_head   → Summary output  (same format as input, M < N)
          ├─ cls_head       → Class logits     (downstream task)
          └─ recon decoder  → Reconstruction   (same format/size as input)
```

This makes summarisation the **structural middle ground** between classification (encoder only, global output) and segmentation (full encoder-decoder, per-element output).

| Task | Encoder | Decoder | Output |
|------|---------|---------|--------|
| Classification | ✓ | ✗ | 1 global label |
| **Summarisation** | **✓** | **✓ (compact)** | **M < N elements** |
| Segmentation | ✓ | ✓ (full) | N labels (same size) |

---

## Loss Function

```
L = α · CrossEntropy(logits, y) + (1-α) · ReconLoss(recon, x)
```

| Domain | ReconLoss |
|--------|-----------|
| 1D | MSE between reconstructed and original time series |
| 2D | MSE between reconstructed and original image |
| 3D-PC | Symmetric Chamfer distance on point positions |
| 3D-Mesh | MSE between reconstructed and original edge features |
