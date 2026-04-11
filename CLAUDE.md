# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BahuMiteeya** (meaning "multi-dimensional" in Sanskrit) hosts **GeoConvNet** — a research codebase demonstrating that convolution-based neural networks for 1D, 2D, and 3D data are instances of the same symmetry-constrained equivariant operation (per Bronstein et al., Geometric Deep Learning, 2021).

The repo contains three parallel, self-contained sub-projects:
- `src/classification/` — classifying whole samples (ECG, CIFAR-10, ModelNet40, SHREC16)
- `src/segmentation/` — dense per-element labeling (ECG motifs, PASCAL VOC, ShapeNetPart, COSEG)
- `src/summarisation/` — structural compression N→M (ECG, CIFAR-10, ModelNet40, SHREC16)

All sub-projects share the same four geometric domains: **1D time series**, **2D images**, **3D point clouds** (via PyTorch Geometric), and **3D meshes** (MeshCNN edge-based format).

The three tasks form a progression:
| Task | Output | N→? |
|------|--------|-----|
| Classification | 1 global label | N → 1 |
| Summarisation | M-element summary (M ≪ N) | N → M |
| Segmentation | N per-element labels | N → N |

## Installation

```bash
# PyTorch first (adjust CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# Remaining deps (same file in both sub-projects)
pip install -r src/classification/requirements.txt
```

Python 3.10, PyTorch 2.1, CUDA 11.8 is the tested configuration.

## Commands

All commands must be run from within the relevant sub-project directory (`src/classification/`, `src/segmentation/`, or `src/summarisation/`) so that relative imports resolve correctly.

### Training

```bash
cd src/classification   # or src/segmentation

python train.py --domain 1d    --data_dir data/ucr/ECG5000
python train.py --domain 2d    --data_dir data/cifar10 --batch 128
python train.py --domain 3dpc  --data_dir data/modelnet40 --epochs 200 --batch 32
python train.py --domain mesh  --data_dir data/shrec16 --epochs 200
```

Segmentation adds `--num_classes <N>`:
```bash
python train.py --domain 1d   --data_dir data/ucr/ECG5000 --num_classes 6
python train.py --domain 2d   --data_dir data/voc         --num_classes 21
python train.py --domain 3dpc --data_dir data/shapenet    --num_classes 50
python train.py --domain mesh --data_dir data/coseg/vases --num_classes 4
```

Summarisation adds `--alpha <float>` (classification vs. reconstruction weight):
```bash
cd src/summarisation

python train.py --domain 1d   --data_dir data/ucr/ECG5000
python train.py --domain 2d   --data_dir data/cifar10 --batch 128
python train.py --domain 3dpc --data_dir data/modelnet40 --epochs 200
python train.py --domain mesh --data_dir data/shrec16 --epochs 200

# Adjust alpha (default 0.5): 1.0 = classification only, 0.0 = reconstruction only
python train.py --domain 1d --data_dir data/ucr/ECG5000 --alpha 0.8
```

Checkpoints are saved to `checkpoints/best_<domain>.pt`.

### Evaluation

```bash
python evaluate.py --domain 1d   --checkpoint checkpoints/best_1d.pt   --data_dir data/ucr/ECG5000
python evaluate.py --domain 2d   --checkpoint checkpoints/best_2d.pt   --data_dir data/cifar10
python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40
python evaluate.py --domain mesh --checkpoint checkpoints/best_mesh.pt  --data_dir data/shrec16

# Optional PointNet++ baseline comparison (3dpc only):
python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt \
                   --data_dir data/modelnet40 \
                   --compare_pointnet2 checkpoints/pointnet2_standalone.pt
```

### Smoke-testing a model

Each model file has a `__main__` block for a quick forward-pass check:
```bash
python models/conv1d_classifier.py
python models/conv2d_classifier.py
# etc.
```

## Architecture

### Shared training loop design

`train.py` uses three separate `run_epoch_*` functions because each domain has a different data format:
- `run_epoch_standard` — standard `(x, y)` PyTorch DataLoader (1D, 2D)
- `run_epoch_pyg` — PyG `Data` objects with `.y` attribute (3D point cloud)
- `run_epoch_mesh` — list of `(edge_features, neighbor_idx, label)` tuples; processed per-sample due to variable mesh topology (3D mesh)

### Model classes and their inputs

| Domain | Class | Input tensor | Key design |
|---|---|---|---|
| 1D | `GeoConvNet1D` | `(B, 1, T)` | 3 residual blocks, kernel sizes {8,5,3}, global avg pool |
| 2D | `GeoConvNet2D` | `(B, 3, H, W)` | ResNet-18 variant |
| 3D-PC | `GeoConvNet3DPC` | PyG `Data` object | PointNet++ MSG via PyG |
| 3D-Mesh | `GeoConvNet3DMesh` | `(E, 5)` edge feats + `(E, 4)` neighbor idx | Edge-based MeshCNN |

### Data loaders (`datasets/loaders.py`)

- **UCR/ECG5000**: expects `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt` (comma-delimited, first column is 1-indexed label)
- **CIFAR-10 / PASCAL VOC / ModelNet40 / ShapeNetPart**: downloaded automatically by torchvision / PyG
- **SHREC16** (classification) and **COSEG** (segmentation): must be downloaded manually from the MeshCNN repo; stored as `.npz` files with keys `edge_features`, `neighbor_idx`, `label`

### Publications

`publications/LaTeX/` contains two IEEE-format papers:
- `Main_geocovnet_classification_ieee_paper.tex` — classification paper
- `Main_geocovnet_segmentation_ieee_paper.tex` — segmentation paper

Each has its own `.bib` file (`classification_references.bib`, `segmentation_references.bib`).
