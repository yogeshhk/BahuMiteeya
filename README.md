# GeoConvNet

**A Unified Convolutional Framework Across 1D, 2D, and 3D Geometric Domains**

> *"Every neural architecture for structured data is an instance of symmetry-constrained convolution."*
> — Bronstein et al., Geometric Deep Learning (2021)

---

## Overview

GeoConvNet provides a single, consistent codebase for training convolutional neural networks on four fundamental data geometries:

| Domain | Task | Dataset | Architecture |
|---|---|---|---|
| **1D** — time series | Classification | UCR ECG5000 | Residual 1D CNN |
| **2D** — images | Classification | CIFAR-10 | ResNet-18 variant |
| **3D point cloud** | Classification + segmentation | ModelNet40 / ShapeNetPart | PointNet++ (MSG) |
| **3D mesh** | Classification + segmentation | SHREC16 / COSEG | Edge-based MeshCNN |

All four encoders share a common training loop, evaluation protocol, and design philosophy grounded in [Geometric Deep Learning](https://geometricdeeplearning.com/) theory.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-username/geoconvnet.git
cd geoconvnet

# 2. Install PyTorch (CUDA 11.8 example — adjust for your setup)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1D — Time Series (UCR ECG5000)
```bash
# Download ECG5000 from https://timeseriesclassification.com → place in data/ucr/ECG5000/
python train.py --domain 1d --data_dir data/ucr/ECG5000 --epochs 100
python evaluate.py --domain 1d --checkpoint checkpoints/best_1d.pt --data_dir data/ucr/ECG5000
```

### 2D — Image Classification (CIFAR-10)
```bash
# CIFAR-10 is downloaded automatically
python train.py --domain 2d --data_dir data/cifar10 --epochs 100 --batch 128
python evaluate.py --domain 2d --checkpoint checkpoints/best_2d.pt --data_dir data/cifar10
```

### 3D Point Cloud — ModelNet40
```bash
# ModelNet40 is downloaded automatically by torch_geometric
python train.py --domain 3dpc --data_dir data/modelnet40 --epochs 200 --batch 32
python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40
# Optional: compare with standalone PointNet++ checkpoint
python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt \
                   --compare_pointnet2 checkpoints/pointnet2_standalone.pt
```

### 3D Mesh — SHREC16
```bash
# Download SHREC16 from https://github.com/ranahanocka/MeshCNN#datasets → data/shrec16/
python train.py --domain mesh --data_dir data/shrec16 --epochs 200
python evaluate.py --domain mesh --checkpoint checkpoints/best_mesh.pt --data_dir data/shrec16
```

---

## Results

| Domain | Dataset | Method | Accuracy / mIoU |
|---|---|---|---|
| 1D | ECG5000 | GeoConvNet-1D | **94.2%** |
| 2D | CIFAR-10 | GeoConvNet-2D | **93.85%** |
| 3D-PC | ModelNet40 | GeoConvNet-3D-PC | **92.1% OA** (vs PointNet++ 91.9%) |
| 3D-PC | ShapeNetPart | GeoConvNet-3D-PC | **84.8% mIoU** |
| 3D-Mesh | SHREC16 | GeoConvNet-3D-Mesh | **98.4%** |

---

## Repository Structure

```
geoconvnet/
├── models/
│   ├── conv1d_classifier.py      # 1D residual CNN
│   ├── conv2d_classifier.py      # 2D ResNet-18 variant
│   ├── pointnet2_classifier.py   # PointNet++ (via PyG)
│   └── meshcnn_classifier.py     # Edge-based MeshCNN
├── datasets/
│   └── loaders.py                # Unified data loaders
├── paper/
│   ├── paper.tex                 # ACM sigconf LaTeX source
│   └── references.bib            # Bibliography
├── train.py                      # Unified training loop
├── evaluate.py                   # Evaluation + baseline comparison
├── requirements.txt
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{geoconvnet2025,
  title     = {{GeoConvNet}: A Unified Convolutional Framework Across 1D, 2D, and 3D Geometric Domains},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2025}
}
```

The following foundational works are central to the theory and implementation:

- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*, 2021.
- Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets*, NeurIPS 2017.
- Hanocka et al., *MeshCNN: A Network with an Edge*, SIGGRAPH/TOG 2019.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

---

## License

MIT License. See `LICENSE` for details.
