# BahuMeetiya · बहुमितीय

### A Unified Graph Neural Network Framework Across 1D, 2D, and 3D Geometric Domains

> *बहुमितीय (Bahumeetiya)*, Sanskrit for **"multi-dimensional"**.  
> One framework. Three geometric worlds. Graphs all the way down.

---

## What Is This?

Most GNN frameworks are built for one geometry and one task.  
Point cloud networks think in 3D. Graph networks ignore coordinates. Image models live on grids.

**BahuMeetiya asks a different question:**  
What if the same graph-theoretic machinery, with geometry-aware positional encodings and equivariant message passing, could handle a curve, a floorplan, and a molecule with the same code?

This repository is the ongoing answer. It implements classification and segmentation models across:

| Dimension | Data type | Examples |
|-----------|-----------|---------|
| 1D | Curves, sequences, skeletons | Midsurfaces, medial axes, time-series graphs |
| 2D | Planar graphs, floorplans, schematics | Architectural layouts, chip floorplans, circuit schematics |
| 3D | Meshes, point clouds, B-Rep graphs | CAD solids, molecular graphs, scene graphs |

The unifying abstraction: **everything is a graph with geometry attached to it**.

---

## Motivation

Two decades of working with geometry in CAD/CAM software taught me that the hardest problems, midcurve extraction, floorplan retrieval, share a deep structure.

They are all **graph-to-graph or graph prediction problems on geometric data**.

The midcurve of a 2D Profile is a 1D graph embedded in 2D space.  
An architectural floorplan is a planar graph with area and adjacency constraints.

Once you see this, a single framework becomes possible. BahuMeetiya is that framework, built from genuine research need, not tutorial demos.

---

## Tasks Supported

The three implemented tasks form a natural progression in output complexity:

| Task | Output | N→? | Paper |
|------|--------|-----|-------|
| **Classification** | 1 global label | N → 1 | `publications/LaTeX/Main_geocovnet_classification_ieee_paper.tex` |
| **Summarisation** | M-element summary (M ≪ N) | N → M | `publications/LaTeX/Main_geocovnet_summarisation_ieee_paper.tex` |
| **Segmentation** | N per-element labels | N → N | `publications/LaTeX/Main_geocovnet_segmentation_ieee_paper.tex` |

### Classification
Assign a label to an entire graph or each node within it.

- **Graph classification**: "Is this molecule toxic?" / "What building type is this floorplan?"
- **Node classification**: "Is this gate on the critical timing path?" / "Which surface patch belongs to which feature?"

### Summarisation *(new)*
Transform a large structured signal into a smaller, semantically equivalent one — a graph with fewer nodes and edges, a point cloud with fewer points, an image with fewer pixels, or a time series with fewer timesteps — while preserving downstream task performance and approximate reconstructibility.

- **1D**: Compress a 140-step ECG to a 35-step representative summary (4× compression)
- **2D**: Compress a 32×32 image to an 8×8 semantic summary (16× area compression)
- **3D point cloud**: Reduce 1,024 points to 128 representative points (8× compression)
- **3D mesh**: Coarsen a ~2,000-edge mesh to ~250 edges via learned edge collapse (8× compression)

Trained with a dual objective: `α · CrossEntropy(downstream classification) + (1−α) · ReconLoss(reconstruction fidelity)`. The `α` parameter provides a principled knob for trading task preservation against reconstruction quality.

### Segmentation
Partition graph nodes into meaningful groups, the graph equivalent of semantic segmentation.

- 3D mesh segmentation: identify CAD features (holes, fillets, chamfers) from B-Rep graphs
- 2D floorplan segmentation: label rooms, corridors, structural walls
- 1D skeleton segmentation: identify branches, junctions, endpoints of midsurfaces

### Coming: Graph Generation
Graph-to-graph transforms (midsurface extraction, netlist simplification) as non-autoregressive generation with optimal transport training.

---

## The Open Research Questions This Repo Explores

**1. Positional encoding on geometric graphs**  
Standard Laplacian PE ignores coordinates. Raw coordinates break rotation invariance.  
BahuMeetiya experiments with *learned compositions* of structural and geometric signals.

**2. Autoregressive vs. non-autoregressive graph generation**  
For graph-to-graph transforms with no canonical node ordering, non-autoregressive decoding
trained with optimal transport loss is theoretically cleaner. We test this hypothesis.

**3. Dimension-agnostic graph representations**  
Can the same GNN backbone, with appropriate PE, handle 1D, 2D, and 3D geometric graphs
without task-specific architectural surgery?

---

## Getting Started

```bash
git clone https://github.com/yogeshhk/BahuMiteeya.git
cd BahuMiteeya

# Install PyTorch first (adjust CUDA version for your system)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install -r src/classification/requirements.txt
```

**Run 1D time series classification (quickest demo):**
```bash
cd src/classification
# Download ECG5000 → data/ucr/ECG5000/ (see datasets/README.md)
python train.py --domain 1d --data_dir data/ucr/ECG5000 --epochs 100
```

**Run 2D image classification (CIFAR-10 downloads automatically):**
```bash
cd src/classification
python train.py --domain 2d --data_dir data/cifar10 --batch 128
```

**Run 1D time series summarisation (4× compression, balanced dual loss):**
```bash
cd src/summarisation
python train.py --domain 1d --data_dir data/ucr/ECG5000
```

**Run 3D point cloud summarisation (8×, task-preservation-focused):**
```bash
cd src/summarisation
python train.py --domain 3dpc --data_dir data/modelnet40 --alpha 0.8
```

**Run 2D semantic segmentation (PASCAL VOC downloads automatically):**
```bash
cd src/segmentation
python train.py --domain 2d --data_dir data/voc --num_classes 21
```

**Run 3D mesh segmentation:**
```bash
cd src/segmentation
# Download COSEG → data/coseg/vases/ (see datasets/README.md)
python train.py --domain mesh --data_dir data/coseg/vases --num_classes 4
```

See `src/classification/README.md`, `src/summarisation/README.md`, and `src/segmentation/README.md` for full usage.

---

## Dependencies

```
torch >= 2.0
torch-geometric >= 2.4
networkx >= 3.0
numpy
matplotlib
pyyaml
```

Optional (for equivariant 3D experiments):
```
e3nn >= 0.5
```

---

## Datasets

| Dataset | Dimension | Task | Source |
|---------|-----------|------|--------|
| UCR ECG5000 | 1D | Classification / Summarisation | [timeseriesclassification.com](https://timeseriesclassification.com) |
| UCR ECG5000 (motif labels) | 1D | Segmentation | [timeseriesclassification.com](https://timeseriesclassification.com) |
| CIFAR-10 | 2D | Classification / Summarisation | `torchvision` (auto-download) |
| PASCAL VOC 2012 | 2D | Segmentation | `torchvision` (auto-download) |
| ModelNet40 | 3D-PC | Classification / Summarisation | `torch_geometric` (auto-download) |
| ShapeNetPart | 3D-PC | Segmentation | `torch_geometric` (auto-download) |
| SHREC16 | 3D-Mesh | Classification / Summarisation | [MeshCNN repo](https://github.com/ranahanocka/MeshCNN#datasets) |
| COSEG | 3D-Mesh | Segmentation | [MeshCNN repo](https://github.com/ranahanocka/MeshCNN#datasets) |

---

## Background

This framework grew out of two decades of applied geometric computing:

- **Midsurface NN**: Graph-to-graph neural network for extracting 2D midsurfaces from 3D CAD solids, the problem that first made me think seriously about graph generation on geometric data
- **CAD/CAM at Siemens UGS and Autodesk**: Where the geometric problems lived in production
- **AI/ML consulting**: Where the demand for graph-based solutions across industries became clear
- **EDA interest**: Chip netlists as the industrial-scale geometric graph problem of the decade

---

<!-- ## Connection to EDA / Chip Design

The experiments/netlist_timing experiment is a deliberately minimal demonstration
that netlist analysis is a natural GNN task.

A real chip netlist (from Cadence Innovus or Synopsys IC Compiler, or open-source OpenROAD)
can be loaded as a PyG graph and used directly with BahuMeetiya models.

If you are working in EDA and interested in ML-based timing prediction, congestion forecasting,
or placement optimisation, the framework is designed to be extended in this direction.
Reach out or open an issue.
 -->
---

## Research Papers That Inspired This Work

- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (2021)
- Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, GPS (2022)
- Ma et al., *Graph Inductive Biases in Transformers without Message Passing*, GRIT (2023)
- Mirhoseini et al., *A graph placement methodology for fast chip design*, Google, Nature (2021)
- Lim et al., *Sign and Basis Invariant Networks for Spectral Graph Neural Networks*, SignNet (2023)

---

## Author

**Yogesh Kulkarni**  
AI Coach · Graph ML Researcher · Geometric Modeling PhD  
Pune, India

[LinkedIn](https://linkedin.com/in/yogeshkulkarni) · [GitHub](https://github.com/yogeshhk) · yogeshkulkarni@yahoo.com

---

## Licence

MIT. Use it, break it, extend it, and tell me what you find.

---

> *"Graphs are the highest level of abstraction in Geometric Deep Learning,  
> solid, abstract, and possibly mathematical enough for the rest of one's life."*

---

![BahuMeetiya, 1D 2D 3D graph domains unified](docs/banner.png)

<!-- Badge placeholders, activate once CI is set up -->
<!-- ![Tests](https://github.com/yogeshhk/bahumeetiya/actions/workflows/test.yml/badge.svg) -->
<!-- ![Python](https://img.shields.io/badge/python-3.10+-blue) -->
<!-- ![License](https://img.shields.io/badge/license-MIT-green) -->
