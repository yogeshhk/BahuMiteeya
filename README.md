# BahuMeetiya · बहुमितीय

### A Unified Graph Neural Network Framework Across 1D, 2D, and 3D Geometric Domains

> *बहुमितीय (Bahumeetiya)* — Sanskrit for **"multi-dimensional"**.  
> One framework. Three geometric worlds. Graphs all the way down.

---

## What Is This?

Most GNN frameworks are built for one geometry and one task.  
Point cloud networks think in 3D. Graph networks ignore coordinates. Image models live on grids.

**BahuMeetiya asks a different question:**  
What if the same graph-theoretic machinery — with geometry-aware positional encodings and equivariant message passing — could handle a curve, a floorplan, and a molecule with the same code?

This repository is the ongoing answer. It implements classification and segmentation models across:

| Dimension | Data type | Examples |
|-----------|-----------|---------|
| 1D | Curves, sequences, skeletons | Midsurfaces, medial axes, time-series graphs |
| 2D | Planar graphs, floorplans, schematics | Architectural layouts, chip floorplans, circuit schematics |
| 3D | Meshes, point clouds, B-Rep graphs | CAD solids, molecular graphs, scene graphs |

The unifying abstraction: **everything is a graph with geometry attached to it**.

---

## Motivation

Two decades of working with geometry in CAD/CAM software taught me that the hardest problems — midsurface extraction, netlist optimisation, floorplan retrieval — share a deep structure.

They are all **graph-to-graph or graph prediction problems on geometric data**.

The midsurface of a 3D solid is a 2D graph embedded in 3D space.  
A chip netlist is a directed graph with spatial coordinates after placement.  
An architectural floorplan is a planar graph with area and adjacency constraints.

Once you see this, a single framework becomes possible. BahuMeetiya is that framework — built from genuine research need, not tutorial demos.

---

## Architecture Overview

```
bahumeetiya/
│
├── core/
│   ├── encoders/
│   │   ├── laplacian_pe.py        # Laplacian eigenvector positional encoding
│   │   ├── rwse.py                # Random walk structural encoding
│   │   └── geometric_pe.py        # Equivariant coordinate encoding (SE(3))
│   │
│   ├── convolutions/
│   │   ├── sage_conv.py           # GraphSAGE (baseline, handles irregular fanin)
│   │   ├── gat_conv.py            # Graph Attention (edge-feature aware)
│   │   └── equivariant_conv.py    # E(n)-equivariant message passing
│   │
│   └── decoders/
│       ├── node_head.py           # Node classification / regression
│       ├── graph_head.py          # Graph classification (global pooling)
│       └── graph_gen_head.py      # Non-autoregressive graph generation
│
├── domains/
│   ├── dim1/
│   │   ├── dataset.py             # Curve and skeleton graph datasets
│   │   └── model.py               # 1D-specialised GNN config
│   │
│   ├── dim2/
│   │   ├── dataset.py             # Floorplan, schematic, planar graph datasets
│   │   └── model.py               # 2D-specialised GNN config
│   │
│   └── dim3/
│       ├── dataset.py             # Mesh, point cloud, B-Rep datasets
│       └── model.py               # 3D-specialised GNN (equivariant)
│
├── experiments/
│   ├── midsurface_segmentation/   # 3D → 2D graph-to-graph transform
│   ├── floorplan_retrieval/       # 2D spatial graph RAG
│   ├── netlist_timing/            # EDA: timing slack prediction on netlists
│   └── molecule_property/         # 3D: molecular graph classification
│
├── configs/                       # YAML experiment configs
├── notebooks/                     # Walkthroughs and visualisations
└── tests/
```

---

## Tasks Supported

### Classification
Assign a label to an entire graph or each node within it.

- **Graph classification**: "Is this molecule toxic?" / "What building type is this floorplan?"
- **Node classification**: "Is this gate on the critical timing path?" / "Which surface patch belongs to which feature?"

### Segmentation
Partition graph nodes into meaningful groups — the graph equivalent of semantic segmentation.

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
Can the same GNN backbone — with appropriate PE — handle 1D, 2D, and 3D geometric graphs
without task-specific architectural surgery?

---

## Getting Started

```bash
git clone https://github.com/yogeshkulkarni/bahumeetiya.git
cd bahumeetiya
pip install -r requirements.txt
```

**Run the EDA timing experiment (quickest demo):**
```bash
python experiments/netlist_timing/netlist_gnn_timing.py
```
Generates a synthetic chip netlist, trains a GNN to predict timing slack,
and produces a colour-coded visualisation of predicted vs. actual slack across the netlist graph.

**Run floorplan classification:**
```bash
python experiments/floorplan_retrieval/train.py --config configs/floorplan_2d.yaml
```

**Run 3D mesh segmentation:**
```bash
python experiments/midsurface_segmentation/train.py --config configs/midsurface_3d.yaml
```

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
| Midsurface pairs | 3D→2D | Graph generation | Author-generated |
| RPLAN | 2D | Floorplan segmentation | RPLAN public dataset |
| ModelNet40 (graph form) | 3D | Mesh classification | ModelNet |
| QM9 | 3D | Molecule property regression | PyG built-in |
| Synthetic netlists | 2D | Timing slack prediction | This repo |
| SECOM (semiconductor) | tabular+graph | Yield prediction | UCI |

---

## Background

This framework grew out of two decades of applied geometric computing:

- **Midsurface NN**: Graph-to-graph neural network for extracting 2D midsurfaces from 3D CAD solids — the problem that first made me think seriously about graph generation on geometric data
- **CAD/CAM at Siemens UGS and Autodesk**: Where the geometric problems lived in production
- **AI/ML consulting**: Where the demand for graph-based solutions across industries became clear
- **EDA interest**: Chip netlists as the industrial-scale geometric graph problem of the decade

---

## Connection to EDA / Chip Design

The experiments/netlist_timing experiment is a deliberately minimal demonstration
that netlist analysis is a natural GNN task.

A real chip netlist (from Cadence Innovus or Synopsys IC Compiler, or open-source OpenROAD)
can be loaded as a PyG graph and used directly with BahuMeetiya models.

If you are working in EDA and interested in ML-based timing prediction, congestion forecasting,
or placement optimisation — the framework is designed to be extended in this direction.
Reach out or open an issue.

---

## Research Papers That Inspired This Work

- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (2021)
- Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* — GPS (2022)
- Ma et al., *Graph Inductive Biases in Transformers without Message Passing* — GRIT (2023)
- Mirhoseini et al., *A graph placement methodology for fast chip design* — Google, Nature (2021)
- Lim et al., *Sign and Basis Invariant Networks for Spectral Graph Neural Networks* — SignNet (2023)

---

## Author

**Yogesh Kulkarni**  
AI Coach · Graph ML Researcher · Geometric Modeling PhD  
Pune, India

[LinkedIn](https://linkedin.com/in/yogeshkulkarni) · [GitHub](https://github.com/yogeshkulkarni) · yogeshkulkarni@yahoo.com

---

## Licence

MIT. Use it, break it, extend it — and tell me what you find.

---

> *"Graphs are the highest level of abstraction in Geometric Deep Learning —  
> solid, abstract, and possibly mathematical enough for the rest of one's life."*

---

![BahuMeetiya — 1D 2D 3D graph domains unified](docs/banner.png)

<!-- Badge placeholders — activate once CI is set up -->
<!-- ![Tests](https://github.com/yogeshkulkarni/bahumeetiya/actions/workflows/test.yml/badge.svg) -->
<!-- ![Python](https://img.shields.io/badge/python-3.10+-blue) -->
<!-- ![License](https://img.shields.io/badge/license-MIT-green) -->
