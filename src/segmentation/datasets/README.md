# Datasets — Segmentation

This folder's `loaders.py` provides data loaders for all four GeoConvNet segmentation domains.

---

## Download Instructions

### 1D — UCR ECG5000 with synthetic motif labels (manual download required)

1. Go to https://timeseriesclassification.com/description.php?Dataset=ECG5000
2. Download the dataset archive and extract it.
3. **Rename** `ECG5000_TRAIN.ts` → `ECG5000_TRAIN.txt` and `ECG5000_TEST.ts` → `ECG5000_TEST.txt`.
   The loader uses `np.loadtxt(..., delimiter=",")` which expects plain CSV — the `.ts` format header will cause an error if not renamed.
4. Place both files in `data/ucr/ECG5000/`.

Expected layout:
```
data/ucr/ECG5000/
    ECG5000_TRAIN.txt
    ECG5000_TEST.txt
```

> **Note on labels:** UCR ECG5000 provides only per-series class labels. The `UCRMotifDataset` class synthesizes per-timestep segmentation labels via sliding-window template matching against per-class centroids. This is a heuristic — replace `_make_seg_labels()` with ground-truth annotations if available.

---

### 2D — PASCAL VOC 2012 (automatic download)

Downloaded automatically by `torchvision` on first run. Pass `--data_dir data/voc --num_classes 21`.

Label encoding: 0 = background, 1–20 = object classes, 255 = ignore (remapped to -1 by the loader).

---

### 3D Point Cloud — ShapeNetPart (automatic download)

Downloaded automatically by `torch_geometric.datasets.ShapeNet` on first run.
Pass `--data_dir data/shapenet --num_classes 50`. Requires `torch-geometric` to be installed.

---

### 3D Mesh — COSEG (manual download required)

1. Follow the download instructions at https://github.com/ranahanocka/MeshCNN#datasets
2. Convert meshes to `.npz` format using the MeshCNN preprocessing scripts, ensuring `seg_labels` are included.
3. Place the result under `data/coseg/<object_class>/` with this layout:

```
data/coseg/
    vases/
        train/  *.npz
        test/   *.npz
    chairs/
        train/  *.npz
        test/   *.npz
```

Pass `--data_dir data/coseg/vases --num_classes 4` (adjust class count per object set).

Each `.npz` must contain:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `edge_features` | `(E, 5)` | float32 | 5D intrinsic edge features |
| `neighbor_idx` | `(E, 4)` | int32 | Indices of the 4 neighbor edges per edge |
| `seg_labels` | `(E,)` | int32 | Per-edge part label (0-indexed) |
