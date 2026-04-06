# Datasets — Classification

This folder's `loaders.py` provides data loaders for all four GeoConvNet classification domains.

---

## Download Instructions

### 1D — UCR ECG5000 (manual download required)

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

Format: comma-delimited CSV. First column is a 1-indexed class label; remaining columns are timestep values.

---

### 2D — CIFAR-10 (automatic download)

Downloaded automatically by `torchvision` on first run. Pass `--data_dir data/cifar10`.

---

### 3D Point Cloud — ModelNet40 (automatic download)

Downloaded automatically by `torch_geometric.datasets.ModelNet` on first run.
Pass `--data_dir data/modelnet40`. Requires `torch-geometric` to be installed.

---

### 3D Mesh — SHREC16 (manual download required)

1. Follow the download instructions at https://github.com/ranahanocka/MeshCNN#datasets
2. Convert meshes to `.npz` format using the MeshCNN preprocessing scripts.
3. Place the result under `data/shrec16/` with this layout:

```
data/shrec16/
    train/
        class_000/  mesh_001.npz  mesh_002.npz  ...
        class_001/  ...
    test/
        class_000/  ...
        class_001/  ...
```

Each `.npz` must contain:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `edge_features` | `(E, 5)` | float32 | 5D intrinsic edge features (dihedral angle, inner angles, ratios) |
| `neighbor_idx` | `(E, 4)` | int32 | Indices of the 4 neighbor edges per edge |
| `label` | scalar | int | Class label (0-indexed) |
