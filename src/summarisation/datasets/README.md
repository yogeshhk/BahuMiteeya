# Summarisation Datasets

The same four benchmark datasets used in classification and segmentation are reused here. Each domain's summarisation task transforms the input into a smaller but semantically equivalent representation, evaluated by downstream classification accuracy and reconstruction fidelity.

## Dataset Summary

| Domain | Dataset | Input Size | Summary Size | Compression |
|--------|---------|------------|--------------|-------------|
| 1D | UCR ECG5000 | 140 timesteps | 35 timesteps | 4× |
| 2D | CIFAR-10 | 32×32 pixels | 8×8 pixels | 16× area |
| 3D-PC | ModelNet40 | 1,024 points | 128 points | 8× |
| 3D-Mesh | SHREC16 | ~2,000 edges | ~250 edges | 8× |

---

## 1D — UCR ECG5000

- **Source**: [timeseriesclassification.com](https://timeseriesclassification.com/description.php?Dataset=ECG5000)
- **Format**: Comma-delimited `.txt`; first column is 1-indexed label
- **Place at**: `data/ucr/ECG5000/ECG5000_TRAIN.txt` and `ECG5000_TEST.txt`
- **Task**: Compress 140-step ECG signals to 35-step summaries; verify downstream 5-class classification accuracy is preserved

---

## 2D — CIFAR-10

- **Source**: Downloaded automatically via `torchvision.datasets.CIFAR10`
- **Place at**: `data/cifar10/`
- **Task**: Compress 32×32 RGB images to 8×8 summary images; verify 10-class classification accuracy on the compact representation

---

## 3D Point Cloud — ModelNet40

- **Source**: Downloaded automatically via `torch_geometric.datasets.ModelNet`
- **Place at**: `data/modelnet40/`
- **Task**: Compress 1,024-point clouds to 128-point summaries via FPS-based set abstraction; verify 40-class shape classification on the sparser cloud

---

## 3D Mesh — SHREC16

- **Source**: Download from the [MeshCNN repo](https://github.com/ranahanocka/MeshCNN#datasets)
- **Format**: `.npz` files with keys `edge_features (E×5)`, `neighbor_idx (E×4)`, `label`
- **Place at**: `data/shrec16/train/` and `data/shrec16/test/`
- **Task**: Coarsen a ~2,000-edge mesh to a ~250-edge summary via learned edge collapse; verify 30-class shape classification on the coarse mesh

---

## Evaluation Metrics

For each domain, three metrics are reported:

| Metric | Description |
|--------|-------------|
| **Compression ratio** | `\|input\| / \|summary\|` (element count) |
| **Downstream accuracy** | Classification accuracy of a head trained on summaries |
| **Reconstruction quality** | MSE (1D/2D/Mesh) or Chamfer distance (3D-PC) between reconstructed and original signal |
