"""
datasets/loaders.py — Unified dataset loading for all GeoConvNet domains.

Domains:
  1D : UCR Time Series (ECG5000) — CSV format
  2D : CIFAR-10                  — via torchvision
  3D-PC: ModelNet40              — via torch_geometric.datasets.ModelNet
  3D-Mesh: SHREC16               — custom loader for MeshCNN-format data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

# Optional PyG
try:
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, NormalizeScale
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1D: UCR Time Series (ECG5000)
# ---------------------------------------------------------------------------

class UCRDataset(Dataset):
    """
    Loads a UCR time series dataset from its standard 2-column TSV format:
      label  v0  v1  ... vT
    Labels are remapped to [0, num_classes-1].

    Download ECG5000 from: https://timeseriesclassification.com/description.php?Dataset=ECG5000
    Place ECG5000_TRAIN.ts and ECG5000_TEST.ts in data/ucr/ECG5000/
    """

    def __init__(self, path: str):
        data = np.loadtxt(path, delimiter=",")
        self.labels = (data[:, 0].astype(int) - 1)   # remap to 0-indexed
        self.series = data[:, 1:].astype(np.float32)
        # Normalize each series to zero mean, unit std
        m = self.series.mean(axis=1, keepdims=True)
        s = self.series.std(axis=1, keepdims=True) + 1e-8
        self.series = (self.series - m) / s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.series[idx]).unsqueeze(0)  # (1, T)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_ucr_loaders(data_dir: str = "data/ucr/ECG5000",
                    batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    train_path = os.path.join(data_dir, "ECG5000_TRAIN.txt")
    test_path  = os.path.join(data_dir, "ECG5000_TEST.txt")
    train_ds = UCRDataset(train_path)
    test_ds  = UCRDataset(test_path)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4))


# ---------------------------------------------------------------------------
# 2D: CIFAR-10
# ---------------------------------------------------------------------------

def get_cifar10_loaders(data_dir: str = "data/cifar10",
                        batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


# ---------------------------------------------------------------------------
# 3D Point Cloud: ModelNet40
# ---------------------------------------------------------------------------

def get_modelnet40_loaders(data_dir: str = "data/modelnet40",
                           batch_size: int = 32,
                           num_points: int = 1024) -> tuple:
    if not PYG_AVAILABLE:
        raise ImportError("pip install torch-geometric")

    pre_transform = T.Compose([SamplePoints(num_points), NormalizeScale()])
    train_ds = ModelNet(data_dir, name="40", train=True,  pre_transform=pre_transform)
    test_ds  = ModelNet(data_dir, name="40", train=False, pre_transform=pre_transform)
    return (PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
            PyGDataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4))


# ---------------------------------------------------------------------------
# 3D Mesh: SHREC16 (MeshCNN format)
# ---------------------------------------------------------------------------

class SHRECMeshDataset(Dataset):
    """
    Loads mesh edge features in MeshCNN's .npz format.

    Expected directory layout:
        data/shrec16/
            train/
                class_000/  mesh_001.npz  mesh_002.npz ...
                class_001/  ...
            test/
                class_000/  ...

    Each .npz contains:
        edge_features:  (E, 5) float32
        neighbor_idx:   (E, 4) int32
        label:          int
    """

    def __init__(self, root: str, split: str = "train"):
        self.samples = []
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"{split_dir} not found. Download SHREC16 from the MeshCNN repo:\n"
                "  https://github.com/ranahanocka/MeshCNN#datasets"
            )
        class_names = sorted(os.listdir(split_dir))
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            for fn in sorted(os.listdir(cls_dir)):
                if fn.endswith(".npz"):
                    self.samples.append((os.path.join(cls_dir, fn), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        x  = torch.tensor(data["edge_features"], dtype=torch.float32)  # (E, 5)
        nb = torch.tensor(data["neighbor_idx"],  dtype=torch.long)      # (E, 4)
        y  = torch.tensor(label, dtype=torch.long)
        return x, nb, y


def collate_mesh(batch):
    """Variable-size mesh collation — returns a list of (x, nb, y) tuples."""
    return batch  # handled per-sample in training loop


def get_shrec_loaders(data_dir: str = "data/shrec16",
                      batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    train_ds = SHRECMeshDataset(data_dir, split="train")
    test_ds  = SHRECMeshDataset(data_dir, split="test")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_mesh),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_mesh))
