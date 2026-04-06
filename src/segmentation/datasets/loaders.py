"""
datasets/loaders.py — Unified dataset loaders for all GeoConvNet segmentation domains.

Domains and datasets:
  1D  : UCR ECG5000 with synthetic per-timestep motif labels
  2D  : PASCAL VOC 2012 semantic segmentation (21 classes)
  3D-PC: ShapeNetPart per-point part segmentation (50 part labels)
  3D-Mesh: COSEG per-edge part segmentation (MeshCNN format)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

try:
    from torch_geometric.datasets import ShapeNet
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1D: UCR ECG5000 with synthetic per-timestep motif segmentation labels
# ---------------------------------------------------------------------------

class UCRMotifDataset(Dataset):
    """
    Loads UCR ECG5000 and synthesizes per-timestep motif segmentation labels.

    Since UCR provides only per-series labels, we construct per-timestep
    labels by running a sliding-window template matching (argmin DTW-distance
    to the per-class centroid) to assign a motif class or background (0)
    to every timestep.

    For research use, replace self._make_seg_labels() with ground-truth
    annotations if available.

    File format: CSV — first column is 1-indexed class label, remainder
    are timestep values.

    Download ECG5000 from: https://timeseriesclassification.com/
    Place files in data/ucr/ECG5000/ECG5000_TRAIN.txt and ECG5000_TEST.txt
    """

    def __init__(self, path: str, window: int = 20):
        data = np.loadtxt(path, delimiter=",")
        labels_series = data[:, 0].astype(int) - 1   # 0-indexed series label
        series = data[:, 1:].astype(np.float32)

        # Normalize
        m = series.mean(axis=1, keepdims=True)
        s = series.std(axis=1, keepdims=True) + 1e-8
        self.series = (series - m) / s

        # Build per-class centroids then assign per-timestep labels
        self.seg_labels = self._make_seg_labels(self.series, labels_series, window)
        self.T = self.series.shape[1]

    @staticmethod
    def _make_seg_labels(series: np.ndarray, cls_labels: np.ndarray,
                         window: int) -> np.ndarray:
        """
        Heuristic per-timestep labeling:
        Compute per-class mean waveform.  Slide a window of `window` steps
        across each series; assign the window center to the class whose
        mean has smallest L2 distance from the window.  Background (label 0)
        is assigned when the distance exceeds a threshold.
        """
        num_classes = int(cls_labels.max()) + 1
        T = series.shape[1]
        centroids = np.stack([
            series[cls_labels == c].mean(axis=0) for c in range(num_classes)
        ])  # (num_classes, T)

        seg = np.zeros((len(series), T), dtype=np.int64)
        thresh = series.std() * 1.5  # background threshold

        for i, (s, c) in enumerate(zip(series, cls_labels)):
            for t in range(T):
                lo, hi = max(0, t - window // 2), min(T, t + window // 2)
                patch = s[lo:hi]
                cpatch = centroids[c, lo:hi]
                dist = np.linalg.norm(patch - cpatch)
                seg[i, t] = c + 1 if dist < thresh else 0  # 0=background

        return seg  # (N, T) with values in {0,...,num_classes}

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x = torch.tensor(self.series[idx]).unsqueeze(0)       # (1, T)
        y = torch.tensor(self.seg_labels[idx], dtype=torch.long)  # (T,)
        return x, y


def get_ucr_motif_loaders(data_dir: str = "data/ucr/ECG5000",
                          batch_size: int = 32) -> tuple:
    train_ds = UCRMotifDataset(os.path.join(data_dir, "ECG5000_TRAIN.txt"))
    test_ds  = UCRMotifDataset(os.path.join(data_dir, "ECG5000_TEST.txt"))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4))


# ---------------------------------------------------------------------------
# 2D: PASCAL VOC 2012 Semantic Segmentation
# ---------------------------------------------------------------------------

class VOCSegDataset(Dataset):
    """
    PASCAL VOC 2012 semantic segmentation.

    torchvision.datasets.VOCSegmentation provides PIL images and segmentation
    masks. We resize to a fixed crop, normalize, and return (image, mask).

    Download: torchvision handles automatic download to data_dir.
    Labels: 0=background, 1-20=object classes, 255=ignore.
    """

    def __init__(self, root: str, split: str = "train",
                 crop_size: int = 480):
        self.base = torchvision.datasets.VOCSegmentation(
            root, year="2012", image_set=split, download=True)
        self.crop_size = crop_size
        self.img_tf = T.Compose([
            T.Resize((crop_size, crop_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        self.mask_tf = T.Compose([
            T.Resize((crop_size, crop_size),
                     interpolation=T.InterpolationMode.NEAREST,
                     antialias=False),  # no antialiasing for discrete label maps
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        img  = self.img_tf(img)
        mask = torch.as_tensor(np.array(self.mask_tf(mask)),
                               dtype=torch.long)
        # Replace VOC ignore index (255) with -1 (PyTorch CE ignore_index)
        mask[mask == 255] = -1
        return img, mask


def get_voc_loaders(data_dir: str = "data/voc",
                    batch_size: int = 8,
                    crop_size: int = 480) -> tuple:
    train_ds = VOCSegDataset(data_dir, split="train",  crop_size=crop_size)
    val_ds   = VOCSegDataset(data_dir, split="val",    crop_size=crop_size)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True))


# ---------------------------------------------------------------------------
# 3D Point Cloud: ShapeNetPart per-point part segmentation
# ---------------------------------------------------------------------------

def get_shapenetpart_loaders(data_dir: str = "data/shapenet",
                             batch_size: int = 16,
                             category: str = None) -> tuple:
    """
    ShapeNetPart via torch_geometric.datasets.ShapeNet.

    category: optional single category (e.g., 'Chair'). None = all 16.
    Each Data object has:
        .pos  (N, 3)  — point coordinates
        .y    (N,)    — per-point part label (0-49 global, remapped per category)
        .batch (N,)   — batch index
    """
    if not PYG_AVAILABLE:
        raise ImportError("pip install torch-geometric")

    train_ds = ShapeNet(data_dir, split="trainval", categories=category,
                        include_normals=False)
    test_ds  = ShapeNet(data_dir, split="test",     categories=category,
                        include_normals=False)
    return (PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4),
            PyGDataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=4))


# ---------------------------------------------------------------------------
# 3D Mesh: COSEG per-edge part segmentation (MeshCNN format)
# ---------------------------------------------------------------------------

class COSEGDataset(Dataset):
    """
    COSEG mesh segmentation dataset in MeshCNN .npz format.

    Directory layout (mirrors MeshCNN repo):
        data/coseg/<object_class>/train/  *.npz
        data/coseg/<object_class>/test/   *.npz

    Each .npz contains:
        edge_features: (E, 5)   float32 — 5D intrinsic edge features
        neighbor_idx:  (E, 4)   int32   — 4-neighbor indices
        seg_labels:    (E,)     int32   — per-edge part label

    Download from: https://github.com/ranahanocka/MeshCNN#datasets

    Args:
        root:    Root data directory, e.g. 'data/coseg/vases'.
        split:   'train' or 'test'.
    """

    def __init__(self, root: str, split: str = "train"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"COSEG split directory not found: {split_dir}\n"
                "Download from: https://github.com/ranahanocka/MeshCNN#datasets"
            )
        self.files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir) if f.endswith(".npz")
        ])
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x  = torch.tensor(data["edge_features"], dtype=torch.float32)  # (E, 5)
        nb = torch.tensor(data["neighbor_idx"],  dtype=torch.long)      # (E, 4)
        y  = torch.tensor(data["seg_labels"],    dtype=torch.long)      # (E,)
        return x, nb, y


def collate_mesh(batch):
    """Variable-topology collation — return list, handle per-sample in loop."""
    return batch


def get_coseg_loaders(data_dir: str = "data/coseg/vases",
                      batch_size: int = 1) -> tuple:
    train_ds = COSEGDataset(data_dir, split="train")
    test_ds  = COSEGDataset(data_dir, split="test")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_mesh),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                       collate_fn=collate_mesh))
