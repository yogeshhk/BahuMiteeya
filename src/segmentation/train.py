"""
train.py — Unified GeoConvNet segmentation training script.

All domains use per-element cross-entropy loss and track mIoU.

Usage:
  python train.py --domain 1d    --data_dir data/ucr/ECG5000    --num_classes 6
  python train.py --domain 2d    --data_dir data/voc             --num_classes 21
  python train.py --domain 3dpc  --data_dir data/shapenet        --num_classes 50
  python train.py --domain mesh  --data_dir data/coseg/vases     --num_classes 4
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from datasets.loaders import (
    get_ucr_motif_loaders,
    get_voc_loaders,
    get_shapenetpart_loaders,
    get_coseg_loaders,
)
from models.conv1d_segmenter    import GeoConvNet1D
from models.conv2d_segmenter    import GeoConvNet2D
from models.pointnet2_segmenter import GeoConvNet3DPCSeg
from models.meshcnn_segmenter   import GeoConvNet3DMeshSeg


# ---------------------------------------------------------------------------
# mIoU metric (domain-agnostic: works on any flat array of pred/label)
# ---------------------------------------------------------------------------

def compute_miou(preds: torch.Tensor, labels: torch.Tensor,
                 num_classes: int, ignore_index: int = -1) -> float:
    """
    Compute mean Intersection-over-Union over all valid classes.

    Args:
        preds:        (N,) flat predicted label tensor.
        labels:       (N,) flat ground-truth label tensor.
        num_classes:  Total number of classes.
        ignore_index: Label value to ignore (e.g., VOC void=255 mapped to -1).
    Returns:
        mIoU as a float in [0,1]. Returns 0.0 if no valid predictions exist
        (e.g. all labels are ignore_index), which can appear as unexpectedly
        low mIoU in early training epochs with small batches.
    """
    valid = labels != ignore_index
    preds, labels = preds[valid], labels[valid]
    iou_list = []
    for c in range(num_classes):
        pred_c  = preds  == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum().item()
        union        = (pred_c | label_c).sum().item()
        if union > 0:
            # Only include classes that appear in predictions OR ground truth;
            # absent classes are excluded from the mean (not treated as IoU=0).
            iou_list.append(intersection / union)
    return float(np.mean(iou_list)) if iou_list else 0.0


# ---------------------------------------------------------------------------
# Per-domain epoch runners
# ---------------------------------------------------------------------------

def run_epoch_1d(model, loader, criterion, optimizer, device, train,
                 num_classes):
    model.train() if train else model.eval()
    total_loss, total_miou, n, n_elems = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            # x: (B, 1, T)   y: (B, T)
            x, y = x.to(device), y.to(device)
            logits = model(x)                  # (B, K, T)
            loss   = criterion(logits, y)       # CE over all (B*T) timesteps
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1).view(-1).cpu()
            labs  = y.view(-1).cpu()
            total_loss  += loss.item() * y.numel()   # weight by B*T elements
            total_miou  += compute_miou(preds, labs, num_classes) * y.size(0)
            n       += y.size(0)
            n_elems += y.numel()
    # Divide loss by total elements (B*T) so reported value is per-token CE,
    # consistent with run_epoch_2d / run_epoch_3dpc.
    return total_loss / max(n_elems, 1), total_miou / max(n, 1)


def run_epoch_2d(model, loader, criterion, optimizer, device, train,
                 num_classes):
    model.train() if train else model.eval()
    total_loss, total_miou, n, n_elems = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            # x: (B, 3, H, W)   y: (B, H, W) with -1 for ignore
            x, y = x.to(device), y.to(device)
            logits = model(x)                  # (B, K, H, W)
            loss   = criterion(logits, y)       # CE ignores -1
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1).view(-1).cpu()
            labs  = y.view(-1).cpu()
            total_loss  += loss.item() * y.numel()   # weight by B*H*W pixels
            total_miou  += compute_miou(preds, labs, num_classes,
                                        ignore_index=-1) * y.size(0)
            n       += y.size(0)
            n_elems += y.numel()
    # Divide loss by total pixels so reported value is per-pixel CE,
    # consistent with run_epoch_1d (per-timestep) and run_epoch_3dpc (per-point).
    return total_loss / max(n_elems, 1), total_miou / max(n, 1)


def run_epoch_3dpc(model, loader, criterion, optimizer, device, train,
                   num_classes):
    model.train() if train else model.eval()
    total_loss, total_miou, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for data in loader:
            data  = data.to(device)
            logits = model(data)                # (N_total, K)
            y      = data.y.view(-1)            # (N_total,)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1).cpu()
            labs  = y.cpu()
            # Per-shape mIoU (average over shapes in batch)
            batch = data.batch.cpu()
            B = batch.max().item() + 1
            shape_mious = []
            for b in range(B):
                mask = batch == b
                shape_mious.append(
                    compute_miou(preds[mask], labs[mask], num_classes))
            total_loss  += loss.item() * B
            total_miou  += float(np.mean(shape_mious)) * B
            n += B
    return total_loss / max(n, 1), total_miou / max(n, 1)


def run_epoch_mesh(model, loader, criterion, optimizer, device, train,
                   num_classes):
    model.train() if train else model.eval()
    total_loss, total_miou, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            for (x, nb, y) in batch:
                # x: (E,5)  nb: (E,4)  y: (E,)
                x, nb, y = x.to(device), nb.to(device), y.to(device)
                logits = model(x, nb)           # (E, K)
                loss   = criterion(logits, y)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                preds = logits.argmax(dim=1).cpu()
                total_loss  += loss.item()
                total_miou  += compute_miou(preds, y.cpu(), num_classes)
                n += 1
    return total_loss / max(n, 1), total_miou / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoConvNet unified segmentation trainer")
    parser.add_argument("--domain",      type=str, required=True,
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--data_dir",    type=str, default="data")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override number of classes (auto-set per domain if not given)")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch",       type=int, default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--out_dir",     type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Domain-specific setup
    # ------------------------------------------------------------------
    if args.domain == "1d":
        nc = args.num_classes or 6           # 5 motif classes + background
        model        = GeoConvNet1D(in_channels=1, num_classes=nc).to(device)
        train_loader, val_loader = get_ucr_motif_loaders(args.data_dir, args.batch)
        criterion    = nn.CrossEntropyLoss()  # y shape: (B, T)
        run_epoch    = lambda *a: run_epoch_1d(*a, num_classes=nc)
        n_epochs     = args.epochs

    elif args.domain == "2d":
        nc = args.num_classes or 21          # PASCAL VOC: 21 classes
        model        = GeoConvNet2D(num_classes=nc).to(device)
        train_loader, val_loader = get_voc_loaders(args.data_dir, args.batch)
        criterion    = nn.CrossEntropyLoss(ignore_index=-1)
        run_epoch    = lambda *a: run_epoch_2d(*a, num_classes=nc)
        n_epochs     = args.epochs

    elif args.domain == "3dpc":
        nc = args.num_classes or 50          # ShapeNetPart: 50 part labels
        model        = GeoConvNet3DPCSeg(num_classes=nc).to(device)
        train_loader, val_loader = get_shapenetpart_loaders(args.data_dir, args.batch)
        criterion    = nn.CrossEntropyLoss()
        run_epoch    = lambda *a: run_epoch_3dpc(*a, num_classes=nc)
        n_epochs     = args.epochs * 2       # 3D needs more epochs

    elif args.domain == "mesh":
        nc = args.num_classes or 4           # COSEG vases: 4 parts
        model        = GeoConvNet3DMeshSeg(num_classes=nc).to(device)
        train_loader, val_loader = get_coseg_loaders(args.data_dir, batch_size=1)
        criterion    = nn.CrossEntropyLoss()
        run_epoch    = lambda *a: run_epoch_mesh(*a, num_classes=nc)
        n_epochs     = args.epochs * 2

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"[GeoConvNet] domain={args.domain} | classes={nc} | device={device}")
    print(f"             epochs={n_epochs} | lr={args.lr} | batch={args.batch}")

    best_miou = 0.0
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_miou = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_miou = run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train_loss={tr_loss:.4f}  train_mIoU={tr_miou:.4f}  "
              f"val_loss={va_loss:.4f}  val_mIoU={va_miou:.4f}")

        if va_miou > best_miou:
            best_miou = va_miou
            path = os.path.join(args.out_dir, f"best_{args.domain}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "miou": best_miou}, path)
            print(f"  → Saved best checkpoint (mIoU={best_miou:.4f}) → {path}")

    print(f"\n[Done] Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
