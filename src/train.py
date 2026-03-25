"""
train.py — Unified GeoConvNet training script for all four domains.

Usage:
  python train.py --domain 1d    --data_dir data/ucr/ECG5000
  python train.py --domain 2d    --data_dir data/cifar10
  python train.py --domain 3dpc  --data_dir data/modelnet40
  python train.py --domain mesh  --data_dir data/shrec16
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.loaders import (
    get_ucr_loaders, get_cifar10_loaders,
    get_modelnet40_loaders, get_shrec_loaders,
)
from models.conv1d_classifier   import GeoConvNet1D
from models.conv2d_classifier   import GeoConvNet2D
from models.pointnet2_classifier import GeoConvNet3DPC
from models.meshcnn_classifier  import GeoConvNet3DMesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch_standard(model, loader, criterion, optimizer, device, train: bool):
    """1D and 2D training loop (standard PyTorch DataLoader)."""
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_acc  += accuracy(logits, y) * y.size(0)
            n += y.size(0)
    return total_loss / n, total_acc / n


def run_epoch_pyg(model, loader, criterion, optimizer, device, train: bool):
    """3D point cloud training loop (PyG DataLoader)."""
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for data in loader:
            data = data.to(device)
            logits = model(data)
            y = data.y.view(-1)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_acc  += accuracy(logits, y) * y.size(0)
            n += y.size(0)
    return total_loss / n, total_acc / n


def run_epoch_mesh(model, loader, criterion, optimizer, device, train: bool):
    """3D mesh training loop (per-sample, variable topology)."""
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            for (x, nb, y) in batch:
                x, nb, y = x.to(device), nb.to(device), y.to(device).unsqueeze(0)
                logits = model(x, nb)         # (1, num_classes)
                loss = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                total_acc  += (logits.argmax(dim=1) == y).float().item()
                n += 1
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoConvNet unified trainer")
    parser.add_argument("--domain",   type=str, default="2d",
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--out_dir",  type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GeoConvNet] domain={args.domain} | device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model & data loaders
    # ------------------------------------------------------------------
    if args.domain == "1d":
        model      = GeoConvNet1D(in_channels=1, num_classes=5).to(device)
        train_loader, test_loader = get_ucr_loaders(args.data_dir, args.batch)
        run_epoch  = run_epoch_standard
        n_epochs   = args.epochs

    elif args.domain == "2d":
        model      = GeoConvNet2D(num_classes=10).to(device)
        train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch)
        run_epoch  = run_epoch_standard
        n_epochs   = args.epochs

    elif args.domain == "3dpc":
        model      = GeoConvNet3DPC(num_classes=40).to(device)
        train_loader, test_loader = get_modelnet40_loaders(args.data_dir, args.batch)
        run_epoch  = run_epoch_pyg
        n_epochs   = args.epochs * 2  # 3D models typically need more epochs

    elif args.domain == "mesh":
        model      = GeoConvNet3DMesh(num_classes=30).to(device)
        train_loader, test_loader = get_shrec_loaders(args.data_dir, batch_size=1)
        run_epoch  = run_epoch_mesh
        n_epochs   = args.epochs * 2

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, test_loader,  criterion, None,      device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt_path = os.path.join(args.out_dir, f"best_{args.domain}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": best_acc}, ckpt_path)
            print(f"  → Saved best checkpoint (acc={best_acc:.4f}) to {ckpt_path}")

    print(f"\n[Done] Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
