"""
train.py — Unified GeoConvNet summarisation training script.

Each domain's summariser is trained with a combined loss:
  L = alpha * CrossEntropy(logits, y)   [downstream task preservation]
    + (1-alpha) * ReconLoss(recon, x)   [reconstruction fidelity]

ReconLoss per domain:
  1D / 2D / Mesh: MSELoss on signal/image/edge features
  3D-PC:          Symmetric Chamfer distance on point positions

Usage:
  python train.py --domain 1d    --data_dir data/ucr/ECG5000
  python train.py --domain 2d    --data_dir data/cifar10 --batch 128
  python train.py --domain 3dpc  --data_dir data/modelnet40 --epochs 200
  python train.py --domain mesh  --data_dir data/shrec16
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.loaders import (
    get_ucr_loaders, get_cifar10_loaders,
    get_modelnet40_loaders, get_shrec_loaders,
)
from models.conv1d_summariser    import GeoConvNet1DSummariser
from models.conv2d_summariser    import GeoConvNet2DSummariser
from models.pointnet2_summariser import GeoConvNet3DPCSummariser
from models.meshcnn_summariser   import GeoConvNet3DMeshSummariser


# ---------------------------------------------------------------------------
# Chamfer distance (used for 3D-PC reconstruction loss)
# ---------------------------------------------------------------------------

def chamfer_distance(recon: torch.Tensor, orig: torch.Tensor,
                     batch_recon: torch.Tensor,
                     batch_orig: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-batch Chamfer distance.

    Args:
        recon:        (N_total, 3) reconstructed positions.
        orig:         (N_total, 3) original positions.
        batch_recon:  (N_total,)   batch index for reconstructed points.
        batch_orig:   (N_total,)   batch index for original points.
    Returns:
        Scalar mean Chamfer distance.
    """
    B = int(batch_orig.max().item()) + 1
    total = recon.new_zeros(1)
    for b in range(B):
        r = recon[batch_recon == b]
        o = orig[batch_orig   == b]
        if r.size(0) == 0 or o.size(0) == 0:
            continue
        rr    = (r ** 2).sum(1, keepdim=True)
        oo    = (o ** 2).sum(1, keepdim=True)
        dist2 = rr + oo.T - 2.0 * (r @ o.T)
        total = total + dist2.min(1).values.mean() + dist2.min(0).values.mean()
    return total / B


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Per-domain epoch runners
# ---------------------------------------------------------------------------

def run_epoch_standard(model, loader, optimizer, device, train: bool,
                       alpha: float):
    """1D and 2D training loop."""
    model.train() if train else model.eval()
    total_cls, total_recon, total_acc, n = 0.0, 0.0, 0.0, 0
    criterion_cls = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            summary, recon, logits = model(x)

            loss_cls   = criterion_cls(logits, y)
            loss_recon = F.mse_loss(recon, x)
            loss       = alpha * loss_cls + (1.0 - alpha) * loss_recon

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = y.size(0)
            total_cls   += loss_cls.item()   * B
            total_recon += loss_recon.item() * B
            total_acc   += accuracy(logits, y) * B
            n += B

    return total_cls / n, total_recon / n, total_acc / n


def run_epoch_pyg(model, loader, optimizer, device, train: bool,
                  alpha: float):
    """3D point cloud training loop (PyG DataLoader)."""
    model.train() if train else model.eval()
    total_cls, total_recon, total_acc, n = 0.0, 0.0, 0.0, 0
    criterion_cls = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for data in loader:
            data = data.to(device)
            orig_pos   = data.pos          # (N_total, 3) original positions
            orig_batch = data.batch        # (N_total,)

            summary_pos, recon_pos, logits = model(data)
            y = data.y.view(-1)

            loss_cls   = criterion_cls(logits, y)
            # Chamfer distance: reconstruction vs. original point positions
            loss_recon = chamfer_distance(
                recon_pos, orig_pos,
                orig_batch,   # reconstruction has same N as original
                orig_batch,
            )
            loss = alpha * loss_cls + (1.0 - alpha) * loss_recon

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = y.size(0)
            total_cls   += loss_cls.item()   * B
            total_recon += loss_recon.item() * B
            total_acc   += accuracy(logits, y) * B
            n += B

    return total_cls / n, total_recon / n, total_acc / n


def run_epoch_mesh(model, loader, optimizer, device, train: bool,
                   alpha: float):
    """3D mesh training loop (per-sample, variable topology)."""
    model.train() if train else model.eval()
    total_cls, total_recon, total_acc, n = 0.0, 0.0, 0.0, 0
    criterion_cls = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for batch in loader:
            for (x, nb, y) in batch:
                x, nb, y = x.to(device), nb.to(device), y.to(device).unsqueeze(0)

                summary_feats, recon_feats, logits = model(x, nb)

                loss_cls   = criterion_cls(logits, y)
                loss_recon = F.mse_loss(recon_feats, x)
                loss       = alpha * loss_cls + (1.0 - alpha) * loss_recon

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_cls   += loss_cls.item()
                total_recon += loss_recon.item()
                total_acc   += (logits.argmax(dim=1) == y).float().item()
                n += 1

    return total_cls / n, total_recon / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GeoConvNet unified summarisation trainer")
    parser.add_argument("--domain",   type=str, required=True,
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--alpha",    type=float, default=0.5,
                        help="Weight for classification loss "
                             "(1-alpha weights reconstruction loss)")
    parser.add_argument("--out_dir",  type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[GeoConvNet-Summarise] domain={args.domain} | "
          f"alpha={args.alpha} | device={device}")

    # ------------------------------------------------------------------
    # Build model & data loaders
    # ------------------------------------------------------------------
    if args.domain == "1d":
        model        = GeoConvNet1DSummariser(in_channels=1, num_classes=5).to(device)
        train_loader, test_loader = get_ucr_loaders(args.data_dir, args.batch)
        run_epoch    = run_epoch_standard
        n_epochs     = args.epochs

    elif args.domain == "2d":
        model        = GeoConvNet2DSummariser(num_classes=10).to(device)
        train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch)
        run_epoch    = run_epoch_standard
        n_epochs     = args.epochs

    elif args.domain == "3dpc":
        model        = GeoConvNet3DPCSummariser(num_classes=40).to(device)
        train_loader, test_loader = get_modelnet40_loaders(args.data_dir, args.batch)
        run_epoch    = run_epoch_pyg
        n_epochs     = args.epochs * 2

    elif args.domain == "mesh":
        model        = GeoConvNet3DMeshSummariser(num_classes=30).to(device)
        train_loader, test_loader = get_shrec_loaders(args.data_dir, batch_size=1)
        run_epoch    = run_epoch_mesh
        n_epochs     = args.epochs * 2

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        tr_cls, tr_recon, tr_acc = run_epoch(
            model, train_loader, optimizer, device, True,  args.alpha)
        va_cls, va_recon, va_acc = run_epoch(
            model, test_loader,  None,      device, False, args.alpha)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train_cls={tr_cls:.4f}  train_recon={tr_recon:.4f}  "
              f"train_acc={tr_acc:.4f}  "
              f"val_cls={va_cls:.4f}  val_recon={va_recon:.4f}  "
              f"val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = os.path.join(args.out_dir, f"best_{args.domain}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "acc": best_acc}, ckpt)
            print(f"  → Saved best checkpoint (acc={best_acc:.4f}) → {ckpt}")

    print(f"\n[Done] Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
