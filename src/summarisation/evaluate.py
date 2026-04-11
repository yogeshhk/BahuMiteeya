"""
evaluate.py — GeoConvNet summarisation evaluation script.

Reports three metrics per domain:
  1. Downstream accuracy  — classification accuracy on the summary
  2. Reconstruction quality — MSE (1D/2D/Mesh) or Chamfer distance (3D-PC)
  3. Compression ratio    — |input elements| / |summary elements|

Usage:
  python evaluate.py --domain 1d    --checkpoint checkpoints/best_1d.pt   --data_dir data/ucr/ECG5000
  python evaluate.py --domain 2d    --checkpoint checkpoints/best_2d.pt   --data_dir data/cifar10
  python evaluate.py --domain 3dpc  --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40
  python evaluate.py --domain mesh  --checkpoint checkpoints/best_mesh.pt  --data_dir data/shrec16
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from datasets.loaders import (
    get_ucr_loaders, get_cifar10_loaders,
    get_modelnet40_loaders, get_shrec_loaders,
)
from models.conv1d_summariser    import GeoConvNet1DSummariser
from models.conv2d_summariser    import GeoConvNet2DSummariser
from models.pointnet2_summariser import GeoConvNet3DPCSummariser
from models.meshcnn_summariser   import GeoConvNet3DMeshSummariser


# ---------------------------------------------------------------------------
# Chamfer distance (same as train.py)
# ---------------------------------------------------------------------------

def chamfer_distance(recon: torch.Tensor, orig: torch.Tensor,
                     batch_recon: torch.Tensor,
                     batch_orig: torch.Tensor) -> float:
    B = int(batch_orig.max().item()) + 1
    total = 0.0
    for b in range(B):
        r = recon[batch_recon == b]
        o = orig[batch_orig   == b]
        if r.size(0) == 0 or o.size(0) == 0:
            continue
        rr    = (r ** 2).sum(1, keepdim=True)
        oo    = (o ** 2).sum(1, keepdim=True)
        dist2 = rr + oo.T - 2.0 * (r @ o.T)
        total += (dist2.min(1).values.mean() + dist2.min(0).values.mean()).item()
    return total / B


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_standard(model, loader, device):
    """Evaluate 1D and 2D summarisers."""
    model.eval()
    total_acc, total_mse, total_ratio, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        summary, recon, logits = model(x)

        acc   = (logits.argmax(dim=1) == y).float().mean().item()
        mse   = F.mse_loss(recon, x).item()
        # Compression ratio: input elements / summary elements
        ratio = float(x[0].numel()) / float(summary[0].numel())

        B = y.size(0)
        total_acc   += acc   * B
        total_mse   += mse   * B
        total_ratio += ratio * B
        n += B

    return total_acc / n, total_mse / n, total_ratio / n


@torch.no_grad()
def evaluate_pyg(model, loader, device):
    """Evaluate 3D point cloud summariser."""
    model.eval()
    total_acc, total_chamfer, total_ratio, n = 0.0, 0.0, 0.0, 0
    for data in loader:
        data = data.to(device)
        summary_pos, recon_pos, logits = model(data)
        y = data.y.view(-1)

        acc     = (logits.argmax(dim=1) == y).float().mean().item()
        chamfer = chamfer_distance(recon_pos, data.pos, data.batch, data.batch)
        # Per-sample compression: count summary vs. original points
        B = y.size(0)
        n_orig    = data.pos.size(0) / B
        n_summary = summary_pos.size(0) / B
        ratio = n_orig / max(n_summary, 1)

        total_acc     += acc     * B
        total_chamfer += chamfer * B
        total_ratio   += ratio   * B
        n += B

    return total_acc / n, total_chamfer / n, total_ratio / n


@torch.no_grad()
def evaluate_mesh(model, loader, device):
    """Evaluate 3D mesh summariser."""
    model.eval()
    total_acc, total_mse, total_ratio, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        for (x, nb, y) in batch:
            x, nb, y = x.to(device), nb.to(device), y.to(device).unsqueeze(0)
            summary_feats, recon_feats, logits = model(x, nb)

            acc   = (logits.argmax(dim=1) == y).float().item()
            mse   = F.mse_loss(recon_feats, x).item()
            ratio = x.size(0) / max(summary_feats.size(0), 1)

            total_acc   += acc
            total_mse   += mse
            total_ratio += ratio
            n += 1

    return total_acc / n, total_mse / n, total_ratio / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GeoConvNet summarisation evaluator")
    parser.add_argument("--domain",     type=str, required=True,
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--batch",      type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GeoConvNet-Summarise] Evaluating domain={args.domain} | device={device}")

    # ------------------------------------------------------------------
    # Load model & data
    # ------------------------------------------------------------------
    if args.domain == "1d":
        model  = GeoConvNet1DSummariser(in_channels=1, num_classes=5).to(device)
        _, loader = get_ucr_loaders(args.data_dir, args.batch)
        evaluate  = evaluate_standard
        recon_metric_name = "MSE"

    elif args.domain == "2d":
        model  = GeoConvNet2DSummariser(num_classes=10).to(device)
        _, loader = get_cifar10_loaders(args.data_dir, args.batch)
        evaluate  = evaluate_standard
        recon_metric_name = "MSE"

    elif args.domain == "3dpc":
        model  = GeoConvNet3DPCSummariser(num_classes=40).to(device)
        _, loader = get_modelnet40_loaders(args.data_dir, args.batch)
        evaluate  = evaluate_pyg
        recon_metric_name = "Chamfer"

    elif args.domain == "mesh":
        model  = GeoConvNet3DMeshSummariser(num_classes=30).to(device)
        _, loader = get_shrec_loaders(args.data_dir, batch_size=1)
        evaluate  = evaluate_mesh
        recon_metric_name = "MSE"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(train acc={ckpt.get('acc', 0):.4f})")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    acc, recon_val, ratio = evaluate(model, loader, device)

    print(f"\n{'─'*50}")
    print(f"  Domain:               {args.domain.upper()}")
    print(f"  Downstream accuracy:  {acc * 100:.2f}%")
    print(f"  Reconstruction {recon_metric_name:7s}: {recon_val:.6f}")
    print(f"  Compression ratio:    {ratio:.1f}x")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
