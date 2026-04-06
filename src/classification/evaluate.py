"""
evaluate.py — Load a trained GeoConvNet checkpoint and report metrics.
Optionally compares against a standalone PointNet++ checkpoint for the 3D-PC domain.

Usage:
  python evaluate.py --domain 1d   --checkpoint checkpoints/best_1d.pt   --data_dir data/ucr/ECG5000
  python evaluate.py --domain 2d   --checkpoint checkpoints/best_2d.pt   --data_dir data/cifar10
  python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40
  python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/modelnet40 \\
                     --compare_pointnet2 checkpoints/pointnet2_standalone.pt
  python evaluate.py --domain mesh --checkpoint checkpoints/best_mesh.pt  --data_dir data/shrec16
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from datasets.loaders import (
    get_ucr_loaders, get_cifar10_loaders,
    get_modelnet40_loaders, get_shrec_loaders,
)
from models.conv1d_classifier    import GeoConvNet1D
from models.conv2d_classifier    import GeoConvNet2D
from models.pointnet2_classifier import GeoConvNet3DPC
from models.meshcnn_classifier   import GeoConvNet3DMesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions_standard(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def collect_predictions_pyg(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        preds = model(data).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(data.y.view(-1).cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def collect_predictions_mesh(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        for (x, nb, y) in batch:
            x, nb = x.to(device), nb.to(device)
            pred = model(x, nb).argmax(dim=1).cpu().item()
            all_preds.append(pred)
            all_labels.append(y.item())
    return np.array(all_preds), np.array(all_labels)


def print_metrics(preds, labels, domain_name: str):
    oa = (preds == labels).mean()
    print(f"\n{'='*60}")
    print(f"Domain: {domain_name}")
    print(f"Overall Accuracy: {oa:.4f} ({oa*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, digits=4))


# ---------------------------------------------------------------------------
# Standalone PointNet++ baseline comparison
# ---------------------------------------------------------------------------

def evaluate_pointnet2_baseline(data_dir: str, ckpt_path: str, device: torch.device) -> float:
    """
    Evaluates a separately trained PointNet++ checkpoint for comparison.
    The checkpoint should be trained with the same ModelNet40 data split.
    """
    print("\n[Baseline] Evaluating standalone PointNet++ checkpoint...")
    model = GeoConvNet3DPC(num_classes=40).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    _, test_loader = get_modelnet40_loaders(data_dir, batch_size=32)
    preds, labels = collect_predictions_pyg(model, test_loader, device)
    oa = (preds == labels).mean()
    print(f"[PointNet++ baseline] Overall Accuracy: {oa:.4f}")
    return oa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoConvNet evaluator")
    parser.add_argument("--domain",          type=str, required=True,
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--data_dir",        type=str, default="data")
    parser.add_argument("--compare_pointnet2", type=str, default=None,
                        help="Path to standalone PointNet++ checkpoint for comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    if args.domain == "1d":
        model = GeoConvNet1D(in_channels=1, num_classes=5).to(device)
        model.load_state_dict(ckpt["model"])
        _, test_loader = get_ucr_loaders(args.data_dir)
        preds, labels  = collect_predictions_standard(model, test_loader, device)
        print_metrics(preds, labels, "1D (UCR ECG5000)")

    elif args.domain == "2d":
        model = GeoConvNet2D(num_classes=10).to(device)
        model.load_state_dict(ckpt["model"])
        _, test_loader = get_cifar10_loaders(args.data_dir)
        preds, labels  = collect_predictions_standard(model, test_loader, device)
        print_metrics(preds, labels, "2D (CIFAR-10)")

    elif args.domain == "3dpc":
        model = GeoConvNet3DPC(num_classes=40).to(device)
        model.load_state_dict(ckpt["model"])
        _, test_loader = get_modelnet40_loaders(args.data_dir)
        preds, labels  = collect_predictions_pyg(model, test_loader, device)
        print_metrics(preds, labels, "3D-PC (ModelNet40)")

        if args.compare_pointnet2:
            evaluate_pointnet2_baseline(args.data_dir, args.compare_pointnet2, device)

    elif args.domain == "mesh":
        model = GeoConvNet3DMesh(num_classes=30).to(device)
        model.load_state_dict(ckpt["model"])
        _, test_loader = get_shrec_loaders(args.data_dir)
        preds, labels  = collect_predictions_mesh(model, test_loader, device)
        print_metrics(preds, labels, "3D-Mesh (SHREC16)")


if __name__ == "__main__":
    main()
