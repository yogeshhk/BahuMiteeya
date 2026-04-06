"""
evaluate.py — Evaluate a trained GeoConvNet segmentation checkpoint.
Reports per-element mIoU and per-class IoU breakdown.
Optionally compares 3D-PC against a standalone PointNet++ checkpoint.

Usage:
  python evaluate.py --domain 1d   --checkpoint checkpoints/best_1d.pt   --data_dir data/ucr/ECG5000  --num_classes 6
  python evaluate.py --domain 2d   --checkpoint checkpoints/best_2d.pt   --data_dir data/voc           --num_classes 21
  python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/shapenet      --num_classes 50
  python evaluate.py --domain 3dpc --checkpoint checkpoints/best_3dpc.pt --data_dir data/shapenet      --num_classes 50 \\
                     --compare_pointnet2 checkpoints/pointnet2_standalone.pt
  python evaluate.py --domain mesh --checkpoint checkpoints/best_mesh.pt  --data_dir data/coseg/vases  --num_classes 4
"""

import argparse
import torch
import numpy as np

from datasets.loaders import (
    get_ucr_motif_loaders, get_voc_loaders,
    get_shapenetpart_loaders, get_coseg_loaders,
)
from models.conv1d_segmenter    import GeoConvNet1D
from models.conv2d_segmenter    import GeoConvNet2D
from models.pointnet2_segmenter import GeoConvNet3DPCSeg
from models.meshcnn_segmenter   import GeoConvNet3DMeshSeg


# ---------------------------------------------------------------------------
# Per-class IoU
# ---------------------------------------------------------------------------

def per_class_iou(all_preds: np.ndarray, all_labels: np.ndarray,
                  num_classes: int, ignore_index: int = -1):
    """Returns (per_class_iou array, mean_iou float)."""
    valid = all_labels != ignore_index
    p, l  = all_preds[valid], all_labels[valid]
    ious  = []
    for c in range(num_classes):
        inter = ((p == c) & (l == c)).sum()
        union = ((p == c) | (l == c)).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    valid_ious = [v for v in ious if not np.isnan(v)]
    return np.array(ious), float(np.mean(valid_ious)) if valid_ious else 0.0


def print_iou_report(domain: str, ious: np.ndarray, miou: float,
                     class_names: list = None):
    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"Mean IoU: {miou:.4f}  ({miou*100:.2f}%)")
    print(f"\nPer-class IoU:")
    for i, iou in enumerate(ious):
        name = class_names[i] if class_names else f"class_{i:02d}"
        val  = f"{iou*100:.2f}%" if not np.isnan(iou) else "  n/a  "
        print(f"  {name:20s}: {val}")


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_1d(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for x, y in loader:
        logits = model(x.to(device))  # (B, K, T)
        all_p.append(logits.argmax(1).view(-1).cpu().numpy())
        all_l.append(y.view(-1).numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


@torch.no_grad()
def collect_2d(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for x, y in loader:
        logits = model(x.to(device))  # (B, K, H, W)
        all_p.append(logits.argmax(1).view(-1).cpu().numpy())
        all_l.append(y.view(-1).numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


@torch.no_grad()
def collect_3dpc(model, loader, device):
    """Returns instance-averaged mIoU and flat preds/labels."""
    model.eval()
    all_p, all_l = [], []
    shape_mious  = []
    for data in loader:
        data   = data.to(device)
        logits = model(data)              # (N_total, K)
        preds  = logits.argmax(1).cpu()
        labels = data.y.view(-1).cpu()
        batch  = data.batch.cpu()
        B = batch.max().item() + 1
        for b in range(B):
            m = batch == b
            _, iou = per_class_iou(preds[m].numpy(), labels[m].numpy(),
                                   num_classes=logits.size(1))
            shape_mious.append(iou)
        all_p.append(preds.numpy())
        all_l.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_l), float(np.mean(shape_mious))


@torch.no_grad()
def collect_mesh(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        for (x, nb, y) in batch:
            logits = model(x.to(device), nb.to(device))
            all_p.append(logits.argmax(1).cpu().numpy())
            all_l.append(y.numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoConvNet segmentation evaluator")
    parser.add_argument("--domain",           required=True,
                        choices=["1d", "2d", "3dpc", "mesh"])
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--data_dir",         default="data")
    parser.add_argument("--num_classes",      type=int, default=None)
    parser.add_argument("--compare_pointnet2", default=None,
                        help="Path to standalone PointNet++ checkpoint (3dpc only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=True)

    if args.domain == "1d":
        nc    = args.num_classes or 6
        model = GeoConvNet1D(in_channels=1, num_classes=nc).to(device)
        model.load_state_dict(ckpt["model"])
        _, loader = get_ucr_motif_loaders(args.data_dir)
        p, l = collect_1d(model, loader, device)
        ious, miou = per_class_iou(p, l, nc)
        print_iou_report("1D — Motif Segmentation (UCR ECG5000)", ious, miou)

    elif args.domain == "2d":
        nc    = args.num_classes or 21
        model = GeoConvNet2D(num_classes=nc).to(device)
        model.load_state_dict(ckpt["model"])
        _, loader = get_voc_loaders(args.data_dir)
        p, l = collect_2d(model, loader, device)
        ious, miou = per_class_iou(p, l, nc, ignore_index=-1)
        voc_classes = ["background","aeroplane","bicycle","bird","boat","bottle",
                       "bus","car","cat","chair","cow","diningtable","dog","horse",
                       "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        print_iou_report("2D — Semantic Segmentation (PASCAL VOC 2012)",
                         ious, miou, voc_classes)

    elif args.domain == "3dpc":
        nc    = args.num_classes or 50
        model = GeoConvNet3DPCSeg(num_classes=nc).to(device)
        model.load_state_dict(ckpt["model"])
        _, loader = get_shapenetpart_loaders(args.data_dir)
        p, l, instance_miou = collect_3dpc(model, loader, device)
        ious, cat_miou = per_class_iou(p, l, nc)
        print_iou_report("3D-PC — Part Segmentation (ShapeNetPart)", ious, cat_miou)
        print(f"\n  Instance-averaged mIoU: {instance_miou:.4f} ({instance_miou*100:.2f}%)")

        if args.compare_pointnet2:
            print(f"\n[Baseline] Loading standalone PointNet++ from {args.compare_pointnet2}")
            base  = GeoConvNet3DPCSeg(num_classes=nc).to(device)
            bckpt = torch.load(args.compare_pointnet2, map_location=device, weights_only=True)
            base.load_state_dict(bckpt["model"])
            _, loader2 = get_shapenetpart_loaders(args.data_dir)
            bp, bl, base_miou = collect_3dpc(base, loader2, device)
            _, base_cat_miou  = per_class_iou(bp, bl, nc)
            print(f"  PointNet++ standalone — instance mIoU: {base_miou:.4f}, "
                  f"category mIoU: {base_cat_miou:.4f}")
            print(f"  GeoConvNet-3D-PC      — instance mIoU: {instance_miou:.4f}, "
                  f"category mIoU: {cat_miou:.4f}")
            diff = instance_miou - base_miou
            print(f"  Delta: {diff:+.4f} ({'↑ better' if diff >= 0 else '↓ lower'})")

    elif args.domain == "mesh":
        nc    = args.num_classes or 4
        model = GeoConvNet3DMeshSeg(num_classes=nc).to(device)
        model.load_state_dict(ckpt["model"])
        _, loader = get_coseg_loaders(args.data_dir)
        p, l = collect_mesh(model, loader, device)
        ious, miou = per_class_iou(p, l, nc)
        print_iou_report("3D-Mesh — Part Segmentation (COSEG)", ious, miou)


if __name__ == "__main__":
    main()
