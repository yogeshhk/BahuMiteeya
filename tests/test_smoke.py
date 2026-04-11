"""
tests/test_smoke.py — Forward-pass smoke tests for all 8 GeoConvNet models.

These tests verify that each model:
  - Instantiates without error
  - Accepts the documented input format
  - Produces output of the documented shape
  - Has a non-zero parameter count

Run from the repo root:
  pytest tests/test_smoke.py -v

PyTorch Geometric is required for the 3D point cloud tests.
Skip them with:
  pytest tests/test_smoke.py -v -k "not pyg"
"""

import sys
import os
import pytest
import torch

# Make both sub-project src trees importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "classification"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "segmentation"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------

def test_conv1d_classifier():
    from models.conv1d_classifier import GeoConvNet1D
    model = GeoConvNet1D(in_channels=1, num_classes=5)
    x = torch.randn(4, 1, 140)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4,5), got {out.shape}"
    assert param_count(model) > 0


def test_conv2d_classifier():
    from models.conv2d_classifier import GeoConvNet2D
    model = GeoConvNet2D(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2,10), got {out.shape}"
    assert param_count(model) > 0


@pytest.mark.pyg
def test_pointnet2_classifier():
    try:
        from torch_geometric.data import Data
    except ImportError:
        pytest.skip("torch_geometric not installed")
    from models.pointnet2_classifier import GeoConvNet3DPC
    B, N = 2, 1024
    pos   = torch.randn(B * N, 3)
    batch = torch.repeat_interleave(torch.arange(B), N)
    data  = Data(pos=pos, batch=batch, x=None)
    model = GeoConvNet3DPC(num_classes=40)
    out   = model(data)
    assert out.shape == (B, 40), f"Expected ({B},40), got {out.shape}"
    assert param_count(model) > 0


def test_meshcnn_classifier():
    from models.meshcnn_classifier import GeoConvNet3DMesh
    E  = 2000
    x  = torch.randn(E, 5)
    nb = torch.randint(0, E, (E, 4))
    model = GeoConvNet3DMesh(num_classes=30)
    out   = model(x, nb)
    assert out.shape == (1, 30), f"Expected (1,30), got {out.shape}"
    assert param_count(model) > 0


def test_meshcnn_classifier_small_mesh():
    """Verify MeshPool early-exit (E < target) doesn't crash on CPU or CUDA."""
    from models.meshcnn_classifier import GeoConvNet3DMesh
    # E=200 < first pool target (1500): exercises the early-exit path in all 3 pools
    E  = 200
    x  = torch.randn(E, 5)
    nb = torch.randint(0, E, (E, 4))
    model = GeoConvNet3DMesh(num_classes=30)
    out   = model(x, nb)
    assert out.shape == (1, 30)


# ---------------------------------------------------------------------------
# Segmentation models
# ---------------------------------------------------------------------------

def test_conv1d_segmenter():
    from models.conv1d_segmenter import GeoConvNet1D
    model = GeoConvNet1D(in_channels=1, num_classes=6)
    T   = 140
    x   = torch.randn(4, 1, T)
    out = model(x)
    assert out.shape == (4, 6, T), f"Expected (4,6,{T}), got {out.shape}"
    assert param_count(model) > 0


def test_conv2d_segmenter():
    from models.conv2d_segmenter import GeoConvNet2D
    model = GeoConvNet2D(num_classes=21)
    x   = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 21, 128, 128), f"Expected (2,21,128,128), got {out.shape}"
    assert param_count(model) > 0


@pytest.mark.pyg
def test_pointnet2_segmenter():
    try:
        from torch_geometric.data import Data
    except ImportError:
        pytest.skip("torch_geometric not installed")
    from models.pointnet2_segmenter import GeoConvNet3DPCSeg
    B, N = 2, 512
    pos   = torch.randn(B * N, 3)
    batch = torch.repeat_interleave(torch.arange(B), N)
    data  = Data(pos=pos, batch=batch, x=None)
    model = GeoConvNet3DPCSeg(num_classes=50)
    out   = model(data)
    assert out.shape == (B * N, 50), f"Expected ({B*N},50), got {out.shape}"
    assert param_count(model) > 0


def test_meshcnn_segmenter():
    from models.meshcnn_segmenter import GeoConvNet3DMeshSeg
    E  = 2000
    x  = torch.randn(E, 5)
    nb = torch.randint(0, E, (E, 4))
    model = GeoConvNet3DMeshSeg(num_classes=4)
    out   = model(x, nb)
    assert out.shape == (E, 4), f"Expected ({E},4), got {out.shape}"
    assert param_count(model) > 0


def test_meshcnn_segmenter_device_consistency():
    """MeshPool early-exit must return keep_idx on the same device as input."""
    from models.meshcnn_segmenter import MeshPool
    E = 100  # less than any pool target → exercises early-exit
    x  = torch.randn(E, 64)
    nb = torch.randint(0, E, (E, 4))
    pool = MeshPool(target=1500)
    x_p, nb_p, keep_idx = pool(x, nb)
    assert keep_idx.device == x.device, (
        f"keep_idx on {keep_idx.device}, x on {x.device} — device mismatch"
    )
