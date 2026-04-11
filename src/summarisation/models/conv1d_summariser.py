"""
GeoConvNet-1D: Convolutional Summariser for 1D Time Series.

Task: produce a shorter representative time series (summary) from a longer
input, preserving semantic content while reducing temporal resolution.

Theoretical basis:
  1D translation-equivariant convolution (G = (Z,+)) with strided
  downsampling produces a coarser representation that summarises the
  original signal at a lower temporal resolution.  The summariser learns
  to compress a T-step signal into a T/compression-step summary while:
    (a) preserving class-discriminative information — measured by downstream
        classification accuracy on the summary (cross-entropy loss), and
    (b) retaining enough information to approximately reconstruct the
        original — measured by MSE reconstruction loss.

Dataset: UCR ECG5000 (T=140 timesteps, 5 classes)
Default: 4x compression → T_out = 35 timesteps

Training objective:
  L = alpha * CrossEntropy(logits, y) + (1-alpha) * MSE(recon, x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EncoderStage1D(nn.Module):
    """Stride-2 downsampling conv + dilated residual block."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        pad = dilation
        self.res = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.act(x + self.res(x))


class DecoderStage1D(nn.Module):
    """Stride-2 transposed conv upsample + skip concatenation + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(2) != skip.size(2):
            x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# GeoConvNet-1D: Summariser
# ---------------------------------------------------------------------------

class GeoConvNet1DSummariser(nn.Module):
    """
    Translation-equivariant 1D convolutional summariser.

    Compresses a T-step univariate time series to a T/4-step summary
    (2 stride-2 encoder stages), then reconstructs the original from
    the summary features via a symmetric decoder.

    Architecture:
        stem  (T, C)
        enc1  (T/2, C)    ← stride-2 + dilated res block
        enc2  (T/4, 2C)   ← stride-2 + dilated res block  [SUMMARY FEATURES]
        ─────────────────────────────────────────────────
        summary_head: (T/4, 2C) → (T/4, 1)   [1-channel summary signal]
        ─────────────────────────────────────────────────
        dec1  (T/2, C)    ← transposed conv + skip from enc1
        dec0  (T,   C)    ← transposed conv + skip from stem
        recon_head: (T, C) → (T, 1)           [reconstructed signal]
        ─────────────────────────────────────────────────
        cls_head: GAP(enc2) → Linear → (K,)   [downstream classification]

    Args:
        in_channels:  Input channels (1 for univariate time series).
        num_classes:  Number of downstream classification classes.
        base_ch:      Base channel width.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5,
                 base_ch: int = 64):
        super().__init__()
        C = base_ch

        # Stem — no downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Encoder: 2 stride-2 stages → 4x total compression
        self.enc1 = EncoderStage1D(C,   C,   dilation=1)   # T/2
        self.enc2 = EncoderStage1D(C,   C*2, dilation=2)   # T/4  [summary level]

        # Summary head: project rich features → 1-channel readable signal
        self.summary_head = nn.Conv1d(C*2, in_channels, kernel_size=1)

        # Reconstruction decoder (symmetric to encoder)
        self.dec1 = DecoderStage1D(C*2, C,   C)    # T/4 → T/2 + skip(enc1)
        self.dec0 = DecoderStage1D(C,   C,   C)    # T/2 → T   + skip(stem)
        self.recon_head = nn.Conv1d(C, in_channels, kernel_size=1)

        # Downstream classification head (on compressed features)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(C*2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, T) — batch of time series.
        Returns:
            summary: (B, 1, T//4)     — compressed representative signal.
            recon:   (B, 1, T)        — reconstruction of x from summary features.
            logits:  (B, num_classes) — downstream classification logits.
        """
        s  = self.stem(x)          # (B, C,  T)
        e1 = self.enc1(s)          # (B, C,  T/2)
        e2 = self.enc2(e1)         # (B, 2C, T/4)  ← summary features

        summary = self.summary_head(e2)        # (B, 1, T/4)

        d1   = self.dec1(e2, e1)               # (B, C,  T/2)
        d0   = self.dec0(d1, s)                # (B, C,  T)
        recon = self.recon_head(d0)            # (B, 1, T)

        logits = self.cls_head(e2)             # (B, K)

        return summary, recon, logits

    def summarise(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the compressed summary signal. No grad tracking."""
        with torch.no_grad():
            summary, _, _ = self.forward(x)
        return summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeoConvNet1DSummariser(in_channels=1, num_classes=5)
    T = 140
    x = torch.randn(4, 1, T)
    summary, recon, logits = model(x)
    print(f"Input shape:   {x.shape}")            # (4, 1, 140)
    print(f"Summary shape: {summary.shape}")      # (4, 1, 35)
    print(f"Recon shape:   {recon.shape}")        # (4, 1, 140)
    print(f"Logits shape:  {logits.shape}")       # (4, 5)
    print(f"Compression:   {T // summary.size(2)}x")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
