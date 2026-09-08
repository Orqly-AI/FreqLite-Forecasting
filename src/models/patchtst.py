"""PatchTST-small (Nie et al., ICLR 2023), compact channel-independent variant.

Tiny config for the 4 GB budget (default d_model=64, 2 layers, 4 heads). CI: each
channel is an independent univariate sequence; patches are linearly embedded, a
transformer encoder processes them, and a flatten+linear head maps to H. A RevIN
instance norm wraps the model (standard for PatchTST).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .revin import RevIN


class PatchTST(nn.Module):
    def __init__(
        self,
        L: int,
        H: int,
        C: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.2,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.patch_len = patch_len
        self.stride = stride
        self.revin = RevIN(eps=eps, affine=True)

        # number of patches after padding the end so the last partial patch fits
        self.pad = stride
        n_patches = (L + self.pad - patch_len) // stride + 1
        self.n_patches = n_patches

        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Linear(n_patches * d_model, H)

    def _patchify(self, s: torch.Tensor) -> torch.Tensor:
        # s: (N, L) -> pad end by replicating last value, then unfold into patches
        s = torch.cat([s, s[:, -1:].repeat(1, self.pad)], dim=1)
        patches = s.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches  # (N, n_patches, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        s = x.permute(0, 2, 1).reshape(B * C, L)  # CI fold
        s = self.revin.normalize(s)

        p = self._patchify(s)  # (N, n_patches, patch_len)
        z = self.embed(p) + self.pos
        z = self.drop(z)
        z = self.encoder(z)  # (N, n_patches, d_model)
        z = z.reshape(z.shape[0], -1)
        out = self.head(z)  # (N, H)

        out = self.revin.denormalize(out)
        return out.reshape(B, C, self.H).permute(0, 2, 1).contiguous()
