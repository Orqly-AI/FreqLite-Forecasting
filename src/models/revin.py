"""RevIN — Reversible Instance Normalization (Kim et al., ICLR 2022).

Channel-independent, instance-wise normalization over the lookback. We use a
*shared scalar* affine (gamma, beta) rather than per-channel affine to keep the
parameter count independent of C (matches the FreqLite spec, Sec. 3.2). This is
the exact base case that A-RevIN generalizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L) univariate batch (channels folded into N)
        self._mean = x.mean(dim=1, keepdim=True)
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        # y: (N, H) prediction in normalized space
        if self.affine:
            y = (y - self.beta) / self.gamma
        return y * self._std + self._mean
