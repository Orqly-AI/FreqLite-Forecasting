"""Ablation-only model variants for FreqLite (method_spec Sec. 9).

FreqLiteFixedMA (variant A2): replace the learnable spectral decomposition with
DLinear's fixed moving-average trend/seasonal split, while keeping A-RevIN and
the two-head + identity-sum structure. This isolates the value of the *learnable
spectral* split vs. a fixed time-domain split, holding everything else fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .arevin import ARevIN
from .baselines import _MovingAvg


class FreqLiteFixedMA(nn.Module):
    def __init__(self, L: int, H: int, C: int, kernel_size: int = 25,
                 arevin: dict | None = None, **kwargs) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.decomp = _MovingAvg(kernel_size)
        acfg = arevin or {}
        self.arevin = ARevIN(
            H=H, eps=acfg.get("eps", 1e-5), affine=acfg.get("affine", True),
            adaptive=acfg.get("adaptive", True),
            use_lambda=acfg.get("use_lambda", True),
            init_rho_logit=acfg.get("init_rho_logit", -4.0),
        )
        self.head_trend = nn.Linear(L, H)
        self.head_seasonal = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        s = x.permute(0, 2, 1).reshape(B * C, L)
        s_n = self.arevin.normalize(s)
        trend = self.decomp(s_n)
        seasonal = s_n - trend
        p = self.head_trend(trend) + self.head_seasonal(seasonal)
        out = self.arevin.denormalize(p)
        return out.reshape(B, C, self.H).permute(0, 2, 1).contiguous()
