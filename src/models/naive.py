"""Naive repeat-last baseline (sanity floor, no parameters)."""

from __future__ import annotations

import torch
import torch.nn as nn


class NaiveRepeatLast(nn.Module):
    """ŷ_t = x_L for all t (repeat the last observed value over the horizon).

    Channel-wise; no trainable parameters. Input x: (B, L, C) -> (B, H, C).
    """

    def __init__(self, L: int, H: int, C: int, **kwargs) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        # a dummy parameter so optimizer/.parameters() never empties (harness-safe)
        self.register_buffer("_zero", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]  # (B, 1, C)
        return last.expand(-1, self.H, -1).contiguous()
