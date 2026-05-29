"""Linear / frequency baselines, all channel-independent (CI).

CI convention: input x is (B, L, C). We permute to (B, C, L), fold channels into
the batch -> (B*C, L), apply a univariate model with shared weights, and reshape
the (B*C, H) output back to (B, H, C). This keeps parameter counts independent of
C and matches the strong-baseline setting used throughout the LTSF literature.

Implemented here:
  * NLinear  — subtract last value, Linear(L,H), add it back.
  * DLinear  — moving-average trend/seasonal split + two Linear(L,H), summed.
  * RLinear  — RevIN + single Linear(L,H).
  * FITS     — RIN + low-pass rFFT cutoff + single COMPLEX Linear + irFFT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .revin import RevIN


def _to_ci(x: torch.Tensor):
    """(B, L, C) -> (B*C, L), returns folded tensor and (B, C) for unfolding."""
    B, L, C = x.shape
    x = x.permute(0, 2, 1).reshape(B * C, L)
    return x, B, C


def _from_ci(y: torch.Tensor, B: int, C: int) -> torch.Tensor:
    """(B*C, H) -> (B, H, C)."""
    H = y.shape[-1]
    return y.reshape(B, C, H).permute(0, 2, 1).contiguous()


class NLinear(nn.Module):
    """Simplest distribution-shift-robust linear model (Zeng et al., 2023)."""

    def __init__(self, L: int, H: int, C: int, **kwargs) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.linear = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, B, C = _to_ci(x)
        last = s[:, -1:]  # subtract the last observed value, add it back after
        out = self.linear(s - last) + last
        return _from_ci(out, B, C)


class _MovingAvg(nn.Module):
    """Moving average with edge padding (DLinear trend extractor)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (N, L). Pad both ends by replicating the boundary values.
        pad = (self.kernel_size - 1) // 2
        front = s[:, :1].repeat(1, pad)
        end = s[:, -1:].repeat(1, self.kernel_size - 1 - pad)
        s_pad = torch.cat([front, s, end], dim=1)
        return self.avg(s_pad.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """Trend/seasonal decomposition + two linear heads (Zeng et al., 2023)."""

    def __init__(self, L: int, H: int, C: int, kernel_size: int = 25, **kwargs) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.decomp = _MovingAvg(kernel_size)
        self.lin_trend = nn.Linear(L, H)
        self.lin_seasonal = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, B, C = _to_ci(x)
        trend = self.decomp(s)
        seasonal = s - trend
        out = self.lin_seasonal(seasonal) + self.lin_trend(trend)
        return _from_ci(out, B, C)


class RLinear(nn.Module):
    """RevIN + single linear layer (Li et al., 2023). A-RevIN's base case."""

    def __init__(self, L: int, H: int, C: int, eps: float = 1e-5,
                 affine: bool = True, **kwargs) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.revin = RevIN(eps=eps, affine=affine)
        self.linear = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, B, C = _to_ci(x)
        s = self.revin.normalize(s)
        out = self.linear(s)
        out = self.revin.denormalize(out)
        return _from_ci(out, B, C)


class FITS(nn.Module):
    """FITS — Frequency Interpolation Time Series (Xu, Zeng, Xu, ICLR 2024).

    Faithful re-implementation of arXiv 2307.03756, channel-independent, ~10k
    params. Pipeline:
      1. RIN (= instance norm; here mean-subtraction + std-division, the FITS
         "Reversible Instance Norm"). We use the standard RevIN form.
      2. rFFT of the length-L lookback  -> F = L//2 + 1 complex bins.
      3. Low-pass cutoff: keep only the lowest ``n_keep`` bins (dominant
         harmonics). ``n_keep`` is set from the cutoff-frequency hyperparameter
         (COF). Default heuristic: keep bins up to ``cutoff_ratio`` of F, but
         the spec exposes it in config.
      4. A single COMPLEX linear layer maps the kept low-freq spectrum (length
         n_keep) to the spectrum of length (L+H) low-freq bins
         (n_keep_out = ceil(n_keep * (L+H)/L)) — i.e. reconstruct + extrapolate.
      5. irFFT back to length (L+H); take the last H steps as the forecast.
         FITS also scales by (L+H)/L to compensate irFFT length change.
      6. de-normalize (RIN inverse).
    """

    def __init__(
        self,
        L: int,
        H: int,
        C: int,
        cutoff_ratio: float = 0.25,
        n_keep: int | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.eps = eps
        self.out_len = L + H

        F_in = L // 2 + 1
        F_out = self.out_len // 2 + 1
        if n_keep is None:
            n_keep = max(1, int(round(cutoff_ratio * F_in)))
        n_keep = min(n_keep, F_in)
        # keep proportional number of low-freq bins in the longer spectrum
        n_keep_out = min(F_out, int(round(n_keep * self.out_len / L)))
        n_keep_out = max(n_keep_out, n_keep)
        self.n_keep = n_keep
        self.n_keep_out = n_keep_out

        # Single complex-valued linear layer (FITS core). PyTorch supports
        # complex weights; this is exactly one complex Linear with bias.
        self.freq_linear = nn.Linear(n_keep, n_keep_out, bias=True, dtype=torch.cfloat)
        self.length_ratio = self.out_len / L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, B, C = _to_ci(x)  # (N, L)
        # ---- RIN (instance norm) ----
        mean = s.mean(dim=1, keepdim=True)
        std = torch.sqrt(s.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        s_n = (s - mean) / std

        # ---- rFFT + low-pass cutoff ----
        spec = torch.fft.rfft(s_n, dim=1)  # (N, F_in) complex
        low = spec[:, : self.n_keep]  # keep lowest n_keep bins

        # ---- complex linear: reconstruct + extrapolate ----
        out_low = self.freq_linear(low)  # (N, n_keep_out) complex

        # place into a full-length output spectrum (zeros above cutoff)
        F_out = self.out_len // 2 + 1
        full = torch.zeros(s.shape[0], F_out, dtype=out_low.dtype, device=s.device)
        full[:, : self.n_keep_out] = out_low

        # ---- irFFT to (L+H); compensate energy for the length change ----
        recon = torch.fft.irfft(full, n=self.out_len, dim=1) * self.length_ratio

        pred_n = recon[:, -self.H :]  # last H steps = forecast
        # ---- de-normalize ----
        out = pred_n * std + mean
        return _from_ci(out, B, C)
