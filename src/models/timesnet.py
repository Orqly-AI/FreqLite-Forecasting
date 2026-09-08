"""TimesNet-small (Wu et al., ICLR 2023; arXiv 2210.02186) — 4 GB-sized variant.

Faithful to the official ``Time-Series-Library`` forecasting model, scaled down:

* Non-stationary normalization (subtract per-channel lookback mean, divide by
  std; re-applied at the output) — as in the official forecast forward.
* Embedding: a linear map of the C input channels to d_model channels at every
  time step (we use a plain ``Linear(C -> d_model)``; the official repo uses a
  token conv + temporal-feature embedding, which we drop since we do not feed
  calendar features — documented simplification).
* **Forecast head choice (documented):** we use the OFFICIAL repo's forecast
  head — a ``Linear(L -> L+H)`` time-projection applied right after the
  embedding, so the TimesBlocks operate on the full length-(L+H) sequence and
  the last H steps (after a ``Linear(d_model -> C)`` projection) are the
  forecast — rather than the simpler per-channel ``Linear(L -> H)`` after the
  blocks.
* TimesBlock: rFFT over time to find the top-k dominant periods (DC excluded),
  reshape 1D -> 2D as (cycles x period), Inception-style multi-kernel 2D convs,
  reshape back, aggregate the k branches weighted by softmax of the FFT
  amplitudes (gradients flow through the amplitudes, as in the official code),
  plus a residual connection and per-block LayerNorm.

Like iTransformer, TimesNet is NOT channel-independent: the embedding mixes all
C channels. Memory scales with B * (L+H) * d_ff and the 2D conv activations;
the small config (d_model=32, d_ff=32, top_k=3, e_layers=2, num_kernels=4) fits
easily in 4 GB; large-C datasets only grow the two C<->d_model linear maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Inception_Block_V1 of the official repo: parallel 2D convs with kernel
    sizes 1,3,5,... (2i+1), same-padded, averaged."""

    def __init__(self, in_ch: int, out_ch: int, num_kernels: int = 4) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


def _fft_periods(x: torch.Tensor, k: int):
    """Find the top-k dominant periods of x: (B, T, D) via rFFT amplitude.

    Returns (periods: list[int], weights: (B, k) amplitudes at those bins).
    Amplitudes keep their graph so the softmax aggregation is trained, exactly
    as in the official ``FFT_for_Period``.
    """
    T = x.shape[1]
    xf = torch.fft.rfft(x, dim=1)                      # (B, F, D) complex
    # detached: used only to SELECT the top-k bins (no grad through indices)
    amp = xf.abs().mean(0).mean(-1).detach()           # (F,) batch+channel mean
    amp[0] = 0.0                                       # exclude DC
    k = min(k, amp.shape[0] - 1)
    _, top = torch.topk(amp, k)
    top = top.detach().cpu().tolist()                  # python ints (determinism)
    periods = [max(1, T // f) for f in top]
    weights = xf.abs().mean(-1)[:, top]                # (B, k)
    return periods, weights


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int) -> None:
        super().__init__()
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        periods, weights = _fft_periods(x, self.k)
        outs = []
        for p in periods:
            # pad T up to a multiple of the period, reshape 1D -> 2D
            if T % p != 0:
                length = (T // p + 1) * p
                pad = x.new_zeros(B, length - T, D)
                z = torch.cat([x, pad], dim=1)
            else:
                length = T
                z = x
            # (B, length, D) -> (B, D, cycles, period): rows=inter-period,
            # cols=intra-period variation
            z = z.reshape(B, length // p, p, D).permute(0, 3, 1, 2).contiguous()
            z = self.conv(z)
            z = z.permute(0, 2, 3, 1).reshape(B, length, D)[:, :T]
            outs.append(z)
        out = torch.stack(outs, dim=-1)                    # (B, T, D, k)
        w = F.softmax(weights, dim=1)                      # (B, k), amplitude-weighted
        out = (out * w.unsqueeze(1).unsqueeze(1)).sum(-1)  # aggregate branches
        return out + x                                     # residual


class TimesNet(nn.Module):
    def __init__(
        self,
        L: int,
        H: int,
        C: int,
        d_model: int = 32,
        d_ff: int = 32,
        top_k: int = 3,
        e_layers: int = 2,
        num_kernels: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.eps = eps

        self.embed = nn.Linear(C, d_model)
        self.drop = nn.Dropout(dropout)
        # official forecast head: extend the time axis to L+H BEFORE the blocks
        self.predict_linear = nn.Linear(L, L + H)
        self.blocks = nn.ModuleList(
            TimesBlock(d_model, d_ff, top_k, num_kernels) for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        # Non-stationary normalization (official forecast forward)
        mean = x.mean(dim=1, keepdim=True)                            # (B,1,C)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        xn = (x - mean) / std

        z = self.drop(self.embed(xn))                                 # (B, L, d)
        z = self.predict_linear(z.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L+H, d)
        for blk in self.blocks:
            z = self.norm(blk(z))
        out = self.proj(z)                                            # (B, L+H, C)

        out = out * std + mean                                        # de-normalize
        return out[:, -self.H:]                                       # (B, H, C)
