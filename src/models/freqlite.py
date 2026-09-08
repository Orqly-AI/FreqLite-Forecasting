"""FreqLite — lightweight frequency-decomposed linear model with A-RevIN.

Implements docs/method_spec.md (v1.0) verbatim:
  * Sec. 2 learnable, lossless, partition-of-unity spectral decomposition (K bands)
  * Sec. 3 A-RevIN adaptive reversible instance normalization
  * Sec. 4 per-band linear heads + identity-sum (or gated) recombination

Channel-independent: x (B, L, C) -> fold to (B*C, L) -> (B*C, H) -> (B, H, C).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arevin import ARevIN


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _inv_softplus(y: float) -> float:
    # inverse of softplus: x = log(exp(y) - 1)
    return math.log(math.expm1(y))


class SpectralDecomposition(nn.Module):
    """Learnable soft spectral masks forming a partition of unity over rfft bins.

    Default K=2 (low/high). Masks are real, applied bin-wise to the complex
    spectrum (zero phase distortion), and sum to 1 at every bin so the
    decomposition is lossless before the heads.
    """

    def __init__(
        self,
        L: int,
        K: int = 2,
        init_cutoff: float = 0.25,
        init_sharpness: float = 10.0,
        learnable: bool = True,
        mask_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.L = L
        self.K = K
        self.Fbins = L // 2 + 1
        self.mask_eps = mask_eps

        # normalized frequency per bin in [0,1]
        omega = torch.linspace(0.0, 1.0, self.Fbins)
        self.register_buffer("omega", omega, persistent=False)

        if K < 1:
            raise ValueError("K must be >= 1")
        if K == 1:
            # all-pass single band; no parameters (degenerate to a single head)
            self.cutoffs = None
            self.sharpness = None
            return

        # K-1 cutoffs and sharpnesses.
        # cutoffs are parameterized to stay ordered in (0,1) via cumulative
        # softplus; for K=2 this is just sigmoid(c).
        if K == 2:
            c_raw = torch.tensor([_logit(init_cutoff)])
        else:
            # spread initial cutoffs uniformly in (0,1)
            init = torch.linspace(0, 1, K + 1)[1:-1]  # K-1 interior points
            c_raw = torch.logit(init.clamp(0.02, 0.98))
        tau_raw = torch.full((K - 1,), float(_inv_softplus(init_sharpness)))

        if learnable:
            self.c_raw = nn.Parameter(c_raw)
            self.tau_raw = nn.Parameter(tau_raw)
        else:
            self.register_buffer("c_raw", c_raw, persistent=True)
            self.register_buffer("tau_raw", tau_raw, persistent=True)

    def _cutoffs_sharpness(self):
        if self.K == 2:
            cutoffs = torch.sigmoid(self.c_raw)  # (1,)
        else:
            # ordered cutoffs in (0,1) via normalized cumulative softplus
            sp = F.softplus(self.c_raw)
            cum = torch.cumsum(sp, dim=0)
            cutoffs = cum / (1.0 + sp.sum())  # strictly increasing, in (0,1)
        sharpness = F.softplus(self.tau_raw) + self.mask_eps
        return cutoffs, sharpness

    def masks(self) -> torch.Tensor:
        """Return (K, F) real masks summing to 1 over K at every bin."""
        if self.K == 1:
            return torch.ones(1, self.Fbins, device=self.omega.device)
        cutoffs, sharpness = self._cutoffs_sharpness()
        omega = self.omega  # (F,)
        # cumulative low-pass thresholds s_j(omega) = sigmoid(-tau_j (omega - c_j))
        # m_1 = s_1; m_k = s_k - s_{k-1}; m_K = 1 - s_{K-1}
        sigs = []
        for j in range(self.K - 1):
            sigs.append(torch.sigmoid(-sharpness[j] * (omega - cutoffs[j])))
        masks = []
        masks.append(sigs[0])
        for k in range(1, self.K - 1):
            masks.append(sigs[k] - sigs[k - 1])
        masks.append(1.0 - sigs[-1])
        return torch.stack(masks, dim=0)  # (K, F)

    def forward(self, s_n: torch.Tensor) -> torch.Tensor:
        """s_n: (N, L) -> band signals (K, N, L). Sums over K reconstruct s_n."""
        if self.K == 1:
            return s_n.unsqueeze(0)
        S = torch.fft.rfft(s_n, n=self.L, dim=1)  # (N, F) complex
        m = self.masks().to(S.real.dtype)  # (K, F)
        # broadcast: (K,1,F) * (1,N,F) -> (K,N,F)
        Sk = m.unsqueeze(1) * S.unsqueeze(0)
        bands = torch.fft.irfft(Sk, n=self.L, dim=2)  # (K, N, L)
        return bands

    @torch.no_grad()
    def learned_params(self) -> dict:
        if self.K == 1:
            return {"K": 1}
        cutoffs, sharpness = self._cutoffs_sharpness()
        return {
            "K": self.K,
            "cutoffs": cutoffs.detach().cpu().numpy().tolist(),
            "sharpness": sharpness.detach().cpu().numpy().tolist(),
        }


class FreqLite(nn.Module):
    def __init__(
        self,
        L: int,
        H: int,
        C: int,
        K: int = 2,
        decomposition: dict | None = None,
        arevin: dict | None = None,
        recombination: str = "sum",
        **kwargs,
    ) -> None:
        super().__init__()
        self.L, self.H, self.C, self.K = L, H, C, K
        self.recombination = recombination

        dcfg = decomposition or {}
        self.decomp = SpectralDecomposition(
            L=L,
            K=K,
            init_cutoff=dcfg.get("init_cutoff", 0.25),
            init_sharpness=dcfg.get("init_sharpness", 10.0),
            learnable=dcfg.get("learnable", True),
            mask_eps=dcfg.get("mask_eps", 1e-3),
        )

        acfg = arevin or {}
        self.arevin = ARevIN(
            H=H,
            eps=acfg.get("eps", 1e-5),
            affine=acfg.get("affine", True),
            adaptive=acfg.get("adaptive", True),
            use_lambda=acfg.get("use_lambda", True),
            init_rho_logit=acfg.get("init_rho_logit", -4.0),
        )

        # one linear head per band
        self.heads = nn.ModuleList([nn.Linear(L, H) for _ in range(K)])

        if recombination == "gate":
            self.gate = nn.Parameter(torch.zeros(K))  # softmax over bands
        else:
            self.gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        s = x.permute(0, 2, 1).reshape(B * C, L)  # (N, L)

        s_n = self.arevin.normalize(s)  # (N, L)
        bands = self.decomp(s_n)  # (K, N, L)

        preds = []
        for k in range(self.K):
            preds.append(self.heads[k](bands[k]))  # (N, H)
        preds = torch.stack(preds, dim=0)  # (K, N, H)

        if self.gate is not None:
            g = torch.softmax(self.gate, dim=0).view(self.K, 1, 1)
            p = (g * preds).sum(dim=0)
        else:
            p = preds.sum(dim=0)  # identity-sum recombination (default)

        out = self.arevin.denormalize(p)  # (N, H)
        return out.reshape(B, C, self.H).permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def learned_params(self) -> dict:
        out = {"decomposition": self.decomp.learned_params(),
               "arevin": self.arevin.learned_params()}
        if self.gate is not None:
            out["gate"] = torch.softmax(self.gate, dim=0).detach().cpu().numpy().tolist()
        return out
