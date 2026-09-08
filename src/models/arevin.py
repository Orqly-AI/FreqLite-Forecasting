"""A-RevIN — Adaptive Reversible Instance Normalization (FreqLite, Sec. 3).

Primary contribution of FreqLite. Generalizes RevIN: the forward normalization
is identical, but denormalization gains a horizon-adaptive, gated correction
that strictly contains RevIN as the rho=0 special case.

Shapes are channel-independent: all tensors are (N, L) / (N, H) with N = B*C.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ARevIN(nn.Module):
    def __init__(
        self,
        H: int,
        eps: float = 1e-5,
        affine: bool = True,
        adaptive: bool = True,
        use_lambda: bool = True,
        init_rho_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.H = H
        self.eps = eps
        self.affine = affine
        self.adaptive = adaptive
        self.use_lambda = use_lambda

        if affine:
            self.gamma = nn.Parameter(torch.ones(1))  # shared scalar affine
            self.beta = nn.Parameter(torch.zeros(1))

        if adaptive:
            # per-horizon-step corrections, all init to identity (0)
            self.a = nn.Parameter(torch.zeros(H))  # log-scale correction
            self.b = nn.Parameter(torch.zeros(H))  # shift correction (sigma units)
            if use_lambda:
                self.lam = nn.Parameter(torch.zeros(H))  # drift propagation
            # gate rho = sigmoid(r); init near 0 => starts as RevIN
            self.r = nn.Parameter(torch.tensor(float(init_rho_logit)))

        self._mean = None
        self._std = None
        self._drift = None

    # ---- forward normalization (identical to RevIN) ----
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=1, keepdim=True)  # (N,1)
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)

        # observed non-stationarity feature: recent-vs-early level drift (detached)
        L = x.shape[1]
        half = L // 2
        early = x[:, :half].mean(dim=1, keepdim=True)
        recent = x[:, half:].mean(dim=1, keepdim=True)
        self._drift = ((recent - early) / self._std).detach()  # (N,1), in sigma units

        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    # ---- adaptive denormalization (the new part) ----
    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        # y: (N, H) prediction in normalized space
        if self.affine:
            y = (y - self.beta) / self.gamma
        base = y * self._std + self._mean  # standard RevIN denorm

        if not self.adaptive:
            return base

        rho = torch.sigmoid(self.r)
        scale = torch.exp(rho * self.a).unsqueeze(0)  # (1,H)
        shift = rho * (self.b * self._std)  # (N,H) via broadcast of (H,) * (N,1)
        if self.use_lambda:
            shift = shift + rho * (self.lam * (self._drift * self._std))
        # apply multiplicative scale about the mean level, then additive shift.
        # base = scale*(y*std) + mean + shift, keeping mean as the pivot so that
        # scale acts on the de-meaned signal (cleaner identity at rho=0).
        centered = base - self._mean
        return scale * centered + self._mean + shift

    @torch.no_grad()
    def learned_params(self) -> dict:
        """Report learned A-RevIN parameters for the interpretability figures."""
        out = {}
        if self.affine:
            out["gamma"] = float(self.gamma)
            out["beta"] = float(self.beta)
        if self.adaptive:
            out["rho"] = float(torch.sigmoid(self.r))
            out["a"] = self.a.detach().cpu().numpy().tolist()
            out["b"] = self.b.detach().cpu().numpy().tolist()
            if self.use_lambda:
                out["lambda"] = self.lam.detach().cpu().numpy().tolist()
        return out
