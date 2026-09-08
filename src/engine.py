"""Training / evaluation harness for FreqLite and baselines.

Implements the standard LTSF protocol (method_spec Sec. 5): MSE loss in the
train-z-scored space, Adam, type1 LR step decay, early stopping on val loss,
gradient clipping, fixed seeds. Records efficiency metrics (params, FLOPs,
sec/epoch, peak GPU mem) for every run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .utils import peak_memory_mb, reset_peak_memory


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mse_mae(pred: torch.Tensor, true: torch.Tensor):
    """Mean MSE and MAE over all elements (train-z-scored space)."""
    diff = pred - true
    mse = torch.mean(diff * diff).item()
    mae = torch.mean(torch.abs(diff)).item()
    return mse, mae


# --------------------------------------------------------------------------- #
# Parameter / FLOP accounting
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> int:
    """Trainable parameter count. Complex params count as 2 reals (re+im)."""
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            if torch.is_complex(p):
                n *= 2
            total += n
    return total


def analytic_flops_per_series(model: nn.Module, L: int, H: int) -> int:
    """Analytic forward MACs*2 per univariate series for the linear-class models.

    For linear heads y=Wx+b with W:(H,L): 2*H*L FLOPs (mul+add). We report a
    coarse analytic count (the dominant term) per series; transformer models
    return 0 here and rely on the empirical efficiency table instead.
    """
    name = type(model).__name__.lower()
    if "freqlite" in name:
        K = getattr(model, "K", 2)
        # K heads (2HL each) + FFT/iFFT (~K * 5 L log2 L) — heads dominate
        import math

        fft = int(2 * K * 5 * L * max(1, math.log2(max(L, 2))))
        return K * 2 * H * L + fft
    if "dlinear" in name:
        return 2 * (2 * H * L)  # two heads
    if name in ("nlinear", "rlinear"):
        return 2 * H * L
    if "fits" in name:
        nk = getattr(model, "n_keep", L // 2 + 1)
        nko = getattr(model, "n_keep_out", nk)
        # complex matmul ~ 4 real MACs; *2 for FLOPs
        return 2 * 4 * nk * nko
    if "naive" in name:
        return 0
    return 0  # patchtst: use empirical timing/memory only


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    best_val: float = float("inf")
    epochs_run: int = 0
    sec_per_epoch: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    params: int = 0
    flops_per_series: int = 0
    history: list = field(default_factory=list)


def _type1_lr(epoch: int, base_lr: float) -> float:
    # type1 schedule (DLinear/Autoformer): halve each epoch after epoch 1
    return base_lr * (0.5 ** max(0, epoch - 1))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    tot_mse = tot_mae = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        b = x.shape[0]
        mse, mae = mse_mae(pred, y)
        tot_mse += mse * b
        tot_mae += mae * b
        n += b
    return tot_mse / n, tot_mae / n


def train_model(
    model,
    loaders,
    device,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 20,
    patience: int = 3,
    grad_clip_norm: float = 1.0,
    lr_schedule: str = "type1",
    L: int = 336,
    H: int = 96,
    verbose: bool = False,
) -> TrainResult:
    model.to(device)
    res = TrainResult(
        params=count_params(model),
        flops_per_series=analytic_flops_per_series(model, L, H),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:  # naive baseline — no training, just evaluate
        reset_peak_memory()
        vmse, _ = evaluate(model, loaders["val"], device)
        res.best_val = vmse
        res.peak_gpu_mem_mb = peak_memory_mb()
        return res

    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    reset_peak_memory()
    epoch_times = []

    for epoch in range(1, max_epochs + 1):
        if lr_schedule == "type1":
            for g in opt.param_groups:
                g["lr"] = _type1_lr(epoch, lr)
        model.train()
        t0 = time.perf_counter()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip_norm:
                nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        vmse, vmae = evaluate(model, loaders["val"], device)
        res.history.append({"epoch": epoch, "val_mse": vmse, "val_mae": vmae})
        if verbose:
            print(f"    epoch {epoch:2d}  val_mse={vmse:.5f} val_mae={vmae:.5f}")

        if vmse < best_val - 1e-7:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    res.epochs_run = epoch
    res.sec_per_epoch = sum(epoch_times) / len(epoch_times)
    res.peak_gpu_mem_mb = peak_memory_mb()
    res.best_val = best_val
    if best_state is not None:
        model.load_state_dict(best_state)
    return res
