"""Smoke test: prove the full pipeline runs end-to-end in <4 GB before full runs.

Checks, on small configs:
  1. data loaders build with correct shapes for an ETT dataset;
  2. every model forward-passes with shape (B,L,C)->(B,H,C);
  3. FreqLite's spectral decomposition is a lossless partition of unity
     (sum of bands == normalized input);
  4. A short train loop runs for FreqLite + each baseline and records
     params / peak GPU memory; asserts peak < 3.5 GB (4 GB guardrail);
  5. FreqLite degenerates to RLinear numerically when K=1 + RevIN (rho frozen 0).

Run: .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import make_loaders
from src.engine import count_params, evaluate, train_model
from src.models import available_models, build_model
from src.models.freqlite import FreqLite
from src.utils import get_device, peak_memory_mb, reset_peak_memory, set_seed

MEM_GUARD_MB = 3500.0


def check_decomposition_lossless():
    print("[3] FreqLite decomposition lossless partition-of-unity ...")
    torch.manual_seed(0)
    for K in (1, 2, 3, 4):
        m = FreqLite(L=96, H=96, C=7, K=K)
        s = torch.randn(5, 96)
        bands = m.decomp(s)  # (K, N, L)
        recon = bands.sum(dim=0)
        err = (recon - s).abs().max().item()
        assert err < 1e-4, f"K={K} reconstruction error {err}"
        masks = m.decomp.masks() if K > 1 else torch.ones(1, m.decomp.Fbins)
        pou = (masks.sum(dim=0) - 1.0).abs().max().item()
        print(f"    K={K}: recon_err={err:.2e}  partition_err={pou:.2e}")
        assert pou < 1e-5, f"masks do not partition unity (K={K})"
    print("    OK")


def check_models_forward(device):
    print("[2] forward pass shapes for all models ...")
    B, L, H, C = 4, 96, 48, 7
    x = torch.randn(B, L, C, device=device)
    for name in available_models():
        cfg = {}
        m = build_model(name, L=L, H=H, C=C, cfg=cfg).to(device)
        y = m(x)
        assert y.shape == (B, H, C), f"{name}: got {tuple(y.shape)}"
        print(f"    {name:10s} params={count_params(m):>8d}  out={tuple(y.shape)}")
    print("    OK")


def check_freqlite_nests_rlinear(device):
    print("[5] FreqLite(K=1, non-adaptive RevIN) == RLinear (same weights) ...")
    L, H, C = 96, 48, 7
    set_seed(0)
    fl = FreqLite(
        L=L, H=H, C=C, K=1,
        arevin={"adaptive": False, "affine": True},
    ).to(device)
    from src.models.baselines import RLinear

    rl = RLinear(L=L, H=H, C=C).to(device)
    # copy weights: FreqLite head -> RLinear linear; affine gamma/beta
    rl.linear.load_state_dict(fl.heads[0].state_dict())
    rl.revin.gamma.data.copy_(fl.arevin.gamma.data)
    rl.revin.beta.data.copy_(fl.arevin.beta.data)
    x = torch.randn(3, L, C, device=device)
    with torch.no_grad():
        d = (fl(x) - rl(x)).abs().max().item()
    print(f"    max abs diff = {d:.2e}")
    assert d < 1e-5, "FreqLite K=1 does not match RLinear"
    print("    OK")


def check_train_and_memory(device):
    print("[4] short train loop + memory for each model on ETTh1 (L=96,H=48) ...")
    L, H = 96, 48
    loaders, meta = make_loaders("ETTh1", L=L, H=H, batch_size=32)
    print(f"    ETTh1 C={meta['C']} sizes={meta['sizes']}")
    worst_mem = 0.0
    for name in available_models():
        set_seed(2021)
        reset_peak_memory()
        m = build_model(name, L=L, H=H, C=meta["C"])
        res = train_model(
            m, loaders, device,
            lr=1e-3, max_epochs=2, patience=3, L=L, H=H, verbose=False,
        )
        tmse, tmae = evaluate(m, loaders["test"], device)
        worst_mem = max(worst_mem, res.peak_gpu_mem_mb)
        print(
            f"    {name:10s} params={res.params:>8d} "
            f"flops/series={res.flops_per_series:>9d} "
            f"sec/ep={res.sec_per_epoch:6.2f} peakMB={res.peak_gpu_mem_mb:7.1f} "
            f"test_mse={tmse:.4f} test_mae={tmae:.4f}"
        )
    print(f"    worst peak GPU mem = {worst_mem:.1f} MB (guard {MEM_GUARD_MB})")
    assert worst_mem < MEM_GUARD_MB, "exceeded 4 GB memory guard!"
    print("    OK")


def main() -> int:
    device = get_device()
    print(f"device: {device}  cuda.is_available()={torch.cuda.is_available()}")
    print("[1] data loaders ...")
    loaders, meta = make_loaders("ETTh1", L=96, H=48, batch_size=32)
    x, y = next(iter(loaders["train"]))
    print(f"    batch x={tuple(x.shape)} y={tuple(y.shape)}  OK")

    check_models_forward(device)
    check_decomposition_lossless()
    check_train_and_memory(device)
    check_freqlite_nests_rlinear(device)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
