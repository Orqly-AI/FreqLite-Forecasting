"""Smoke test for the two modern baselines: iTransformer and TimesNet-small.

Checks (per task #14):
  1. forward shapes (B,L,C)->(B,H,C), including a big-C case (C=862, traffic-like)
     at the per-dataset batch size from configs/default.yaml;
  2. 1 training epoch on ETTh1, L=336, H=96 under the standard protocol;
  3. reports params and peak GPU memory; asserts peak < 3500 MB.

Run: .venv\\Scripts\\python.exe scripts\\smoke_new_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data import make_loaders
from src.engine import count_params, evaluate, train_model
from src.models import build_model
from src.utils import get_device, peak_memory_mb, reset_peak_memory, set_seed

MEM_GUARD_MB = 3500.0
NEW_MODELS = ["itransformer", "timesnet"]


def main() -> int:
    device = get_device()
    print(f"device: {device}  cuda={torch.cuda.is_available()}")
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    overrides = cfg.get("model_overrides", {})

    # [1] forward shapes, incl. channel-mixing big-C case at the config batch size
    print("[1] forward shapes ...")
    for name in NEW_MODELS:
        mcfg = overrides.get(name, {})
        for B, L, H, C in [(32, 336, 96, 7), (32, 96, 720, 21), (8, 336, 720, 862)]:
            set_seed(0)
            m = build_model(name, L=L, H=H, C=C, cfg=mcfg).to(device)
            reset_peak_memory()
            x = torch.randn(B, L, C, device=device)
            y = m(x)
            loss = y.square().mean()
            loss.backward()  # include backward activations in the peak
            assert y.shape == (B, H, C), f"{name}: got {tuple(y.shape)}"
            print(f"    {name:13s} B={B:2d} L={L} H={H} C={C:3d}: "
                  f"out={tuple(y.shape)} params={count_params(m):>9,d} "
                  f"fwd+bwd peakMB={peak_memory_mb():7.1f}")
            del m, x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("    OK")

    # [2] 1 epoch on ETTh1 L=336 H=96, standard protocol
    print("[2] 1 training epoch on ETTh1 (L=336, H=96, bs=32) ...")
    L, H = 336, 96
    loaders, meta = make_loaders("ETTh1", L=L, H=H, batch_size=32)
    worst = 0.0
    for name in NEW_MODELS:
        set_seed(2021)
        reset_peak_memory()
        m = build_model(name, L=L, H=H, C=meta["C"], cfg=overrides.get(name, {}))
        res = train_model(m, loaders, device, lr=1e-3, max_epochs=1, patience=3,
                          L=L, H=H)
        tmse, tmae = evaluate(m, loaders["test"], device)
        worst = max(worst, res.peak_gpu_mem_mb)
        print(f"    {name:13s} params={res.params:>9,d} "
              f"sec/ep={res.sec_per_epoch:6.1f} peakMB={res.peak_gpu_mem_mb:7.1f} "
              f"val_mse={res.best_val:.4f} test_mse={tmse:.4f} test_mae={tmae:.4f}")
    assert worst < MEM_GUARD_MB, f"peak {worst:.0f} MB exceeds {MEM_GUARD_MB} guard"
    print(f"    worst peak = {worst:.1f} MB < {MEM_GUARD_MB}  OK")
    print("\nNEW-BASELINE SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
