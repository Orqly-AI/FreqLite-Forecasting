"""Non-stationarity rescue study (FreqLite A-RevIN investigation).

Motivation: on the stationary ETT/Weather benchmarks the A-RevIN gate collapses
to rho~=0 (plain RevIN), because the gated correction is gradient-starved at the
init r=-4 (rho~=0.018, a=b=lam=0). This study tests A-RevIN where it SHOULD help
-- genuinely non-stationary series (exchange_rate, national_illness/ILI) -- and
with the gate un-trapped (higher rho init / effectively ungated).

The decisive, architecture-controlled comparison:
    rlinear  (K=1 + plain RevIN)        vs
    arevin_linear (K=1 + A-RevIN, rho-init 0)
Same Linear backbone, only the normalization differs -> isolates A-RevIN.

Writes results/nonstationary.csv (resumable). NO fabricated numbers.

Usage: .venv\\Scripts\\python.exe scripts\\run_nonstationary.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import make_loaders
from src.engine import evaluate, train_model
from src.models import build_model
from src.utils import get_device, set_seed

OUT = ROOT / "results" / "nonstationary.csv"
FIELDS = ["variant", "dataset", "lookback", "horizon", "seed",
          "test_mse", "test_mae", "val_mse", "rho", "params", "epochs_run"]

# Training protocol — identical to the main study for comparability.
TR = dict(lr=1e-3, weight_decay=0.0, max_epochs=20, patience=3,
          grad_clip_norm=1.0, lr_schedule="type1")

DECOMP = {"init_cutoff": 0.25, "init_sharpness": 10.0,
          "learnable": True, "mask_eps": 1e-3}


def _arevin(rho_logit, adaptive=True, use_lambda=True):
    return {"affine": True, "eps": 1e-5, "adaptive": adaptive,
            "use_lambda": use_lambda, "init_rho_logit": rho_logit}


def _freqlite_cfg(K, rho_logit, adaptive=True, use_lambda=True):
    return {"K": K, "recombination": "sum", "decomposition": dict(DECOMP),
            "arevin": _arevin(rho_logit, adaptive, use_lambda)}


# variant label -> (model_name, cfg)  [cfg=None for plain baselines]
VARIANTS = {
    # baselines
    "rlinear":          ("rlinear",  None),
    "dlinear":          ("dlinear",  {"kernel_size": 25}),
    "fits":             ("fits",     {"cutoff_ratio": 0.25}),
    # isolate A-RevIN (K=1, same backbone as RLinear, only norm differs)
    "arevin_linear":    ("freqlite", _freqlite_cfg(K=1, rho_logit=0.0)),
    # full FreqLite at three gate inits
    "freqlite_def":     ("freqlite", _freqlite_cfg(K=2, rho_logit=-4.0)),   # trapped (as main study)
    "freqlite_arevin":  ("freqlite", _freqlite_cfg(K=2, rho_logit=0.0)),    # rho starts 0.5
    "freqlite_forced":  ("freqlite", _freqlite_cfg(K=2, rho_logit=8.0)),    # rho~=1, ungated
}

# dataset -> (lookback, [horizons], batch_size)
GRID = {
    "exchange_rate":     (96, [96, 192, 336, 720], 32),
    "national_illness":  (36, [24, 36, 48, 60],    32),
}
SEEDS = [2021, 2022, 2023]


def read_done():
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["variant"], r["dataset"],
                          int(r["lookback"]), int(r["horizon"]), int(r["seed"])))
    return done


def append_row(row):
    exists = OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = read_done()
    device = get_device()
    print(f"device={device} cuda={torch.cuda.is_available()}  out={OUT}")
    n = 0
    t0 = time.perf_counter()
    for dataset, (L, horizons, bs) in GRID.items():
        for H in horizons:
            loaders, meta = make_loaders(dataset, L=L, H=H, batch_size=bs)
            C = meta["C"]
            for label, (mname, cfg) in VARIANTS.items():
                for seed in SEEDS:
                    key = (label, dataset, L, H, seed)
                    if key in done:
                        continue
                    set_seed(seed, deterministic=True)
                    model = build_model(mname, L=L, H=H, C=C, cfg=cfg)
                    res = train_model(model, loaders, device, L=L, H=H, **TR)
                    mse, mae = evaluate(model, loaders["test"], device)
                    rho = ""
                    if hasattr(model, "learned_params"):
                        lp = model.learned_params()
                        rho = lp.get("arevin", {}).get("rho", "")
                    append_row({
                        "variant": label, "dataset": dataset, "lookback": L,
                        "horizon": H, "seed": seed,
                        "test_mse": f"{mse:.6f}", "test_mae": f"{mae:.6f}",
                        "val_mse": f"{res.best_val:.6f}",
                        "rho": (f"{rho:.4f}" if isinstance(rho, float) else ""),
                        "params": res.params, "epochs_run": res.epochs_run,
                    })
                    n += 1
                    print(f"  [{n}] {label:16s} {dataset:17s} H={H:3d} s={seed} "
                          f"mse={mse:.4f} mae={mae:.4f} rho={rho if rho=='' else f'{rho:.3f}'} "
                          f"p={res.params}")
    print(f"\nDONE: {n} new cells in {(time.perf_counter()-t0)/60:.1f} min -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
