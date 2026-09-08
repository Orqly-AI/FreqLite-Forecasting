"""Distribution-shift stress protocol (A-RevIN robustness analysis).

Motivation: A-RevIN's claim is robustness to lookback->horizon distribution
shift. This script tests that claim DIRECTLY: train the standard way, then
stratify the *test windows* by how far each window's input has drifted from the
training distribution, and compare models on Near / Mid / Far thirds. If
A-RevIN helps where it should, the Far-third gap vs. RLinear shrinks.

Protocol
--------
Datasets x (L, H): ETTh1/ETTm2/weather at L=336, exchange_rate at L=96; H=96.
Models (architecture-controlled ladder, same Linear backbone):
  * rlinear        — RevIN + Linear (the normalization baseline)
  * arevin_linear  — FreqLite cfg K=1 + A-RevIN (init_rho_logit=0.0):
                     ONLY the normalization differs from rlinear
  * freqlite       — full FreqLite (K=2 bands, A-RevIN, rho_logit=0.0)
Training: standard protocol (Adam 1e-3, type1 decay, 20 epochs, patience 3,
MSE), seeds {2021, 2022, 2023}, deterministic.

Shift score (documented)
------------------------
The shift score of a test window is the mean over channels of
|window_input_mean - train_mean| / train_std. The data pipeline (src/data.py)
already z-scores every series with TRAIN-split statistics, so in the model's
input space train_mean = 0 and train_std = 1 per channel; the score therefore
reduces to mean_c | mean_t x[t, c] | of the z-scored input window. Windows are
ranked by this score and split into equal thirds: Near / Mid / Far.

Output: results/shift_stress.csv, one row per (dataset, model, seed, third)
with the third's mean test MSE/MAE and mean shift score. Resumable. No
fabricated numbers — everything comes from real runs.

Usage:
  .venv\\Scripts\\python.exe scripts\\run_shift_stress.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import make_loaders
from src.engine import train_model
from src.models import build_model
from src.utils import get_device, set_seed

OUT = ROOT / "results" / "shift_stress.csv"
FIELDS = ["dataset", "model", "seed", "lookback", "horizon", "third",
          "n_windows", "shift_mean", "mse", "mae", "params", "epochs_run"]

# Training protocol — identical to the main study for comparability.
TR = dict(lr=1e-3, weight_decay=0.0, max_epochs=20, patience=3,
          grad_clip_norm=1.0, lr_schedule="type1")

DECOMP = {"init_cutoff": 0.25, "init_sharpness": 10.0,
          "learnable": True, "mask_eps": 1e-3}


def _freqlite_cfg(K: int, rho_logit: float) -> dict:
    return {"K": K, "recombination": "sum", "decomposition": dict(DECOMP),
            "arevin": {"affine": True, "eps": 1e-5, "adaptive": True,
                       "use_lambda": True, "init_rho_logit": rho_logit}}


# label -> (registry name, build cfg)
MODELS = {
    "rlinear":       ("rlinear",  None),
    "arevin_linear": ("freqlite", _freqlite_cfg(K=1, rho_logit=0.0)),
    "freqlite":      ("freqlite", _freqlite_cfg(K=2, rho_logit=0.0)),
}

# dataset -> lookback (H fixed at 96 everywhere)
GRID = {"ETTh1": 336, "ETTm2": 336, "weather": 336, "exchange_rate": 96}
H = 96
BATCH_SIZE = 32
SEEDS = [2021, 2022, 2023]
THIRDS = ["Near", "Mid", "Far"]


@torch.no_grad()
def per_window_eval(model, loader, device):
    """Iterate the test loader in order (shuffle=False) and return per-window
    arrays: mse, mae (mean over H steps x C channels) and shift score
    (mean over channels of |mean over time of the z-scored input|)."""
    model.eval()
    mses, maes, shifts = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        d = pred - y
        mses.append(d.pow(2).mean(dim=(1, 2)).cpu())
        maes.append(d.abs().mean(dim=(1, 2)).cpu())
        # shift score: train_mean=0, train_std=1 in z-scored space (see docstring)
        shifts.append(x.mean(dim=1).abs().mean(dim=1).cpu())
    return (torch.cat(mses).numpy(),
            torch.cat(maes).numpy(),
            torch.cat(shifts).numpy())


def read_done() -> set:
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["model"], int(r["seed"]), r["third"]))
    return done


def append_row(row: dict) -> None:
    exists = OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = read_done()
    device = get_device()
    print(f"device={device} cuda={torch.cuda.is_available()}  out={OUT}")
    n = 0
    t0 = time.perf_counter()
    for dataset, L in GRID.items():
        loaders, meta = make_loaders(dataset, L=L, H=H, batch_size=BATCH_SIZE)
        C = meta["C"]
        for label, (mname, cfg) in MODELS.items():
            for seed in SEEDS:
                if all((dataset, label, seed, t) in done for t in THIRDS):
                    continue
                set_seed(seed, deterministic=True)
                model = build_model(mname, L=L, H=H, C=C, cfg=cfg)
                res = train_model(model, loaders, device, L=L, H=H, **TR)
                mse_w, mae_w, shift_w = per_window_eval(
                    model, loaders["test"], device)

                # rank windows by shift score, split into equal thirds
                order = np.argsort(shift_w, kind="stable")
                groups = np.array_split(order, 3)  # Near, Mid, Far
                for third, idx in zip(THIRDS, groups):
                    append_row({
                        "dataset": dataset, "model": label, "seed": seed,
                        "lookback": L, "horizon": H, "third": third,
                        "n_windows": len(idx),
                        "shift_mean": f"{shift_w[idx].mean():.6f}",
                        "mse": f"{mse_w[idx].mean():.6f}",
                        "mae": f"{mae_w[idx].mean():.6f}",
                        "params": res.params, "epochs_run": res.epochs_run,
                    })
                n += 1
                far, near = groups[2], groups[0]
                print(f"  [{n}] {label:14s} {dataset:14s} s={seed} "
                      f"near={mse_w[near].mean():.4f} mid={mse_w[groups[1]].mean():.4f} "
                      f"far={mse_w[far].mean():.4f} (overall {mse_w.mean():.4f})")
    print(f"\nDONE: {n} new runs in {(time.perf_counter()-t0)/60:.1f} min -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
