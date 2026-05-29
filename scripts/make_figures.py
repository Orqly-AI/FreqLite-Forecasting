"""Generate paper figures from results.

Figures (saved as PDF + PNG in results/figures/):
  1. accuracy_vs_params.{pdf,png}  — MSE vs. parameter count scatter (efficiency).
  2. learned_filter.{pdf,png}      — learned low/high spectral masks per dataset.
  3. arevin_profile.{pdf,png}      — learned A-RevIN per-step a/b/lambda profiles.

Fig 1 reads results/main_results_agg.csv. Figs 2-3 train a FreqLite per dataset
(short, deterministic) and read its learned_params(); the trained values are also
dumped to results/learned_params.json for the writer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import make_loaders
from src.engine import train_model
from src.models.freqlite import FreqLite
from src.utils import get_device, set_seed

FIG_DIR = ROOT / "results" / "figures"
MODEL_LABEL = {
    "naive": "Naive", "nlinear": "NLinear", "dlinear": "DLinear",
    "rlinear": "RLinear", "fits": "FITS", "patchtst": "PatchTST*",
    "freqlite": "FreqLite",
}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf / .png")


def fig_accuracy_vs_params(agg_path: Path, H: int = 720, L: int = 336):
    if not agg_path.exists():
        print("no agg csv; skip accuracy_vs_params")
        return
    df = pd.read_csv(agg_path)
    sub = df[(df.horizon == H) & (df.lookback == L) & (df.model != "naive")]
    if sub.empty:
        print("no rows for accuracy_vs_params; skip")
        return
    # average MSE across datasets per model; params from first cell
    g = sub.groupby("model").agg(mse=("mse_mean", "mean"),
                                 params=("params", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for _, r in g.iterrows():
        m = r["model"]
        marker = "*" if m == "freqlite" else "o"
        size = 220 if m == "freqlite" else 90
        ax.scatter(r["params"], r["mse"], s=size, marker=marker, zorder=3,
                   label=MODEL_LABEL.get(m, m))
        ax.annotate(MODEL_LABEL.get(m, m), (r["params"], r["mse"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel(f"Mean test MSE (H={H})")
    ax.set_title(f"Accuracy vs. model size  (L={L}, averaged over datasets)")
    ax.grid(True, alpha=0.3)
    _save(fig, "accuracy_vs_params")


def _train_freqlite(dataset, L, H, device, max_epochs):
    loaders, meta = make_loaders(dataset, L=L, H=H, batch_size=32)
    set_seed(2021)
    model = FreqLite(L=L, H=H, C=meta["C"], K=2)
    train_model(model, loaders, device, lr=1e-3, max_epochs=max_epochs,
                patience=3, L=L, H=H)
    return model


def fig_learned(datasets, L, H, device, max_epochs):
    learned = {}
    # ---- filter shapes ----
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    # ---- A-RevIN profiles ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.2))
    for d in datasets:
        model = _train_freqlite(d, L, H, device, max_epochs)
        lp = model.learned_params()
        learned[d] = lp
        # filter masks
        with torch.no_grad():
            masks = model.decomp.masks().cpu().numpy()  # (K,F)
            omega = model.decomp.omega.cpu().numpy()
        ax1.plot(omega, masks[0], label=f"{d} low", lw=1.8)
        ax1.plot(omega, masks[-1], "--", label=f"{d} high", lw=1.2, alpha=0.7)
        # A-RevIN profiles
        a = np.array(lp["arevin"].get("a", []))
        b = np.array(lp["arevin"].get("b", []))
        lam = np.array(lp["arevin"].get("lambda", []))
        rho = lp["arevin"].get("rho", 0.0)
        steps = np.arange(1, len(a) + 1)
        axes2[0].plot(steps, np.exp(rho * a), label=d, lw=1.5)
        axes2[1].plot(steps, rho * b, label=d, lw=1.5)
        if len(lam):
            axes2[2].plot(steps, rho * lam, label=d, lw=1.5)

    ax1.set_xlabel("Normalized frequency $\\omega$ (0=DC, 1=Nyquist)")
    ax1.set_ylabel("Mask weight")
    ax1.set_title(f"Learned spectral masks (L={L}, H={H})")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    _save(fig1, "learned_filter")

    axes2[0].set_title("scale $\\exp(\\rho a_t)$")
    axes2[1].set_title("shift $\\rho b_t$ ($\\sigma$ units)")
    axes2[2].set_title("drift gain $\\rho\\lambda_t$")
    for ax in axes2:
        ax.set_xlabel("horizon step $t$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig2.suptitle(f"Learned A-RevIN horizon-correction profiles (L={L}, H={H})")
    _save(fig2, "arevin_profile")

    (ROOT / "results" / "learned_params.json").write_text(
        json.dumps(learned, indent=2), encoding="utf-8")
    print("wrote results/learned_params.json")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="results/main_results_agg.csv")
    ap.add_argument("--datasets", default="ETTh1,ETTm2,weather")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--max-epochs", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    fig_accuracy_vs_params(ROOT / args.agg, H=args.H, L=args.L)
    fig_learned(args.datasets.split(","), args.L, args.H, device, args.max_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
