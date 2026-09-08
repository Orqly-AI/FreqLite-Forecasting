"""Synthetic controlled non-stationarity study (A-RevIN generalization).

Generates synthetic series = stationary seasonal base (sum of sinusoids + noise)
plus a non-stationary random-walk LEVEL DRIFT of controllable magnitude delta.
For each delta we compare, on the SAME backbone, RLinear (plain RevIN) vs
A-RevIN-Linear (K=1 + A-RevIN, rho-init 0), plus full FreqLite. Hypothesis: the
A-RevIN advantage over RevIN, and the learned gate rho, both grow with delta;
at delta=0 (stationary) they tie. This isolates the mechanism in a controlled
setting, complementing the ILI result.

Reuses the standard data pipeline (writes data/synth_d*.csv, 70/10/20 split,
train-stat z-score). Writes results/synthetic.csv + a table + a figure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import make_loaders
from src.engine import evaluate, train_model
from src.models import build_model
from src.utils import get_device, set_seed

DATA = ROOT / "data"
OUT = ROOT / "results" / "synthetic.csv"
T = ROOT / "results" / "tables"; F = ROOT / "results" / "figures"
N, C = 4000, 4              # length, channels
PERIODS = [24.0, 168.0]     # daily / weekly-like seasonality
DELTAS = [0.0, 0.5, 1.0, 2.0, 4.0]
L, H, SEEDS = 96, 96, [2021, 2022, 2023]
TR = dict(lr=1e-3, weight_decay=0.0, max_epochs=20, patience=3,
          grad_clip_norm=1.0, lr_schedule="type1")


def make_templates():
    """Build a SHARED stationary base and a SHARED persistent (extrapolatable)
    trend template per channel, with a single fixed seed, so that across delta the
    ONLY change is the trend magnitude (MSE is comparable; delta=0 is stationary).
    The trend uses AR(1)-persistent increments -> locally linear, predictable
    slope that continues across the lookback->horizon boundary, which is exactly
    the non-stationarity A-RevIN's drift-extrapolation term targets (unlike a pure
    random walk, whose slope is unpredictable)."""
    rng = np.random.default_rng(12345)
    t = np.arange(N)
    base_list, trend_list = [], []
    for c in range(C):
        base = sum(rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * t / p + rng.uniform(0, 2*np.pi))
                   for p in PERIODS) + 0.3 * rng.standard_normal(N)
        # AR(1)-persistent slope -> integrate -> persistent, extrapolatable trend
        phi, incr = 0.995, np.zeros(N)
        raw = rng.standard_normal(N)
        for i in range(1, N):
            incr[i] = phi * incr[i-1] + (1 - phi) * raw[i]
        trend = np.cumsum(incr)
        trend = (trend - trend.mean()) / (trend.std() + 1e-8)  # unit-std template
        base_list.append(base); trend_list.append(trend)
    return base_list, trend_list


def gen_csv(delta: float, base_list, trend_list) -> str:
    """series_c = base_c + delta * trend_c (shared base/trend across delta)."""
    name = f"synth_d{delta:g}".replace(".", "p")
    df = pd.DataFrame({"date": np.arange(N)})
    for c in range(C):
        df[f"ch{c}"] = base_list[c] + delta * trend_list[c]
    df.to_csv(DATA / f"{name}.csv", index=False)
    return name


def arevin_cfg(K):
    return {"K": K, "recombination": "sum",
            "decomposition": {"init_cutoff": 0.25, "init_sharpness": 10.0,
                              "learnable": True, "mask_eps": 1e-3},
            "arevin": {"affine": True, "eps": 1e-5, "adaptive": True,
                       "use_lambda": True, "init_rho_logit": 0.0}}


VARIANTS = {"rlinear": ("rlinear", None),
            "arevin_linear": ("freqlite", arevin_cfg(1)),
            "freqlite": ("freqlite", arevin_cfg(2))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()
    base_list, trend_list = make_templates()
    rows = []
    for delta in DELTAS:
        name = gen_csv(delta, base_list, trend_list)
        loaders, meta = make_loaders(name, L=L, H=H, batch_size=32)
        Cc = meta["C"]
        for label, (mname, cfg) in VARIANTS.items():
            for seed in SEEDS:
                set_seed(seed, deterministic=True)
                model = build_model(mname, L=L, H=H, C=Cc, cfg=cfg)
                res = train_model(model, loaders, device, L=L, H=H, **TR)
                mse, mae = evaluate(model, loaders["test"], device)
                rho = ""
                if hasattr(model, "learned_params"):
                    rho = model.learned_params().get("arevin", {}).get("rho", "")
                rows.append(dict(delta=delta, variant=label, seed=seed,
                                 mse=mse, mae=mae,
                                 rho=(f"{rho:.4f}" if isinstance(rho, float) else "")))
                print(f"  delta={delta:>4} {label:14s} s={seed} mse={mse:.4f} "
                      f"rho={'' if rho=='' else f'{rho:.3f}'}")
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    g = d.groupby(["delta", "variant"]).agg(mse=("mse", "mean"),
                                            rho=("rho", lambda s: pd.to_numeric(s, errors="coerce").mean())).reset_index()
    piv = g.pivot(index="delta", columns="variant", values="mse")
    rho_a = g[g.variant == "arevin_linear"].set_index("delta").rho
    impr = (piv["rlinear"] - piv["arevin_linear"]) / piv["rlinear"] * 100  # % A-RevIN better

    # ---- table ----
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"$\delta$ & RLinear & A-RevIN-Lin. & $\Delta$\% & learned $\bar\rho$ \\", r"\midrule"]
    for delta in DELTAS:
        lines.append(f"{delta:g} & {piv.loc[delta,'rlinear']:.4f} & "
                     f"{piv.loc[delta,'arevin_linear']:.4f} & {impr.loc[delta]:+.2f} & "
                     f"{rho_a.loc[delta]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "synthetic_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # ---- figure ----
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(DELTAS, impr.values, "o-", color="C0", label="A-RevIN MSE reduction (\\%)")
    ax1.set_xlabel(r"injected drift magnitude $\delta$")
    ax1.set_ylabel("MSE reduction vs RevIN (%)", color="C0")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax2 = ax1.twinx()
    ax2.plot(DELTAS, rho_a.values, "s--", color="C3", label=r"learned gate $\bar\rho$")
    ax2.set_ylabel(r"learned gate $\bar\rho$", color="C3")
    ax1.set_title("A-RevIN engages as non-stationarity grows")
    fig.tight_layout()
    fig.savefig(F / "synthetic_drift.pdf"); fig.savefig(F / "synthetic_drift.png", dpi=150)
    print("\nwrote results/synthetic.csv, tables/synthetic_table.tex, figures/synthetic_drift.{pdf,png}")
    print("\nDrift sweep (A-RevIN vs RevIN):")
    for delta in DELTAS:
        print(f"  delta={delta:>4}: RLinear {piv.loc[delta,'rlinear']:.4f} -> A-RevIN "
              f"{piv.loc[delta,'arevin_linear']:.4f}  ({impr.loc[delta]:+.2f}%)  rho={rho_a.loc[delta]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
