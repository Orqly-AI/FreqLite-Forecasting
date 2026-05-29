"""Statistical significance of FreqLite's gains over baselines.

Paired tests across (dataset, horizon, seed) cells: paired t-test and Wilcoxon
signed-rank on per-cell MSE, FreqLite vs each baseline. Writes
results/tables/significance_table.tex and prints a summary. Reads main_results.csv
(L=336 and L=96) and nonstationary.csv (ILI).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"; T.mkdir(parents=True, exist_ok=True)


def paired(df, base, ref="freqlite", key=("dataset", "horizon", "seed")):
    """Return (n, mean_ref, mean_base, pct_improve, t_p, wilcoxon_p)."""
    a = df[df.model == ref].set_index(list(key)).test_mse
    b = df[df.model == base].set_index(list(key)).test_mse
    j = pd.concat({"ref": a, "base": b}, axis=1).dropna()
    if len(j) < 3:
        return None
    d = j["base"] - j["ref"]  # positive => ref (FreqLite) better
    t_p = stats.ttest_rel(j["base"], j["ref"]).pvalue
    try:
        w_p = stats.wilcoxon(j["base"], j["ref"]).pvalue
    except ValueError:
        w_p = float("nan")
    pct = d.mean() / j["base"].mean() * 100
    return len(j), j["ref"].mean(), j["base"].mean(), pct, t_p, w_p


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


df = pd.read_csv(ROOT / "results" / "main_results.csv")
rows = []
print("=== Paired significance: FreqLite vs baseline (positive %% = FreqLite better) ===")
for L in ["L336", "L96"]:
    d = df[df.lookback_label == L]
    for base in ["rlinear", "nlinear", "dlinear", "fits", "patchtst"]:
        r = paired(d, base)
        if r is None:
            continue
        n, mref, mbase, pct, tp, wp = r
        print(f"  {L} FreqLite vs {base:8s}: n={n:2d}  {pct:+5.2f}%  "
              f"t-p={tp:.1e} ({stars(tp)})  wilcoxon-p={wp:.1e} ({stars(wp)})")
        rows.append((L, base, n, pct, tp, wp))

# ILI from the non-stationary study (FreqLite default = freqlite_arevin)
ns = ROOT / "results" / "nonstationary.csv"
if ns.exists():
    nsd = pd.read_csv(ns)
    ili = nsd[nsd.dataset == "national_illness"].copy()
    ili["model"] = ili["variant"]
    r = paired(ili, "rlinear", ref="freqlite_arevin")
    if r:
        n, mref, mbase, pct, tp, wp = r
        print(f"  ILI FreqLite vs rlinear : n={n:2d}  {pct:+5.2f}%  "
              f"t-p={tp:.1e} ({stars(tp)})  wilcoxon-p={wp:.1e} ({stars(wp)})")
        rows.append(("ILI", "rlinear", n, pct, tp, wp))

# LaTeX table
lines = [r"\begin{tabular}{llrrl}", r"\toprule",
         r"Setting & Baseline & $n$ & $\Delta$MSE (\%) & $p$ (Wilcoxon) \\", r"\midrule"]
disp = {"L336": "$L{=}336$", "L96": "$L{=}96$", "ILI": "ILI"}
for L, base, n, pct, tp, wp in rows:
    lines.append(f"{disp.get(L,L)} & {base.upper() if base!='patchtst' else 'PatchTST'} & {n} "
                 f"& {pct:+.2f} & {wp:.1e}~({stars(wp)}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "significance_table.tex").write_text("\n".join(lines), encoding="utf-8")
print("\nwrote results/tables/significance_table.tex")
print("(positive dMSE% = FreqLite lower error; *** p<0.001, ** p<0.01, * p<0.05, ns = not significant)")
