"""Emit LaTeX tables + summary for the non-stationarity / A-RevIN study.

Reads results/nonstationary.csv and writes:
  * results/tables/nonstationary_table.tex   (MSE/MAE: baselines vs FreqLite on
        exchange_rate + ILI; FreqLite = rho-init-0 default = variant 'freqlite_arevin')
  * results/tables/arevin_gate_table.tex      (ILI gate-strength sweep:
        RLinear -> A-RevIN-Linear -> FreqLite(rho0) -> FreqLite(forced), + learned rho)
Also prints the key numbers used in prose (no fabrication; everything from the CSV).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "results" / "nonstationary.csv")
T = ROOT / "results" / "tables"
T.mkdir(parents=True, exist_ok=True)

agg = d.groupby(["variant", "dataset", "horizon"]).agg(
    mse=("test_mse", "mean"), mae=("test_mae", "mean"),
    mse_std=("test_mse", "std"), rho=("rho", "mean")).reset_index()

DISP = {"exchange_rate": "Exchange", "national_illness": "ILI"}
# headline table: the model we advocate is FreqLite with rho-init 0 == 'freqlite_arevin'
HEAD = [("rlinear", "RLinear"), ("dlinear", "DLinear"), ("fits", "FITS"),
        ("freqlite_arevin", "FreqLite")]


def fmt(x):
    return f"{x:.3f}"


def best_per_row(rows):
    return min(rows, key=lambda kv: kv[1])[0]


# ---- Table 1: non-stationary benchmark (MSE) ----
lines = [r"\begin{tabular}{ll" + "c" * len(HEAD) + "}", r"\toprule",
         "Dataset & $H$ & " + " & ".join(n for _, n in HEAD) + r" \\", r"\midrule"]
for ds in ["exchange_rate", "national_illness"]:
    hs = sorted(agg[agg.dataset == ds].horizon.unique())
    for i, H in enumerate(hs):
        vals = {v: agg[(agg.variant == v) & (agg.dataset == ds) & (agg.horizon == H)].mse.values
                for v, _ in HEAD}
        vals = {v: float(a[0]) for v, a in vals.items() if len(a)}
        bestv = best_per_row(list(vals.items())) if vals else None
        cells = []
        for v, _ in HEAD:
            if v in vals:
                s = fmt(vals[v])
                cells.append(r"\textbf{" + s + "}" if v == bestv else s)
            else:
                cells.append("--")
        dname = DISP[ds] if i == 0 else ""
        lines.append(f"{dname} & {H} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
(T / "nonstationary_table.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote nonstationary_table.tex")

# ---- Table 2: ILI gate sweep ----
SWEEP = [("rlinear", "RLinear (RevIN)", False),
         ("arevin_linear", "A-RevIN-Linear", True),
         ("freqlite_def", "FreqLite ($\\rho_0{=}{-}4$, trapped)", True),
         ("freqlite_arevin", "FreqLite ($\\rho_0{=}0$)", True),
         ("freqlite_forced", "FreqLite (forced, $\\rho{\\approx}1$)", True)]
ili = agg[agg.dataset == "national_illness"]
hs = sorted(ili.horizon.unique())
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         "Model & $\\bar\\rho$ & " + " & ".join(f"$H{{=}}{H}$" for H in hs) + r" & Avg \\",
         r"\midrule"]
# compute avg per variant for bolding
avg_by_v = {v: ili[ili.variant == v].mse.mean() for v, _, _ in SWEEP}
bestvar = min(avg_by_v, key=avg_by_v.get)
for v, name, has_rho in SWEEP:
    row = ili[ili.variant == v]
    rho = row.rho.mean()
    rho_s = f"{rho:.2f}" if has_rho and pd.notna(rho) else "--"
    cells = []
    for H in hs:
        m = row[row.horizon == H].mse
        cells.append(fmt(float(m.values[0])) if len(m) else "--")
    avg = avg_by_v[v]
    avg_s = (r"\textbf{" + fmt(avg) + "}") if v == bestvar else fmt(avg)
    lines.append(f"{name} & {rho_s} & " + " & ".join(cells) + f" & {avg_s}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "arevin_gate_table.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote arevin_gate_table.tex")

# ---- key numbers for prose ----
print("\n=== KEY NUMBERS (for writer; all from results/nonstationary.csv) ===")
ex = agg[agg.dataset == "exchange_rate"]
print("Exchange: A-RevIN effect (RLinear - A-RevIN-Linear) per H:")
for H in sorted(ex.horizon.unique()):
    r = ex[(ex.variant == "rlinear") & (ex.horizon == H)].mse.values[0]
    a = ex[(ex.variant == "arevin_linear") & (ex.horizon == H)].mse.values[0]
    print(f"   H={H}: {r-a:+.4f}  (rlinear {r:.4f} vs arevin {a:.4f})")
print("ILI: A-RevIN isolated effect (RLinear - A-RevIN-Linear) per H:")
imp = []
for H in hs:
    r = ili[(ili.variant == "rlinear") & (ili.horizon == H)].mse.values[0]
    a = ili[(ili.variant == "arevin_linear") & (ili.horizon == H)].mse.values[0]
    imp.append((r - a) / r * 100)
    print(f"   H={H}: {r-a:+.4f}  ({(r-a)/r*100:+.2f}%)  rlinear {r:.4f} vs arevin {a:.4f}")
fl = ili[ili.variant == "freqlite_arevin"]; rl = ili[ili.variant == "rlinear"]
red = [(rl[rl.horizon==H].mse.values[0]-fl[fl.horizon==H].mse.values[0])/rl[rl.horizon==H].mse.values[0]*100 for H in hs]
print(f"ILI FreqLite vs RLinear reduction per H: {[round(x,1) for x in red]} %  (mean {sum(red)/len(red):.1f}%)")
print(f"ILI learned rho: arevin_linear={ili[ili.variant=='arevin_linear'].rho.mean():.2f}, "
      f"freqlite_def={ili[ili.variant=='freqlite_def'].rho.mean():.2f}, "
      f"freqlite_arevin={ili[ili.variant=='freqlite_arevin'].rho.mean():.2f}, "
      f"forced={ili[ili.variant=='freqlite_forced'].rho.mean():.2f}")
