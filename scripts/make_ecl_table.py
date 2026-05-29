"""Emit results/tables/ecl_table.tex from results/ecl_results.csv (Electricity,
321 channels, L=336). Compact MSE-per-horizon table; best non-Naive in bold.
Run after run_experiments.py finishes the ECL grid."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "ecl_results.csv"
T = ROOT / "results" / "tables"; T.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(src)
g = d.groupby(["model", "horizon"]).test_mse.mean().reset_index()
piv = g.pivot(index="horizon", columns="model", values="test_mse")

DISP = [("nlinear", "NLinear"), ("dlinear", "DLinear"), ("rlinear", "RLinear"),
        ("fits", "FITS"), ("freqlite", "FreqLite")]
present = [(m, n) for m, n in DISP if m in piv.columns]
Hs = sorted(piv.index)

lines = [r"\begin{tabular}{l" + "c" * len(present) + "}", r"\toprule",
         "$H$ & " + " & ".join(n for _, n in present) + r" \\", r"\midrule"]
for H in Hs:
    row = {m: piv.loc[H, m] for m, _ in present if not pd.isna(piv.loc[H, m])}
    best = min(row, key=row.get) if row else None
    cells = []
    for m, _ in present:
        if m in row:
            s = f"{row[m]:.3f}"
            cells.append(r"\textbf{" + s + "}" if m == best else s)
        else:
            cells.append("--")
    lines.append(f"{H} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "ecl_table.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote results/tables/ecl_table.tex")
print(piv[[m for m, _ in present]].round(4).to_string())
avg = piv[[m for m, _ in present]].mean().sort_values()
print("\nECL mean MSE per model:\n" + avg.round(4).to_string())
