"""Aggregate the distribution-shift stress study into a LaTeX table + figure.

Reads results/shift_stress.csv (written by scripts/run_shift_stress.py) and
emits:
  * results/tables/shift_stress_table.tex — booktabs; rows = dataset x third
    (Near/Mid/Far by input-shift score) plus a 'Far-Near gap' summary row per
    dataset; columns = RLinear / A-RevIN-Linear / FreqLite. MSE averaged over
    seeds, best per row in bold.
  * results/figures/shift_stress.{pdf,png} — MSE relative to RLinear (=1.0)
    vs. shift third, one line per model, one panel per dataset.

Everything comes from the CSV; nothing is fabricated.

Usage:
  .venv\\Scripts\\python.exe scripts\\make_shift_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "shift_stress.csv"
TDIR = ROOT / "results" / "tables"
FDIR = ROOT / "results" / "figures"

MODELS = [("rlinear", "RLinear"),
          ("arevin_linear", "A-RevIN-Linear"),
          ("freqlite", "FreqLite")]
THIRDS = ["Near", "Mid", "Far"]
DISP = {"ETTh1": "ETTh1", "ETTm2": "ETTm2", "weather": "Weather",
        "exchange_rate": "Exchange"}


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}; run scripts/run_shift_stress.py first")
        return 1
    d = pd.read_csv(CSV)
    TDIR.mkdir(parents=True, exist_ok=True)
    FDIR.mkdir(parents=True, exist_ok=True)

    # mean over seeds -> (dataset, model, third)
    agg = (d.groupby(["dataset", "model", "third"])
             .agg(mse=("mse", "mean"), mse_std=("mse", "std"),
                  shift=("shift_mean", "mean"))
             .reset_index())
    datasets = [ds for ds in DISP if ds in set(agg.dataset)]

    def cell(ds, m, t):
        r = agg[(agg.dataset == ds) & (agg.model == m) & (agg.third == t)]
        return float(r.mse.values[0]) if len(r) else None

    # ---- LaTeX table ----
    lines = [r"\begin{tabular}{ll" + "c" * len(MODELS) + "}", r"\toprule",
             "Dataset & Shift third & "
             + " & ".join(n for _, n in MODELS) + r" \\", r"\midrule"]
    for ds in datasets:
        for i, t in enumerate(THIRDS):
            vals = {m: cell(ds, m, t) for m, _ in MODELS}
            vals = {m: v for m, v in vals.items() if v is not None}
            best = min(vals, key=vals.get) if vals else None
            cells = []
            for m, _ in MODELS:
                if m in vals:
                    s = f"{vals[m]:.3f}"
                    cells.append(r"\textbf{" + s + "}" if m == best else s)
                else:
                    cells.append("--")
            name = DISP[ds] if i == 0 else ""
            lines.append(f"{name} & {t} & " + " & ".join(cells) + r" \\")
        # Far-Near gap summary (absolute MSE increase from Near to Far third;
        # smaller = more robust to input-distribution shift)
        gaps = {}
        for m, _ in MODELS:
            f, nr = cell(ds, m, "Far"), cell(ds, m, "Near")
            if f is not None and nr is not None:
                gaps[m] = f - nr
        bestg = min(gaps, key=gaps.get) if gaps else None
        cells = []
        for m, _ in MODELS:
            if m in gaps:
                s = f"{gaps[m]:+.3f}"
                cells.append(r"\textbf{" + s + "}" if m == bestg else s)
            else:
                cells.append("--")
        lines.append(r" & \emph{Far$-$Near gap} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    out_tex = TDIR / "shift_stress_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_tex}")

    # ---- figure: MSE relative to RLinear per third, one panel per dataset ----
    ncol = 2
    nrow = (len(datasets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             squeeze=False, sharex=True)
    styles = {"rlinear": dict(color="0.45", marker="o", ls="--"),
              "arevin_linear": dict(color="tab:blue", marker="s", ls="-"),
              "freqlite": dict(color="tab:red", marker="^", ls="-")}
    for ax_i, ds in enumerate(datasets):
        ax = axes[ax_i // ncol][ax_i % ncol]
        ref = [cell(ds, "rlinear", t) for t in THIRDS]
        for m, name in MODELS:
            ys = []
            for t, r in zip(THIRDS, ref):
                v = cell(ds, m, t)
                ys.append(v / r if (v is not None and r) else float("nan"))
            ax.plot(THIRDS, ys, label=name, lw=1.4, ms=4, **styles[m])
        ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(DISP[ds], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("MSE / RLinear", fontsize=8)
    # hide unused panels
    for j in range(len(datasets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle("Test MSE by input-distribution-shift third", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    for ext in ("pdf", "png"):
        p = FDIR / f"shift_stress.{ext}"
        fig.savefig(p, dpi=200)
        print(f"wrote {p}")

    # ---- key numbers for prose ----
    print("\n=== Far-Near MSE gap (mean over seeds; smaller = more robust) ===")
    for ds in datasets:
        parts = []
        for m, name in MODELS:
            f, nr = cell(ds, m, "Far"), cell(ds, m, "Near")
            if f is not None and nr is not None:
                parts.append(f"{name}: {f - nr:+.4f}")
        print(f"  {DISP[ds]:9s} " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
