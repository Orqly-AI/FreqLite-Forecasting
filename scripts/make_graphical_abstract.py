"""Build an Elsevier-sized graphical abstract (landscape, single image):
left = simplified FreqLite pipeline; right = accuracy-vs-params trade-off (real
numbers from results/main_results.csv, L=336). Saves
results/figures/graphical_abstract.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results" / "figures"; F.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "results" / "main_results.csv")
d = df[df.lookback_label == "L336"]
g = d[d.model != "naive"].groupby("model").agg(
    mse=("test_mse", "mean"), params=("params", "median")).reset_index()

NAMES = {"nlinear": "NLinear", "dlinear": "DLinear", "rlinear": "RLinear",
         "fits": "FITS", "freqlite": "FreqLite", "patchtst": "PatchTST"}

fig = plt.figure(figsize=(13.0, 5.0))            # ~2600x1000 px @200dpi (Elsevier OK)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)

# ---------- left: simplified pipeline ----------
axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
axL.set_xlim(0, 10); axL.set_ylim(0, 10)


def box(x, y, w, h, text, fc):
    axL.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                 linewidth=1.2, edgecolor="#333", facecolor=fc))
    axL.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10.5)


def arrow(x1, y1, x2, y2):
    axL.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                  mutation_scale=14, lw=1.4, color="#333"))


axL.text(5, 9.4, "FreqLite", ha="center", fontsize=15, fontweight="bold")
box(0.2, 6.4, 1.7, 1.2, r"$\mathbf{x}$", "#eef3ff")
box(2.3, 6.4, 2.0, 1.2, "A-RevIN\nnormalize", "#ffe6c7")
box(4.7, 6.4, 2.1, 1.2, "freq. split\n(low / high)", "#eef3ff")
box(3.5, 3.9, 1.9, 1.1, "Linear\n(low)", "#d8f0d8")
box(5.7, 3.9, 1.9, 1.1, "Linear\n(high)", "#d8f0d8")
box(7.4, 5.0, 1.9, 1.2, r"$\Sigma$ + A-RevIN" + "\ndenormalize", "#ffe6c7")
box(7.6, 7.2, 1.5, 1.0, r"$\hat{\mathbf{y}}$", "#eef3ff")
arrow(1.9, 7.0, 2.3, 7.0)
arrow(4.3, 7.0, 4.7, 7.0)
arrow(5.5, 6.4, 4.6, 5.0)   # split -> low
arrow(6.0, 6.4, 6.6, 5.0)   # split -> high
arrow(5.4, 4.45, 7.4, 5.4)  # low -> sum
arrow(6.6, 4.45, 7.4, 5.6)  # high -> sum
arrow(8.35, 6.2, 8.35, 7.2)  # denorm -> y
axL.text(5, 2.6, "Learnable, lossless spectral split + per-band linear heads;",
         ha="center", fontsize=9.5)
axL.text(5, 2.0, "A-RevIN adapts to non-stationarity, else reduces to RevIN.",
         ha="center", fontsize=9.5)

# ---------- right: accuracy vs params ----------
axR = fig.add_subplot(gs[0, 1])
# per-model label placement (avoid clipping at the right edge)
OFFS = {"patchtst": (-8, -12, "right"), "dlinear": (-8, 8, "right"),
        "freqlite": (10, 2, "left"), "fits": (8, -2, "left"),
        "nlinear": (8, 4, "left"), "rlinear": (8, -10, "left")}
for _, r in g.iterrows():
    is_fl = r.model == "freqlite"
    axR.scatter(r.params, r.mse, s=180 if is_fl else 90,
                c="#d62728" if is_fl else "#1f77b4",
                marker="*" if is_fl else "o", zorder=3,
                edgecolor="k", linewidth=0.6)
    dx, dy, ha = OFFS.get(r.model, (6, -4, "left"))
    axR.annotate(NAMES[r.model], (r.params, r.mse), textcoords="offset points",
                 xytext=(dx, dy), ha=ha, fontsize=9.5,
                 fontweight="bold" if is_fl else "normal")
axR.set_xscale("log")
axR.margins(x=0.18)
axR.set_xlabel("# parameters (log scale)", fontsize=11)
axR.set_ylabel("avg. test MSE  (lower is better)", fontsize=11)
axR.set_title("Accuracy vs. model size  ($L{=}336$)", fontsize=12)
axR.grid(True, ls=":", alpha=0.5)

fig.suptitle("Transformer-level long-horizon accuracy at ~4x fewer parameters "
             "— on a single 4 GB laptop GPU",
             y=0.05, fontsize=11.5, fontweight="bold")
fig.subplots_adjust(top=0.92, bottom=0.18, left=0.02, right=0.94)
fig.savefig(F / "graphical_abstract.pdf")
fig.savefig(F / "graphical_abstract.png", dpi=200)
print("wrote results/figures/graphical_abstract.{pdf,png}")
print(g.assign(name=g.model.map(NAMES)).round(4).to_string(index=False))
