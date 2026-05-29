# FreqLite

> **A lightweight frequency-decomposed linear model for long-term time-series forecasting — matches/beats a PatchTST Transformer at ~4× fewer parameters, on a single 4 GB laptop GPU.**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c.svg)

Official implementation of *"FreqLite: A Lightweight Frequency-Decomposed Linear
Model with Adaptive Reversible Normalization for Robust Long-Term Time-Series
Forecasting."* FreqLite couples a learnable, lossless spectral decomposition with
**A-RevIN**, a regime-adaptive reversible normalization, and is built to train and
evaluate entirely on a 4 GB laptop GPU (RTX 3050 Ti).

**Highlights**
- 🏆 Best lightweight model on the standard LTSF benchmarks; at long lookback ($L{=}336$) it beats PatchTST (avg MSE **0.3244 vs 0.3587**) at **~4× fewer params**, **~2× less memory**, **~2× faster** — gains statistically significant (paired Wilcoxon $p\approx10^{-6}$).
- 🔁 **A-RevIN** engages under non-stationarity (ILI: up to ~5% MSE reduction; confirmed on a controlled synthetic drift sweep) and reduces exactly to RevIN otherwise.
- 🔬 Fully reproducible: fixed seeds, pinned deps, every number regenerable by a script; runs on commodity hardware.

This repository contains the **fully runnable code** behind the paper: model,
baselines, data pipeline, training harness, experiment/ablation runners, and the
scripts that regenerate every number, table, and figure in the manuscript.

## 1. Environment

* Python **3.12.10**. A virtual environment lives at `.venv`.
* PyTorch is a CUDA 12.4 build. Install **torch first**, then the rest:

```powershell
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py      # verifies CUDA + all imports
```

`check_env.py` confirms `torch.cuda.is_available()` and prints the device / VRAM.
On the target machine this reports: **NVIDIA GeForce RTX 3050 Ti Laptop GPU,
4.29 GB, CUDA available = True.**

## 2. Data

```powershell
.venv\Scripts\python.exe scripts\download_data.py            # ETTh1/2, ETTm1/2, Weather
.venv\Scripts\python.exe scripts\download_data.py --with-ecl # + Electricity (large)
```

Canonical wide-format LTSF CSVs (column `date` + channels), byte-identical to the
Autoformer/DLinear release, downloaded from the public AutonLab Timeseries-PILE
mirror. SHA-256 of each file is written to `data/checksums.txt`. Shapes match the
literature exactly (ETTh 17420 rows, ETTm 69680, Weather 52696).

Splits follow the standard protocol: ETT = 12/4/4 months; Weather/ECL = 70/10/20 %.
Per-channel z-score uses **train-split statistics only**; metrics are reported in
that train-standardized space (the LTSF convention).

## 3. Smoke test (proves the pipeline fits 4 GB)

```powershell
.venv\Scripts\python.exe scripts\smoke_test.py
```

Verifies: loaders/shapes; all models forward-pass; the FreqLite decomposition is a
lossless partition of unity; a short train loop runs with **peak GPU memory well
under 3.5 GB**; FreqLite(K=1, plain RevIN) is numerically identical to RLinear.

## 4. Reproduce the experiments

All runs are deterministic (fixed seeds {2021,2022,2023}, cuDNN deterministic,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`). Runners are **resumable** — a cell already in
the output CSV is skipped, so an interrupted run can simply be re-invoked.

```powershell
# Main grid: 7 models x 5 datasets x 4 horizons x 3 seeds at lookback L=336
.venv\Scripts\python.exe scripts\run_experiments.py --config configs\default.yaml --out results\main_results.csv

# Secondary lookback L=96 (completeness)
.venv\Scripts\python.exe scripts\run_experiments.py --config configs\default.yaml --lookback 96 --out results\main_results.csv

# Ablations (subset grid, isolates each contribution; method_spec Sec. 9)
.venv\Scripts\python.exe scripts\run_ablations.py --config configs\default.yaml --out results\ablations.csv

# Non-stationarity study (A-RevIN analysis): exchange_rate + ILI, gate sweep
.venv\Scripts\python.exe scripts\download_data.py --nonstationary   # exchange_rate, national_illness
.venv\Scripts\python.exe scripts\run_nonstationary.py               # -> results\nonstationary.csv
.venv\Scripts\python.exe scripts\make_nonstationary_table.py        # -> nonstationary_table.tex, arevin_gate_table.tex
```

Useful flags: `--models`, `--datasets`, `--horizons`, `--seeds`, `--max-epochs`,
`--mem-abort-mb` (warns if a cell exceeds the memory guard).

The non-stationarity runner compares RLinear, DLinear, FITS, and four A-RevIN gate
settings (trapped `rho0=-4`, default `rho0=0`, forced `rho~=1`, and a K=1
"A-RevIN-Linear" that isolates the normalization from the decomposition). It logs
the learned gate `rho` per cell, which is what reveals the gradient-trap and the
monotonic ILI improvement reported in the paper.

## 5. Tables and figures

```powershell
.venv\Scripts\python.exe scripts\make_tables.py    # booktabs LaTeX + aggregated CSV
.venv\Scripts\python.exe scripts\make_figures.py   # PDF/PNG figures + learned_params.json
```

Outputs land in `results/`:

* `main_results.csv` — one row per (model, dataset, L, H, seed): test MSE/MAE,
  val MSE, params, analytic FLOPs/series, sec/epoch, peak GPU mem.
* `main_results_agg.csv` — mean/std over seeds per cell.
* `ablations.csv` — ablation variants A0–A8.
* `tables/main_table_L*.tex`, `tables/efficiency_table_L*.tex`, `tables/ablation_table.tex`.
* `figures/accuracy_vs_params.{pdf,png}`, `figures/learned_filter.*`, `figures/arevin_profile.*`.
* `learned_params.json` — learned cutoff/sharpness/rho/lambda profiles (interpretability).

## 6. Models

| Model | Notes |
|---|---|
| Naive | repeat-last sanity floor (no params) |
| NLinear | subtract-last + Linear (Zeng et al. 2023) |
| DLinear | moving-avg decomposition + two linear heads |
| RLinear | RevIN + Linear (A-RevIN's base case) |
| FITS | RIN + low-pass rFFT cutoff + complex Linear + irFFT (Xu et al., ICLR 2024) |
| PatchTST* | small CI transformer; 4 GB-feasible config |
| **FreqLite** | learnable spectral decomposition + per-band linear heads + **A-RevIN** |

All baselines are re-implemented in-repo and **run by us** on identical splits,
normalization, seeds, and hardware — no copied numbers.

## 7. Repository layout

```
src/        model code (FreqLite + baselines), data loaders, training/eval engine
configs/    experiment configs (default.yaml)
scripts/    check_env, download_data, smoke_test, run_experiments, run_ablations,
            make_tables, make_figures
data/       downloaded CSVs + checksums.txt
results/    metrics CSVs, aggregated CSVs, tables/, figures/
docs/       method_spec.md, related_work.md
paper/      LaTeX manuscript
logs/       run logs
```

## 8. Reproducibility notes

* Every reported number is regenerable by the scripts above from the pinned
  `requirements.txt` and fixed seeds; two identical invocations produce
  bit-identical metrics (verified).
* fp32 throughout (models are tiny; avoids fp16 noise).
* No fabricated results — if a model loses on a cell, the CSV/table shows it.

## 9. Citation

If you use this code or its results, please cite:

```bibtex
@article{baig2026freqlite,
  title   = {FreqLite: A Lightweight Frequency-Decomposed Linear Model with
             Adaptive Reversible Normalization for Robust Long-Term
             Time-Series Forecasting},
  author  = {Baig, Mirza Samad Ahmed and Gillani, Syeda Anshrah},
  year    = {2026},
  note    = {Code: https://github.com/Orqly-AI/FreqLite-Forecasting}
}
```

See `CITATION.cff` for machine-readable metadata.

## License

Released under the [MIT License](LICENSE) © 2026 Mirza Samad Ahmed Baig and
Syeda Anshrah Gillani (Orqly-AI).
