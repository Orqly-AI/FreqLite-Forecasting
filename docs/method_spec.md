# FreqLite — Method Specification (v1.0)

**Owner:** methodologist · **Status:** authoritative spec for implementation (task #5) and the Method section (task #10).
**Scope:** This document is the single source of truth for the FreqLite model, its math, tensor shapes, training protocol, experiment matrix, baselines, ablations, and 4 GB-feasible hyperparameters. It is written to be *implementable verbatim* and *honest* about what is and is not novel.

---

> **Post-experiment update (authoritative for the final paper).** Two changes
> followed from the experiments: (1) the A-RevIN gate default is now
> `init_rho_logit = 0` (not −4): at −4 the gate is gradient-starved
> (a=b=λ=0 ⇒ ∂L/∂ρ≈0) and never engages, even on non-stationary data; ρ₀=0 lets
> it engage while still recovering RevIN exactly at ρ=0. (2) A-RevIN is framed as
> a **regime-adaptive** normalization rather than an always-on improvement: it
> helps on strongly non-stationary data (ILI, ~5% MSE reduction, monotone in ρ)
> and reduces to RevIN on stationary benchmarks (neutral). The headline result on
> the standard benchmarks is FreqLite's **efficiency + best-lightweight / long-
> lookback win over PatchTST**, not a large accuracy margin over RLinear. See the
> non-stationarity study (`scripts/run_nonstationary.py`) and `paper/`.

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `L` | lookback (input) length, in time steps |
| `H` | forecast horizon (output) length, in time steps |
| `C` | number of variates (channels) in the multivariate series |
| `x ∈ ℝ^{B×L×C}` | input window |
| `y ∈ ℝ^{B×H×C}` | ground-truth horizon |
| `ŷ ∈ ℝ^{B×H×C}` | prediction |
| `K` | number of frequency bands (default `K=2`: low + high) |
| `F = ⌊L/2⌋ + 1` | number of non-redundant rfft bins for a length-`L` real signal |

**Channel-independence (CI).** Like DLinear/RLinear, FreqLite is channel-independent: every variate is processed by the *same* shared weights, and the channel dimension is folded into the batch. We reshape `x: B×L×C → (B·C)×L`, operate on `N = B·C` univariate series of length `L`, and reshape predictions back to `B×H×C`. All math below is written for a single univariate series of length `L` unless a tensor shape says otherwise. CI is the standard strong-baseline setting (Nie et al., 2023; Han et al., 2024) and keeps the parameter count independent of `C`, which is essential for the 4 GB budget and for ECL (`C=321`).

**Real FFT.** We use `torch.fft.rfft`/`irfft` along the time axis. For a real length-`L` signal, `rfft` returns `F = ⌊L/2⌋+1` complex bins. `irfft(·, n=L)` inverts exactly. All FFTs are length-`L` (lookback length); the model never FFTs the horizon.

---

## 1. Architecture overview (forward pass, end to end)

For one univariate series `s ∈ ℝ^{L}` (in practice batched as `(B·C)×L`):

```
s (B·C, L)
  │
  ▼  A-RevIN normalize  (Sec. 3)  → produces s_n (B·C, L) and stats {μ, σ}
  │
  ▼  Learnable frequency decomposition (Sec. 2)
  │     split s_n into K band signals  {s_n^(k)}_{k=1..K}, each (B·C, L)
  │
  ▼  Per-band linear heads (Sec. 4)
  │     each band k:  Wk ∈ ℝ^{H×L}, bk ∈ ℝ^{H}   →  p^(k) = Wk · s_n^(k) + bk   (B·C, H)
  │
  ▼  Recombination (Sec. 4.2)
  │     p = Σ_k p^(k)            (B·C, H)         [identity recombination, default]
  │     (optional learnable gate g ∈ ℝ^K, see 4.2)
  │
  ▼  A-RevIN denormalize (Sec. 3) using {μ, σ} and horizon-adaptive params → ŷ (B·C, H)
  │
  ▼  reshape → (B, H, C)
```

The model has **three** components, mapping 1-to-1 onto the three paper contributions:
1. Learnable frequency decomposition (Sec. 2).
2. A-RevIN — adaptive reversible instance normalization (Sec. 3).
3. Per-band linear heads + recombination, evaluated under a rigorous efficiency study (Sec. 4 + Sec. 7).

---

## 2. Learnable frequency decomposition

### 2.1 Design goal and honest positioning

The learnable frequency split is a **secondary, lightweight design point**, not the headline contribution (A-RevIN is — see Sec. 3.1 and Sec. 12). We position it carefully and modestly.

DLinear decomposes the series into **trend + seasonal** using a *fixed* moving-average kernel (average pooling of fixed window size). That is exactly a fixed low-pass / high-pass split in the time domain. FreqLite makes this split **learnable in the frequency domain** with a smooth, monotone, parameter-light soft mask, rather than a fixed-width box filter, and generalizes from 2 bands to `K` bands. This is a *modest, defensible* generalization of DLinear's decomposition — we explicitly do **not** claim to be "the first frequency-domain linear model" and do not claim to invent frequency-domain forecasting (cf. FEDformer, FreTS, FiLM, and especially **FITS**).

**Distinction from FITS (the closest prior art).** FITS (ICLR 2024) applies a low-pass cutoff in the rFFT domain and forecasts with a single complex linear layer — it **discards** the high-frequency content above the cutoff. Our split is different in kind: it is a **partition of unity** (Sec. 2.4) that keeps *all* frequency content and routes each band to its **own dedicated head**, so the high-frequency band is **modeled** rather than thrown away. That is the one-sentence frequency-prong contribution: a lossless, learnable band routing that retains and separately models high-frequency residual, in contrast to FITS's lossy low-pass truncation. It (a) subsumes DLinear's fixed split as a special case and (b) costs only `2(K−1)` extra scalar parameters.

**Honesty note.** If the ablations (A2, A3, A7) show the spectral split adds little over RLinear/A-RevIN, we will report that plainly and lean the paper on A-RevIN + the efficiency study; we do not inflate the frequency prong.

### 2.2 Soft spectral masks (the learnable filter)

We define `K` soft masks over the `F` rfft frequency bins. Let normalized frequency for bin `f ∈ {0,…,F−1}` be

```
ω_f = f / (F − 1) ∈ [0, 1]        (ω_0 = 0 = DC, ω_{F-1} = 1 = Nyquist)
```

We parameterize `K−1` learnable **cutoffs** and **sharpnesses** and build smooth complementary masks with a sigmoid ladder. For `K=2` (default):

```
learnable params:  c ∈ ℝ  (raw cutoff),  τ ∈ ℝ  (raw sharpness)
cutoff      = σ(c)            ∈ (0,1)        # sigmoid keeps it in-range, init c=0 → 0.5... see init
sharpness   = softplus(τ) + ε ∈ (0,∞)       # ε = 1e-3 for stability

low-pass mask:   m_low(ω_f)  = σ( -sharpness · (ω_f − cutoff) )      ∈ (0,1)
high-pass mask:  m_high(ω_f) = 1 − m_low(ω_f)
```

`σ` is the logistic sigmoid. `m_low` is a smooth monotone-decreasing low-pass filter; `m_high` is its exact complement so the masks form a **partition of unity** (`Σ_k m_k(ω_f) = 1` for every bin). This guarantees the decomposition is *lossless* before the heads (Sec. 2.4).

**General `K` bands.** Use `K−1` ordered cutoffs `0 < c_1 < … < c_{K−1} < 1` (enforce ordering via cumulative softplus: `c_j = Σ_{i≤j} softplus(δ_i) / (1 + Σ softplus(δ))`). Band masks:
```
m_1 = σ(-τ_1(ω−c_1))
m_k = σ(-τ_k(ω−c_k)) − σ(-τ_{k-1}(ω−c_{k-1}))   for 1 < k < K
m_K = 1 − σ(-τ_{K-1}(ω−c_{K-1}))
```
This still partitions unity by construction. **Default and primary reported model is `K=2`.** `K∈{3,4}` is an ablation only.

### 2.3 Masks are real and applied to complex spectrum

The masks `m_k ∈ ℝ^{F}` are **real-valued** and multiply the complex spectrum bin-wise (same scalar applied to real and imaginary parts → zero phase distortion, pure magnitude gating):

```
S       = rfft(s_n, n=L)            # complex, shape (B·C, F)
S^(k)   = m_k ⊙ S                   # broadcast m_k over batch; complex (B·C, F)
s_n^(k) = irfft(S^(k), n=L)         # real,    shape (B·C, L)
```

Masks are buffered/registered as a function of the learnable scalars and recomputed each forward (cheap: `F` evaluations of a sigmoid). They are **independent of `C`** (shared across all channels) and independent of batch.

### 2.4 Key properties (these are what make it defensible)

1. **Lossless partition.** Since `Σ_k m_k = 1` exactly, `Σ_k s_n^(k) = irfft(Σ_k m_k ⊙ S) = irfft(S) = s_n`. The decomposition perfectly reconstructs the normalized input. The heads then learn band-specific mappings; reconstruction loss is zero *before* the heads.
2. **DLinear is (approximately) a special case.** With `K=2`, a hard mask (`sharpness→∞`) at an appropriate cutoff reproduces an ideal low/high split; DLinear's moving-average trend is a (different, time-domain box) low-pass. So FreqLite's decomposition is a strictly richer, learnable family. We will state this carefully — DLinear's box filter is not *exactly* recoverable, but the *trend/residual* inductive bias is.
3. **Differentiable.** `cutoff`, `sharpness` receive gradients through `irfft`, so the split adapts to each dataset's spectral content during training.
4. **Cheap.** `2(K−1)` extra scalar parameters total. FFT/iFFT are `O(L log L)` per series — negligible vs. the `H·L` linear heads.

### 2.5 Initialization

- `K=2`: init `cutoff ≈ 0.25` (raw `c = logit(0.25) ≈ −1.0986`) so the low band starts as roughly the lowest 25% of frequencies (trend/low-season prior), `sharpness ≈ 10` (raw `τ = softplus^{-1}(10) ≈ 10`). These are starting points; they are *learned*. Document final learned values per dataset in results (interpretability bonus for the paper).
- The init must NOT be a degenerate all-pass/all-stop; assert `0.02 < cutoff < 0.98` at init.

---

## 3. A-RevIN — Adaptive Reversible Instance Normalization

### 3.1 Honest positioning vs. RevIN

**A-RevIN is the PRIMARY contribution of FreqLite.** litreview surveyed the literature and found **no published horizon-adaptive / gated reversible instance normalization** — this is the component with the least overlap with prior art and the clearest novelty. The frequency split (Sec. 2) is a secondary lightweight design point; the efficiency study (Sec. 7/8) is the third prong. The paper's framing leads with A-RevIN.

RevIN (Kim et al., 2022) normalizes each instance (per series, per channel) by its own mean/std computed over the lookback, applies an optional affine `(γ, β)`, runs the model, then denormalizes by inverting the affine and re-applying the stored mean/std. RLinear = RevIN + a single linear layer.

**The well-known weakness:** RevIN denormalizes the *entire horizon* with the *single* lookback mean/std `(μ, σ)`. Under distribution shift / non-stationarity, the horizon's true level and scale drift away from the lookback statistics, and the error grows with horizon distance. FreqLite's A-RevIN addresses this with a **horizon-adaptive, learnable correction to the denormalization statistics**, while keeping exact reversibility in the no-shift limit. This is the second genuine (and modest) contribution. We position it as: *RevIN is a special case of A-RevIN* (set all adaptive params to their identity values).

### 3.2 Normalization (forward, identical to RevIN)

Per univariate series `s ∈ ℝ^L` (batched `(B·C)×L`), over the time axis:

```
μ = mean(s)                 # scalar per series → (B·C, 1)
σ = sqrt(var(s) + ε)        # ε = 1e-5,            (B·C, 1)
s_centered = (s − μ) / σ
s_n = γ ⊙ s_centered + β    # γ, β ∈ ℝ (shared scalars, learnable; standard RevIN affine)
```

`γ, β` are **scalar** affine params shared across all series/channels (standard RevIN uses per-channel affine of size `C`; we use shared scalars to keep CI and `C`-independence — document this choice). `ε = 1e-5`.

### 3.3 Adaptive denormalization (the new part)

Standard RevIN denorm: `ŷ_t = σ · (p_t − β)/γ + μ` for every horizon step `t`.

A-RevIN introduces a **horizon-indexed scale and shift correction** that is learnable and depends on the horizon position `t ∈ {1,…,H}`, plus a **gate** driven by an observed non-stationarity signal of the input. Two learnable vectors of length `H`:

```
a ∈ ℝ^H   (per-step log-scale correction),  init a = 0  → exp(0)=1 (identity)
b ∈ ℝ^H   (per-step shift correction, in units of σ), init b = 0 (identity)
```

and a scalar non-stationarity feature computed per input series (detached statistics of the lookback, *not* learnable, just a feature):

```
drift = (mean(s[:, L/2:]) − mean(s[:, :L/2])) / σ      # (B·C, 1): recent-vs-early level drift, in σ units
trend_strength = |drift|                                # magnitude of drift
```

A **gate** `ρ ∈ [0,1]` (scalar learnable, via sigmoid of raw `r`, init `r` s.t. `ρ≈0`) blends identity denorm with the adaptive correction, so the model can *learn to stay at RevIN* if adaptation does not help:

```
For each horizon step t = 1..H:
  scale_t = exp( ρ · a_t )                       # multiplicative, ≈1 when ρ≈0 or a≈0
  shift_t = ρ · ( b_t · σ  +  λ_t · drift · σ )  # additive, in data units
  ŷ_t = scale_t · [ σ · (p_t − β) / γ ] + μ + shift_t
```

where `λ ∈ ℝ^H` is a third learnable per-step vector (init 0) that lets the model propagate the *observed* lookback drift into the horizon proportionally to step distance (a horizon-aware extrapolation of level). The standard expectation `λ_t` learns to grow with `t` for trending datasets (ETTm trends) and stay ≈0 for stationary ones.

**Reversibility / identity guarantee.** When `ρ = 0` (or `a=b=λ=0`), A-RevIN reduces *exactly* to RevIN denorm `ŷ_t = σ(p_t−β)/γ + μ`. Thus RevIN ⊂ A-RevIN, and training can never be worse than RevIN at the normalization layer up to optimization — a clean, honest ablation story.

### 3.4 Parameter count of A-RevIN

`γ, β` (2 scalars) `+ a, b, λ` (`3H` params) `+ ρ` (1 scalar) `= 3H + 3`. For `H=720` that is 2163 params — negligible. All shared across channels (CI).

### 3.5 What we explicitly do NOT claim

- We do not claim A-RevIN models full distributional shift (no per-step variance modeling beyond the scalar `exp(a_t)`); it is a *first-order, horizon-aware level/scale correction*.
- The `drift` feature is a simple two-half mean difference, not a learned encoder. This keeps it linear-model-cheap and interpretable. Stationarity-aware methods (Non-stationary Transformers, SAN) are heavier; we cite them and position A-RevIN as the lightweight counterpart.

---

## 4. Per-band linear heads and recombination

### 4.1 Heads

One independent linear head per band, mapping lookback `L` → horizon `H`:

```
for k in 1..K:
    p^(k) = s_n^(k) @ W_k^T + b_k          # s_n^(k): (B·C, L); W_k: (H, L); b_k: (H); p^(k): (B·C, H)
```

Each head is a single `nn.Linear(L, H, bias=True)`. Parameter count per head: `H·L + H`. Total heads: `K·(H·L + H)`.

### 4.2 Recombination

**Default (primary model): identity sum.**
```
p = Σ_k p^(k)                              # (B·C, H)
```
Rationale: because the bands sum to the input (Sec. 2.4) and each head is linear, the identity sum keeps the model in the same hypothesis class as a single linear head *plus* the ability to apply different linear maps to different frequency content — which is the entire point. With a *fixed* identity recombination and *fixed* all-pass masks and `K=1`, FreqLite degenerates exactly to RLinear (RevIN+Linear). This nesting is the backbone of the ablation table.

**Optional ablation: learnable band gates.**
```
g ∈ ℝ^K (learnable), softmax or raw;   p = Σ_k g_k · p^(k)
```
Report as an ablation; not in the primary model unless it helps and we can explain why.

### 4.3 Total parameter count (primary `K=2`)

```
heads:    2 · (H·L + H)
decomp:   2(K−1) = 2 scalars
A-RevIN:  3H + 3
TOTAL ≈ 2HL + 5H + 5
```
Example `L=96, H=96`: ≈ `2·96·96 + 5·96 + 5 ≈ 18,432 + 485 ≈ 18.9k` params. `L=336, H=720`: ≈ `2·336·720 + 3600 ≈ 487.5k` params. Compare DLinear (`2HL`), RLinear (`HL`). FreqLite is ~2× a single linear head — still tiny, easily 4 GB-feasible. **This 2× is the honest efficiency cost we must report**; the claim is competitive/better accuracy at still-linear scale, not free.

---

## 5. Loss, optimizer, schedule, seeds (training protocol)

Matches the standard long-term forecasting protocol (Informer/Autoformer/DLinear codebase conventions) so numbers are comparable to published baselines.

- **Loss:** MSE on the horizon in normalized... NO — to match convention, loss is MSE on the **de-normalized** prediction vs. raw target in the *dataset-normalized* space. Concretely: data is z-scored using **train-split statistics** (global per-channel mean/std computed on the training portion only) before windowing — this is the standard LTSF normalization that makes MSE/MAE comparable across papers. A-RevIN/RevIN then operate as an *instance* normalization on top of that, and the loss is computed in the train-z-scored space:
  ```
  loss = MSE(ŷ, y)         # both in train-z-scored units, the LTSF convention
  ```
- **Metrics:** MSE and MAE, computed in the same train-z-scored space, averaged over all horizon steps and channels (standard). Report per `(dataset, H)` and the mean.
- **Optimizer:** Adam, `lr = 1e-3` (heads), with the decomposition/A-RevIN scalars in the same param group at `lr = 1e-3` (they are few; no separate group needed initially — if cutoffs train too slowly, add a 10× group, document it).
- **Weight decay:** `0` (linear LTSF models conventionally use none; document).
- **Batch size:** `32` (ETT, Weather). `16` for ECL (`C=321`) to fit 4 GB — see Sec. 8.
- **Epochs:** max `20` with **early stopping**, patience `3` on validation loss (DLinear convention). Save best-val checkpoint.
- **LR schedule:** step decay — multiply lr by `0.5` each epoch after epoch 1 (the "type1" schedule from the DLinear/Autoformer repo), or cosine; **use type1 step decay as default** for comparability. Document exact schedule in config.
- **Gradient clipping:** clip global norm to `1.0` (cheap insurance; document).
- **Seeds:** run each `(dataset, H, model)` cell with seeds `{2021, 2022, 2023}` (≥3). Set `torch`, `numpy`, `random`, and `torch.cuda` seeds; enable `torch.use_deterministic_algorithms(True)` where feasible and `cudnn.deterministic=True, cudnn.benchmark=False`. Report **mean ± std** over seeds in every table.
- **Precision:** fp32 (these models are tiny; fp32 is safe in 4 GB and avoids fp16 reproducibility noise). No AMP needed.
- **Validation/test:** standard sliding-window evaluation over val/test splits; no leakage (train stats only).

---

## 6. Experiment matrix

### 6.1 Datasets (and splits)

| Dataset | `C` | Freq | Split (train/val/test) | Notes |
|---|---|---|---|---|
| ETTh1 | 7 | hourly | 12/4/4 months (standard) | primary |
| ETTh2 | 7 | hourly | 12/4/4 months | primary |
| ETTm1 | 7 | 15-min | 12/4/4 months | primary |
| ETTm2 | 7 | 15-min | 12/4/4 months | primary |
| Weather | 21 | 10-min | 70/10/20 % | primary |
| Electricity (ECL) | 321 | hourly | 70/10/20 % | **memory-permitting**; CI + bs=16. If OOM, report as limitation. |

Use the canonical ETT split (`12*30*24` train, `4*30*24` val/test for ETTh; `*4` for ETTm). z-score with train statistics per channel.

### 6.2 Horizons and lookback

- **Horizons `H ∈ {96, 192, 336, 720}`** (standard LTSF).
- **Lookback `L`:** primary `L = 336` (strong setting for linear models per RLinear/PatchTST). Also run `L = 96` as a secondary lookback for the main comparison table footnote / a sensitivity figure. **Report `L=336` as the headline; include `L=96` for completeness.**

### 6.3 Full grid (primary)

`6 datasets × 4 horizons × 3 seeds = 72 runs per model` at `L=336`.
Models in the main table: FreqLite, DLinear, RLinear, NLinear, FITS, (PatchTST-small if it fits). Naive (repeat-last) computed analytically (no training).

Total trainable cells ≈ `72 × (5–6 models)` ≈ 360–432 short runs. All are minutes-scale on the 3050 Ti (linear models); PatchTST-small is the only heavier one (Sec. 8).

---

## 7. Baselines

| Baseline | Definition | Why included |
|---|---|---|
| **Naive / repeat-last** | `ŷ_t = x_L` (last observed value, broadcast over `H`), per channel | sanity floor; no params |
| **NLinear** | subtract last value, single `Linear(L,H)`, add last value back | simplest distribution-shift-robust linear |
| **DLinear** | moving-avg trend/seasonal decomposition + two `Linear(L,H)` heads, summed | the decomposition baseline FreqLite generalizes |
| **RLinear** | RevIN + single `Linear(L,H)` | the normalization baseline A-RevIN generalizes |
| **FITS** (Xu, Zeng, Xu, ICLR 2024; arXiv 2307.03756) | RIN/RevIN + low-pass cutoff in the rFFT domain + a single **complex** linear layer mapping the kept low-freq bins to the `(L+H)`-length spectrum + irFFT; CI; ~10k params | **closest prior art to our frequency prong** — must be a baseline. FITS *discards* high-freq via low-pass; FreqLite instead *models* it with a dedicated head. Fits 4 GB. |
| **PatchTST-small** (memory-permitting) | patch length 16, stride 8, `d_model=64`, `n_heads=4`, `e_layers=2`, dropout 0.2, CI | strong transformer ref; tiny config for 4 GB |

We re-implement/configure all baselines in-repo and **run them ourselves** (no copied numbers) so comparisons are on identical splits, normalization, seeds, and hardware. If our reproduced baseline numbers differ from published, we report ours and note the protocol.

PatchTST is explicitly **conditional**: if `(ECL, H=720, L=336)` PatchTST-small OOMs at bs that gives stable training, we (a) reduce batch / use grad accumulation, else (b) drop PatchTST for that cell and note it. Linear baselines never OOM.

---

## 8. 4 GB feasibility and memory budget

The dominant activation memory is the heads' input/output and the linear weight matrices. For CI, effective batch over series is `B·C`.

- Worst linear case: `ECL, C=321, bs=16, L=336, H=720`. Effective rows `= 16·321 = 5136`. Largest activation `p^(k): 5136×720` fp32 ≈ 14.8 MB per band; a handful of such tensors + weights (`720×336` ≈ 0.97 MB each) → well under 4 GB. **Linear FreqLite and all linear baselines fit trivially.** The FFT intermediates are `5136×F` complex ≈ small.
- For ETT/Weather (`C ≤ 21`), bs=32, everything is sub-100 MB.
- **PatchTST-small** is the only memory risk. Mitigations in priority order: (1) bs=16, (2) bs=8 + grad-accum ×2 to keep effective bs=16, (3) reduce `d_model` 64→32, (4) drop the offending cell with a documented limitation. Engineer must `torch.cuda.max_memory_allocated()` log every run; abort+downshift if > 3.5 GB.
- **Always** call `torch.cuda.reset_peak_memory_stats()` at run start and record **peak GPU mem** per run for the efficiency table — it is a reported metric, not just a guardrail.

### 8.1 Efficiency metrics to record (every run)

`params` (count), `FLOPs` (per forward, via `ptflops`/`thop` or analytic `2HL` style count — analytic preferred for exactness on linear models), `train time / epoch` (wall-clock, seconds), `peak GPU mem` (MB). These populate the headline efficiency table (contribution 3).

---

## 9. Ablation plan (isolates each contribution)

All ablations on a representative subset to control cost: **{ETTh1, ETTm2, Weather} × H∈{96, 720} × 3 seeds**, `L=336`. (Two horizons capture short/long behavior.)

| # | Variant | What it tests | Expected/honest reading |
|---|---|---|---|
| A0 | **Full FreqLite** (`K=2`, learnable masks, A-RevIN) | reference | — |
| A1 | − A-RevIN → plain RevIN (`ρ=0` frozen) | value of adaptive denorm | isolates contribution 2 |
| A2 | − learnable decomposition → DLinear-style fixed MA split, keep A-RevIN | value of learnable spectral split | isolates contribution 1 |
| A3 | `K=1` (no decomposition) + A-RevIN | = "A-RevIN-Linear" | shows decomposition's marginal value |
| A4 | `K=1` + plain RevIN | **= RLinear** (degenerate FreqLite) | sanity: FreqLite ⊇ RLinear |
| A5 | `K∈{3,4}` bands | does more bands help? | likely diminishing returns; report honestly |
| A6 | learnable gate recombination (Sec. 4.2) | gated vs identity sum | report; keep only if it helps |
| A7 | freeze decomposition at init (no grad on cutoff/sharpness) | is *learning* the split (vs a fixed soft split) what matters? | separates "soft mask" from "learned mask" |
| A8 | A-RevIN without `λ` (drift propagation) term | value of observed-drift extrapolation | isolates the most novel A-RevIN piece |

A4 and A3 establish the strict nesting RLinear ⊂ FreqLite, which is the cleanest possible defense of novelty: every added component is an opt-in generalization, ablated independently.

**Interpretability artifact:** log learned `cutoff`, `sharpness`, `ρ`, and the `λ_t` profile per dataset; include a figure (engineer/writer) showing the learned filter shape and the learned horizon-correction profile — strong evidence the components do something meaningful, not just add params.

---

## 10. Default hyperparameters (config-ready)

```yaml
model:
  name: freqlite
  L: 336                # lookback (also run 96)
  H: 96                 # set per experiment {96,192,336,720}
  K: 2                  # bands (primary)
  channel_independent: true
  decomposition:
    init_cutoff: 0.25
    init_sharpness: 10.0
    learnable: true
    mask_eps: 1.0e-3
  arevin:
    affine: true        # gamma,beta
    eps: 1.0e-5
    adaptive: true      # a,b,lambda,rho
    init_rho_logit: -4.0   # rho≈0.018 → starts ~RevIN, learns up
  recombination: sum     # {sum, gate}
train:
  loss: mse
  optimizer: adam
  lr: 1.0e-3
  weight_decay: 0.0
  batch_size: 32         # 16 for ECL
  max_epochs: 20
  patience: 3
  lr_schedule: type1     # halve each epoch after epoch 1
  grad_clip_norm: 1.0
  precision: fp32
  seeds: [2021, 2022, 2023]
  deterministic: true
data:
  normalize: zscore_train_stats
  splits: standard       # ETT 12/4/4 months; Weather/ECL 70/10/20
eval:
  metrics: [mse, mae]
  record: [params, flops, sec_per_epoch, peak_gpu_mem_mb]
```

---

## 11. Feasibility risks and honest caveats (for team-lead / paper limitations)

1. **The method may not beat strong baselines on every cell.** Linear LTSF is a saturated, well-tuned regime; RLinear/PatchTST are very strong. Plausible realistic outcome: FreqLite wins or ties on trending/non-stationary datasets (ETTm1/m2 long horizons, where A-RevIN's drift term helps) and on datasets with separable spectral content, and is roughly on par elsewhere. **We will report all cells truthfully**, including losses, and analyze *where* and *why* via the learned-parameter interpretability figures. The paper's defensible thesis even in a tie is "matches strong baselines at ~linear cost with added robustness on non-stationary horizons + interpretable learned filters" — contribution 3 (efficiency) holds regardless.
2. **A-RevIN's `λ` drift term could overfit / hurt on stationary data.** Mitigated by the `ρ` gate initialized near 0 (starts as RevIN) and ablation A8. If it never helps, we demote it to an ablation and keep the simpler A-RevIN (`a,b` only) as primary — decide empirically.
3. **2× parameter cost vs. a single linear head.** Honest and small; reported. Not a free lunch, but still orders of magnitude below transformers.
4. **PatchTST in 4 GB** is the only OOM risk; mitigation ladder in Sec. 8. Worst case: documented limitation, not a fabricated number.
5. **Determinism vs. FFT:** `torch.fft` + `use_deterministic_algorithms(True)` is fine on these sizes; if a non-deterministic CUDA FFT path appears, fall back to CPU FFT for the (tiny) spectral step or accept the documented small variance — verify during implementation.
6. **z-score-train-stats AND instance RevIN both present:** this double-normalization is standard (RLinear does exactly this) but must be implemented in the right order (dataset z-score at data-loading; A-RevIN inside the model). Engineer: be careful the loss is in the train-z-scored space for comparability.

---

## 12. Summary of what is novel vs. prior art (one-paragraph defense)

FreqLite's **primary** contribution is **A-RevIN**: a horizon-adaptive, gated reversible instance normalization that replaces RevIN's fixed, single-statistic denormalization with a learned per-step scale/shift plus an observed-drift extrapolation term, and that **strictly contains RevIN** (the `ρ=0` special case). litreview found no published horizon-adaptive/gated RevIN, so this is the component with the least overlap with prior art and the clearest novelty. The **secondary** contribution is a lightweight design point on the backbone: the single linear head becomes `K` per-band heads fed by a **learnable, lossless, partition-of-unity spectral decomposition** (`2(K−1)` learned scalars). Unlike **FITS** — the closest prior art, which low-pass-truncates and *discards* high-frequency content — our split keeps all frequency content and **models the high-frequency band with a dedicated head**; it generalizes DLinear's fixed moving-average split rather than claiming to be the first frequency-domain linear model. Each axis is independently ablatable and each degenerates to a known baseline, so the novelty is real and minimal-overclaim: FreqLite ⊇ RLinear ⊇ Linear, with DLinear's decomposition as a fixed-mask special case and FITS as the lossy-low-pass contrast. **If ablations show the spectral split adds little over RLinear/A-RevIN, we report that and lean the paper on A-RevIN + efficiency.** The **third** contribution is empirical — a rigorous accuracy-vs-(params/FLOPs/time/memory) study on a 4 GB GPU with full ablations and learned-parameter interpretability.
