# Related Work and Positioning of FreqLite

Owner: `litreview`. This document is the source for the manuscript Related Work
section (Task #9, owned by `writer`). Citation keys below match `paper/refs.bib`.
Every citation was verified against the official venue / arXiv (authors, title,
venue, year, pages).

FreqLite has three contribution prongs (see `method_spec.md`):

1. **Frequency-decomposed linear heads** — a learnable spectral filter splits
   the input into a low-frequency band (trend/seasonal, predictable) and a
   high-frequency band (residual/noise); each band is forecast by a dedicated
   lightweight linear head, then recombined.
2. **Adaptive Reversible Instance Normalization (A-RevIN)** — a learnable
   extension of RevIN that adapts de/normalization to non-stationarity across
   the forecast horizon (horizon-aware / gated).
3. **Rigorous accuracy-vs-efficiency study** — competitive long-horizon
   forecasting at a fraction of params/FLOPs/runtime on a 4 GB laptop GPU, with
   ablations isolating each component.

This document surveys the literature each prong builds on and positions FreqLite
honestly against the closest prior work. **Section 8 is a candid novelty-overlap
assessment** — read it first if you only have time for one section.

---

## 1. Linear forecasters

The "LTSF-Linear" family `[zeng2023dlinear]` ("Are Transformers Effective for
Time Series Forecasting?", AAAI 2023) showed that embarrassingly simple linear
models match or beat elaborate Transformer forecasters on standard long-term
benchmarks. Three variants are relevant baselines for us:

- **Linear**: a single linear layer mapping a length-`L` lookback to a length-`H`
  horizon, applied per channel (channel-independent).
- **NLinear**: subtracts the last value of the lookback before the linear map and
  adds it back afterwards — a one-line normalization that handles simple
  distribution shift between train and test.
- **DLinear**: decomposes the series into a moving-average **trend** and a
  **seasonal/remainder** component (additive series decomposition borrowed from
  Autoformer), forecasts each with its own linear layer, and sums them.

`[li2023rlinear]` ("Revisiting Long-term Time Series Forecasting: An
Investigation on Linear Mapping", arXiv 2023) analyzes *why* linear maps work,
and introduces **RLinear** = RevIN + a single linear layer. Its key findings are
that (i) the linear mapping is what captures periodicity, (ii) RevIN and channel
independence (CI) are the two ingredients that make linear models robust, and
(iii) a single linear layer is often enough.

**Positioning.** DLinear and RLinear are FreqLite's direct competitors and
primary baselines. The crucial conceptual contrast is the **domain of
decomposition**:

- DLinear decomposes in the **time domain** via a fixed moving-average kernel
  into trend + remainder. The split is a low-pass *smoothing* operation; the
  "remainder" is everything the moving average misses and is forecast by an
  unconstrained linear head.
- FreqLite decomposes in the **frequency domain** via a *learnable* spectral
  filter into low- and high-frequency bands. A moving average is one particular
  fixed low-pass filter; FreqLite generalizes this by letting the model learn the
  band split (cutoff/soft mask) end-to-end, and by giving each band its own
  lightweight linear head.

Against RLinear, FreqLite shares the RevIN+linear backbone but (a) replaces the
single linear map with a two-band frequency-decomposed pair of heads and
(b) replaces vanilla RevIN with the horizon-adaptive A-RevIN (Section 2). The
ablation that drops the frequency split and the A-RevIN gate should recover an
RLinear-like model — this is the controlled comparison that isolates our
contribution.

---

## 2. Non-stationarity and reversible normalization

Real time series are non-stationary: the mean/variance drift between the lookback
window and the forecast horizon, and between train and test. Two lines of work
address this.

**RevIN** `[kim2022revin]` (ICLR 2022) is a symmetric, instance-wise
normalize-then-denormalize wrapper with a learnable affine transform. At the
input it removes each instance's mean and standard deviation (computed per
channel over the lookback); at the output it restores them. This removes
non-stationary statistics from the model's internal representation and re-injects
them at the end, and is now a near-universal plug-in (RLinear, PatchTST, etc.).

**Non-stationary Transformers** `[liu2022nonstationary]` (NeurIPS 2022) make a
subtler point: naive stationarization can *over-stationarize*, erasing the
non-stationary signal the model needs to distinguish series. They pair Series
Stationarization with a **De-stationary Attention** that reinjects the removed
statistics into the attention computation, so the model is trained on
stationarized inputs but is still aware of the original scale.

**Positioning — this is FreqLite's clearest novelty.** Vanilla RevIN applies a
**single, horizon-agnostic** denormalization: it adds back exactly the lookback
mean/variance to *every* forecast step `t = 1..H`. But under distribution shift
the appropriate level/scale typically *drifts across the horizon* — the lookback
statistics are a good estimate for `t = 1` and progressively worse for `t = H`.
Our **A-RevIN** keeps RevIN's reversible structure but makes the denormalization
**horizon-aware and gated**: a small learnable module modulates the
mean/variance that is restored as a function of horizon position (and/or a gate
on how much normalization to undo), letting the model interpolate between "trust
the lookback statistics" and "let the prediction set its own level." This is
philosophically aligned with Non-stationary Transformers' de-stationary idea —
the removed statistics should be reinjected *adaptively* rather than identically —
but realized as a lightweight, attention-free, RevIN-compatible module suited to
a linear backbone. To our knowledge, a horizon-adaptive / gated RevIN of this
form has not been published, and it is the component least anticipated by prior
linear or frequency-domain models (FITS and FreqMoE both use plain instance
normalization; see Section 8).

---

## 3. Frequency-domain forecasting

A large body of work moves all or part of the forecasting computation into a
spectral representation, motivated by the fact that long-term series are
dominated by a few periodic components that are sparse and well-separated in the
frequency domain.

- **Autoformer** `[wu2021autoformer]` (NeurIPS 2021) introduces the
  trend/seasonal **series decomposition block** (the moving-average split DLinear
  later reused) and replaces self-attention with an **Auto-Correlation** mechanism
  that discovers period-based dependencies via the FFT.
- **FEDformer** `[zhou2022fedformer]` (ICML 2022) combines series decomposition
  with **frequency-enhanced attention**: it operates on a randomly selected
  subset of Fourier (or wavelet) modes, giving linear complexity and an explicit
  low-rank spectral filter.
- **FiLM** `[zhou2022film]` (NeurIPS 2022) uses Legendre-polynomial projections
  to compress history and Fourier projections to denoise it, with a low-rank
  approximation for speed.
- **TimesNet** `[wu2023timesnet]` (ICLR 2023) uses the FFT to find dominant
  periods, folds the 1D series into 2D tensors indexed by period, and applies
  2D convolutions — a general time-series backbone rather than a pure forecaster.
- **FreTS** `[yi2023frets]` (NeurIPS 2023) is the most relevant of this group:
  it discards attention entirely and learns **frequency-domain MLPs** on the real
  and imaginary parts of the spectrum, over both inter-series (channel) and
  intra-series (time) dimensions.

**Positioning.** FreqLite shares these models' premise — the frequency domain is
the right place to separate predictable structure from noise — but rejects their
*machinery*. FEDformer/FiLM/TimesNet are heavy: attention, polynomial bases, or
2D convolutions, with parameter and FLOP budgets far beyond a 4 GB GPU's
comfortable range for the small-model regime we target. FreTS is lighter but
still uses (complex-valued) MLPs across channels and time. FreqLite instead uses
the spectrum only to *split* the signal into two bands, then forecasts each band
with a **plain real-valued linear head in the time domain**. The frequency domain
is a routing device, not the space the forecast is computed in. This keeps the
model linear-scale while still exploiting the spectral separability these works
demonstrate.

---

## 4. Transformer forecasters

- **Informer** `[zhou2021informer]` (AAAI 2021) makes long-sequence forecasting
  tractable with **ProbSparse** attention (`O(L log L)`), self-attention
  distilling, and a generative one-shot decoder.
- **PatchTST** `[nie2023patchtst]` (ICLR 2023) tokenizes the series into
  subseries **patches** and is strictly **channel-independent**, sharing one
  Transformer across channels. It is the strongest Transformer forecaster on the
  standard benchmarks and (memory permitting) our small-Transformer baseline.
- **iTransformer** `[liu2024itransformer]` (ICLR 2024) **inverts** the
  tokenization: each variate (channel) becomes one token and attention models
  cross-channel dependencies, reversing PatchTST's channel-independent stance.

**Positioning.** These are the high-accuracy / high-cost end of the design space
and motivate the central question of the paper: *how much of their accuracy can a
linear-scale model recover at a fraction of the cost?* We adopt PatchTST's
channel-independent treatment (also validated by RLinear) but reject patching and
attention. FreqLite is the explicit lightweight counterpoint; the
accuracy-vs-efficiency study (Section 6) quantifies the gap.

---

## 5. MLP-mixer and multiscale-decomposition forecasters

- **TSMixer** `[chen2023tsmixer]` (TMLR 2023) is an all-MLP architecture that
  alternates time-mixing and feature-mixing MLPs, showing that MLP mixing rivals
  Transformers without attention.
- **TimeMixer** `[wang2024timemixer]` (ICLR 2024) decomposes the series at
  **multiple sampling scales**, mixes the disentangled multiscale components
  (Past-Decomposable-Mixing), and combines multi-scale predictors
  (Future-Multipredictor-Mixing).

**Positioning.** Both confirm that decomposition + simple mixing is a strong,
attention-free recipe, and TimeMixer's multiscale decomposition is a *temporal*
analogue of our spectral band split. The difference is scope and weight class:
TSMixer/TimeMixer stack several mixing blocks (multi-layer MLPs) across scales and
channels; FreqLite uses a **single frequency split with two linear heads** and no
inter-channel mixing, targeting the smallest viable model rather than the best
MLP-mixer. They are useful efficient-MLP reference points, not direct baselines.

---

## 6. Green / efficient AI

The efficiency prong is grounded in the **Green AI** agenda `[schwartz2020greenai]`
(CACM 2020), which argues that efficiency (compute, energy, parameters) should be
a first-class evaluation criterion reported alongside accuracy, and
`[strubell2019energy]` (ACL 2019), which quantified the financial and
environmental cost of large deep-learning models and recommended reporting
training cost and prioritizing efficient models.

**Positioning.** FreqLite is a deliberately Green-AI-aligned contribution: we
report params, FLOPs, train time/epoch, and peak GPU memory alongside MSE/MAE,
and we demonstrate the whole pipeline on a single 4 GB laptop GPU. This connects
the LTSF-Linear "simpler is competitive" finding `[zeng2023dlinear]` to an
explicit efficiency-reporting discipline, rather than treating efficiency as an
afterthought.

---

## 7. Foundational references

The Transformer `[vaswani2017attention]` (NeurIPS 2017) underlies every
Transformer forecaster above; Adam `[kingma2015adam]` (ICLR 2015) is our
optimizer. Cited for completeness in method/setup, not surveyed.

---

## 8. Novelty-overlap assessment (read this)

Two recent papers are close enough to FreqLite that we must position against them
explicitly and honestly. **Neither kills our contribution, but the
"frequency-decomposed linear" framing must be sharpened, and A-RevIN must be
foregrounded as the load-bearing novelty.** I have flagged this to `team-lead`
and `methodologist`.

### 8.1 FITS — the closest lightweight frequency model

**FITS** `[xu2024fits]` ("Modeling Time Series with 10k Parameters", ICLR 2024)
is the single most dangerous overlap. Verified mechanism (from the paper):

- Applies **rFFT** to the lookback, a **single complex-valued linear layer** that
  performs amplitude scaling + phase shifting (frequency interpolation), then
  **irFFT** back to the time domain.
- Uses **reversible instance normalization (RIN)** — i.e. RevIN — to zero the
  mean (which also removes the dominant 0-frequency term).
- Uses a **fixed low-pass filter** to discard high-frequency components, which is
  what makes it ~5k–10k parameters.
- Channel-independent (weight sharing across channels).

Overlap with FreqLite: both are RevIN + frequency-domain + lightweight + CI. This
is real overlap and reviewers *will* raise it.

**How FreqLite is genuinely different from FITS:**

1. **High-frequency content is modeled, not discarded.** FITS applies a low-pass
   filter and *throws away* the high band. FreqLite *splits* into low and high
   bands and forecasts the high band with its own dedicated head — the explicit
   claim is that residual/high-frequency structure carries forecastable
   information that a low-pass model leaves on the table. This is a testable,
   falsifiable difference (ablation: low-band-only ≈ FITS-like behavior).
2. **Forecast computed in the time domain via two real linear heads**, not a
   single complex linear map in the frequency domain. The spectrum is used only
   to route; this is a different, simpler hypothesis class.
3. **Adaptive (horizon-aware/gated) RevIN.** FITS uses plain RIN. A-RevIN is the
   component with no analogue in FITS and is where our normalization novelty lives.

**Action taken:** do **not** claim "first lightweight frequency-domain linear
model" — FITS owns that. Frame FreqLite as: *band-split-with-dedicated-heads
(keep the high band) + adaptive reversible normalization*, and make FITS a cited
baseline if it runs in our budget (it is tiny, so it should).

### 8.2 FreqMoE — frequency-band experts

**FreqMoE** `[liu2025freqmoe]` (arXiv, Jan 2025) decomposes the spectrum into
bands and assigns each band to an **expert** network, with a **gating** network
that weights experts by frequency magnitude and learnable band boundaries, plus
residual complex-valued prediction blocks. It uses plain instance normalization
(mean/var), not an adaptive RevIN.

Overlap: the "each frequency band gets its own head" idea is shared. Differences:
FreqMoE is a *Mixture-of-Experts* with a learned gate, learnable boundaries, and
complex-valued multi-block residual stacks — substantially heavier and more
complex than FreqLite's **two fixed real linear heads (low/high) recombined by
addition**. FreqMoE has **no adaptive normalization** contribution. It is recent
(2025, arXiv-only at time of writing) so it is concurrent/contemporaneous rather
than established prior art; we cite it, note the conceptual proximity of the band
decomposition, and differentiate on (a) simplicity/weight class and (b) A-RevIN.

### 8.3 Net assessment

- The **frequency band split with per-band linear heads** is *not unique* — FITS
  (low-pass only) and FreqMoE (MoE over bands) occupy adjacent ground. We must
  present it as a *specific lightweight design point* (two real heads, keep the
  high band, additive recombination), not as a brand-new concept, and lean on the
  empirical efficiency story.
- The **A-RevIN (horizon-adaptive / gated reversible normalization)** is the
  prong with the **least prior-art overlap** and should be foregrounded as the
  primary methodological novelty. The cleanest defensible thesis is: *combine a
  lightweight frequency band split with an adaptive reversible normalization, and
  show this beats DLinear/RLinear/FITS-class models at comparable cost, with
  ablations attributing the gains.*
- **Honesty bar:** if ablations show the frequency split adds little over RLinear
  once A-RevIN is present (i.e. A-RevIN does the heavy lifting), we report that
  truthfully and reframe the paper around adaptive normalization for linear
  forecasters. We do not overclaim the spectral decomposition.

---

## Citation-key index (coordinate with `writer`)

| Key | Short name |
|-----|-----------|
| `zeng2023dlinear` | DLinear / LTSF-Linear (AAAI 2023) |
| `li2023rlinear` | RLinear (arXiv 2023) |
| `kim2022revin` | RevIN (ICLR 2022) |
| `liu2022nonstationary` | Non-stationary Transformers (NeurIPS 2022) |
| `wu2021autoformer` | Autoformer (NeurIPS 2021) |
| `zhou2022fedformer` | FEDformer (ICML 2022) |
| `zhou2022film` | FiLM (NeurIPS 2022) |
| `wu2023timesnet` | TimesNet (ICLR 2023) |
| `yi2023frets` | FreTS (NeurIPS 2023) |
| `xu2024fits` | FITS (ICLR 2024) — closest overlap |
| `liu2025freqmoe` | FreqMoE (arXiv 2025) — concurrent overlap |
| `zhou2021informer` | Informer (AAAI 2021) |
| `nie2023patchtst` | PatchTST (ICLR 2023) |
| `liu2024itransformer` | iTransformer (ICLR 2024) |
| `chen2023tsmixer` | TSMixer (TMLR 2023) |
| `wang2024timemixer` | TimeMixer (ICLR 2024) |
| `schwartz2020greenai` | Green AI (CACM 2020) |
| `strubell2019energy` | Energy & Policy (ACL 2019) |
| `vaswani2017attention` | Transformer (NeurIPS 2017) |
| `kingma2015adam` | Adam (ICLR 2015) |
