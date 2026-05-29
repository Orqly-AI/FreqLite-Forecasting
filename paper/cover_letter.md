# Cover Letter — Knowledge-Based Systems

To: The Editor-in-Chief
*Knowledge-Based Systems* (Elsevier)

Dear Editor,

We are pleased to submit our manuscript, **"FreqLite: A Lightweight Frequency-Decomposed Linear Model with Adaptive Reversible Normalization for Robust Long-Term Time-Series Forecasting,"** for consideration as an original research article in *Knowledge-Based Systems*.

Long-term time-series forecasting underpins decision-making across energy, weather, traffic, and industrial monitoring. Recent work has shown that lightweight linear models can match or exceed far heavier Transformers on standard benchmarks, reframing the field around models that are simultaneously accurate and efficient. Our work advances this direction with a model that is both competitive in accuracy and exceptionally cheap to train and deploy — the entire study is conducted on a single 4 GB consumer laptop GPU.

The manuscript makes the following contributions:

1. **FreqLite**, an ultra-lightweight, channel-independent frequency-decomposed linear forecaster. A learnable, lossless, partition-of-unity spectral filter splits the input into frequency bands modeled by per-band linear heads; unlike low-pass-truncation approaches, the high-frequency band is retained and modeled. FreqLite is the strongest lightweight model on the standard benchmarks and, at long lookback, attains lower average error than a PatchTST Transformer while using roughly 4× fewer parameters and 2.2× less memory and training time. These gains, though modest in magnitude, are statistically significant under paired Wilcoxon tests across all matched cells (p ≈ 10⁻⁶).

2. **Adaptive Reversible Instance Normalization (A-RevIN)**, a regime-adaptive normalization that strictly generalizes RevIN (recovered exactly when its gate is closed). It engages under non-stationarity and reduces to RevIN without harm on stationary data. We validate this behavior on both a real strongly non-stationary dataset (ILI) and a controlled synthetic drift study, and we report a methodological insight — a gradient trap in the gate initialization that otherwise keeps the mechanism dormant.

3. **A fully reproducible accuracy–efficiency study on commodity hardware**, with all baselines re-implemented and re-run under identical splits, normalization, seeds, and hardware, and with parameters, FLOPs, training time, and peak memory reported alongside MSE and MAE. Component contributions are isolated through ablations along the strict nesting Linear ⊆ RLinear ⊆ FreqLite.

We believe the work is a strong fit for *Knowledge-Based Systems*, which has a rich record of data-driven prediction methods that balance methodological rigor with practical, deployable efficiency. In keeping with our emphasis on transparency, we report our results honestly, including the cases where FreqLite does not lead (e.g., short-lookback settings where a Transformer is stronger), and we are explicit that the headline strengths on stationary benchmarks are efficiency and the long-lookback result rather than a large accuracy margin.

This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have read and approved the submission. The authors declare no competing interests, and the work received no specific funding.

Thank you for your consideration. We look forward to your response.

Sincerely,

**Mirza Samad Ahmed Baig** (Corresponding author)
Fandaqah, Al Khobar, Saudi Arabia
MirzaSamadcontact@gmail.com

Syeda Anshrah Gillani
Heidelberg University, Heidelberg, Germany
SyedaAnshrah16@gmail.com
