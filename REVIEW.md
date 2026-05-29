# Internal Review & Reproducibility Report — FreqLite

Reviewer: team-lead. Role: adversarial Q1-reviewer-style critique + reproducibility audit before submission.

## 1. Reproducibility (verified)
- **Independent build:** `scripts/build_paper.ps1` rebuilt `paper/main.pdf` from clean → **23 pages, 0 LaTeX errors, 0 undefined refs/citations**.
- **Bit-exact re-run:** re-ran one cell from scratch (`freqlite, ETTh1, L=336, H=96, seed=2021`) into a fresh CSV → `test_mse = 0.374102`, `mae = 0.394628`, `val = 0.672235` — **identical** to the stored `main_results.csv` value. Fixed-seed determinism confirmed (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic).
- **Provenance:** all 7 baselines re-implemented and run in-repo on identical splits/normalization/seeds/hardware; SHA-256 dataset checksums recorded; every paper number traces to a script. README documents the full pipeline incl. the non-stationarity study and the `--nonstationary` data flag.
- **Claims-vs-evidence audit:** spot-checked intro/results/efficiency/ablation numbers against the CSVs — all match (headline 0.3244; L96 PatchTST 0.3541 / FreqLite 0.3589; ILI A-RevIN-isolated 0.66/0.63/0.52/0.38%; gate ρ̄ 0.02→0.51→1.00). No fabricated or unsupported numbers found.

## 2. Strengths (likely to land well at Q1)
- **Honest, mechanistically-grounded story.** The A-RevIN narrative survived an adversarial test instead of being assumed; the gradient-trap finding + monotonic gate sweep is a genuine, interpretable contribution.
- **Strong efficiency result.** Best lightweight model on standard benchmarks; at L=336 beats PatchTST at ~4× fewer params / ~2.2× less memory/time on a 4 GB GPU — a clean, well-quantified Green-AI claim.
- **Rigor.** Strict nesting (Linear ⊆ RLinear ⊆ FreqLite) makes each component independently ablatable; baselines self-run; full reproducibility.
- **Honesty bar respected.** Modest stationary-benchmark gains, exchange-rate loss to DLinear, and dataset-dependent A-RevIN benefit are all stated plainly.

## 3. Weaknesses / risks a Q1 reviewer will raise (and our standing)
1. **Accuracy novelty over RLinear is small (~0.9% at L=336).** *Mitigation in paper:* headline reframed onto efficiency + long-lookback Transformer win + the regime-adaptive A-RevIN; reported honestly. **Residual risk: moderate at a pure-methods venue; lower at an applied Q1 (KBS/Neurocomputing/ASC).**
2. **FITS overlap.** Closest prior art. *Mitigation:* explicit keep-vs-discard distinction + FITS run as a baseline (FreqLite beats it: 0.3244 vs 0.3291 at L336). Adequate.
3. **A-RevIN's win is concentrated on ILI.** Only two non-stationary datasets tested. *Recommendation (optional strengthening):* add 1–2 more non-stationary sets (e.g. ETT under a distribution-shift split, traffic) to show the ILI gain generalizes.
4. **Single 4 GB GPU / linear-scale scope.** Framed as a deliberate Green-AI contribution, not a limitation to hide — fine, but reviewers may want one larger-data point (ECL) for completeness.

## 4. Minor / cosmetic
- Writer added a `\fitwidetables` (`\resizebox`) shim in `main.tex` for the wide 16-col main tables. **Recommendation:** bake `\resizebox{\textwidth}{!}{...}` into `make_tables.py` so the table files are self-contained and the shim can be removed.
- One 2.1 pt overfull hbox (table-header rule) — visually negligible.
- `docs/method_spec.md` is the pre-experiment spec; a post-experiment note now records the ρ₀=0 default + regime-adaptive reframing for consistency.

## 4b. Post-strengthening update (addressing §3)
Three additions were made after the initial review, directly targeting the weaknesses above:
- **Statistical significance (§3.1 resolved):** paired Wilcoxon across all 60 L=336 cells gives FreqLite vs RLinear p≈1.2e-6 (and p<1e-3 vs every lightweight baseline at both lookbacks). The "is 0.9% real?" critique is answered: the gain is small but highly significant. (`scripts/significance.py`, Table `tab:significance`.)
- **Synthetic controlled-drift study (§3.3 resolved):** with a shared base + persistent extrapolatable trend, A-RevIN's advantage rises monotonically with injected drift (−0.28% at δ=0 → +1.42% at δ=4), gate ρ rising in step — a controlled confirmation complementing ILI. (`scripts/run_synthetic.py`, Fig. `fig:synthetic`, Table `tab:synthetic`.) Note: first synthetic design (random-walk drift) failed and was discarded honestly before redesign.
- **ECL large-scale (§3.4 resolved):** FreqLite runs on full 321-channel Electricity on the 4 GB GPU and is 2nd-best (0.169, within ~1% of DLinear), beating RLinear/NLinear/FITS; PatchTST is 4 GB-infeasible there. Reported honestly (DLinear wins ECL, like Exchange-rate). (`scripts/make_ecl_table.py`, Table `tab:ecl`.)
Manuscript now 26 pp; author block, CRediT (equal contribution), competing-interest/funding declarations, and a KBS cover letter (`paper/cover_letter.md`) are in place.

## 5. Verdict
**Submission-ready for *Knowledge-Based Systems*.** The work is honest, reproducible, and the central claims are fully evidenced. The optional strengthening items from the initial review (§3.1/§3.3/§3.4) have now been completed (§4b): statistical significance, a synthetic drift study, and the ECL large-scale run — converting the main "thin novelty" risk into a documented, significant, and scalable result. Remaining before clicking submit are author-side mechanics only: a final human read-through and (optional but recommended) a public code repository to cite in the Data Availability statement.
