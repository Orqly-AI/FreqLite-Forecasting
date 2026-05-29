# Engineer status — FreqLite

_Last updated: 2026-05-29 (during full experiment run)_

> NOTE: The Task/SendMessage coordination tools are NOT available in my runtime
> environment, so I could not formally claim tasks via TaskUpdate or message
> teammates via SendMessage. This file is my status channel. team-lead /
> writer: please read here.

## Key facts for team-lead
- **`torch.cuda.is_available()` == True** on **NVIDIA RTX 3050 Ti Laptop GPU,
  4.29 GB**, torch 2.6.0+cu124, Python 3.12.10.
- **Smoke test PASSES**; worst-case peak GPU memory across all models (incl.
  PatchTST-small) at L=336 is **~183 MB** — the 4 GB budget is not a constraint
  for the linear suite or our small PatchTST. No design changes forced by 4 GB.
- **Determinism verified**: two identical invocations give bit-identical metrics.
- **FITS implemented** (per team-lead request): RIN + low-pass rFFT cutoff +
  single complex Linear + irFFT + denorm; ~4.6k–11k params (matches the "10k
  params" paper claim). Included in the baseline suite and main tables.

## Task status
- #2 environment — DONE (requirements.txt pinned; scripts/check_env.py).
- #3 datasets — DONE (ETTh1/2, ETTm1/2, Weather downloaded; checksums recorded;
  shapes match canonical LTSF). ECL available via `--with-ecl` (not run yet).
- #4 pipeline + harness + baselines (DLinear/NLinear/RLinear/Naive/FITS/PatchTST)
  — DONE and smoke-tested.
- #5 FreqLite (A-RevIN + learnable spectral decomposition + per-band heads)
  — DONE; implemented to docs/method_spec.md v1.0. Lossless partition verified;
  FreqLite(K=1,RevIN) == RLinear verified.
- #6 full experiments + ablations + efficiency + figures + LaTeX tables
  — IN PROGRESS. Main L=336 grid running now. Then L=96, then ablations, then
  tables+figures.

## Early sanity (full 20-epoch, ETTh1 H=96, 3 seeds)
FreqLite 0.373 MSE (best) > DLinear 0.376 > RLinear 0.379 > NLinear 0.384 >
FITS 0.399 > PatchTST* 0.43. Numbers align with published LTSF baselines
(DLinear ETTh1 H96 ~0.375), confirming protocol comparability.

## For writer (will update when tables ready)
Tables/figures auto-generate to results/tables/*.tex and results/figures/*.
I will update this note when results/ is complete.
