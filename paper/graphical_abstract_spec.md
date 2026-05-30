# Graphical abstract — spec (optional but recommended for KBS)

Elsevier graphical-abstract requirements:
- Single image, **landscape**, minimum **531 (h) × 1328 (w) px** (≈ 5.3 × 13.3 cm),
  preferably larger (e.g., 1000 × 2500 px) at high resolution.
- Readable standalone; sans-serif font; minimal text; no unexplained acronyms.
- Accepted formats: TIFF/EPS/PDF/PNG.

## Recommended layout (two panels, left → right)

```
+-------------------------------------------------+----------------------------------+
|  FreqLite (method, simplified)                  |  Headline result                 |
|                                                 |                                  |
|  x  ->  A-RevIN  ->  freq. split (low/high)     |   scatter: MSE (y) vs #params(x) |
|         norm        |          |                |   - FreqLite (low MSE, tiny)  *  |
|                     v          v                |   - PatchTST (higher MSE, huge)  |
|                 Linear      Linear              |   - DLinear/RLinear/FITS         |
|                     \        /                  |                                  |
|                      (sum) -> A-RevIN denorm    |   caption strip:                 |
|                              -> forecast  ŷ     |   "Beats a Transformer at 4x     |
|                                                 |    fewer params on a 4GB laptop" |
+-------------------------------------------------+----------------------------------+
```

- **Left panel:** a clean, simplified version of the Fig. 1 architecture (fewer
  boxes than the paper figure; just the narrative flow). Color: A-RevIN orange,
  linear heads green.
- **Right panel:** the accuracy-vs-parameters trade-off (we already generate
  `results/figures/accuracy_vs_params.{pdf,png}`) — reuse it, highlight FreqLite at
  the bottom-left (low error, few params) vs PatchTST top-right.
- **One-line takeaway** across the bottom: *"Transformer-level long-horizon accuracy
  at ~4× fewer parameters — on a single 4 GB laptop GPU."*

## I can generate this for you
The right panel already exists. I can produce the full graphical abstract as a
single PDF/PNG (`results/figures/graphical_abstract.{pdf,png}`) by combining a
simplified matplotlib pipeline diagram with the existing trade-off plot, sized to
Elsevier spec. Say the word and I'll build it.
