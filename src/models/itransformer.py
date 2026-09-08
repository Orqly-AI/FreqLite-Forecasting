"""iTransformer (Liu et al., ICLR 2024; arXiv 2310.06625) — 4 GB-sized variant.

Faithful "inverted" transformer: instead of attending across time steps, each
VARIATE's full lookback (length L) is embedded as one token by a shared
``Linear(L -> d_model)``; standard transformer encoder layers then attend
ACROSS the C variate tokens (capturing multivariate correlations); finally a
shared ``Linear(d_model -> H)`` head projects each variate token to its
horizon. As in the official implementation:

* there is NO positional embedding — variate tokens carry no order, and
  attention over them is permutation-equivariant by design;
* the encoder layer is post-norm (LayerNorm after attention / FFN);
* a RevIN-style per-instance, per-variate normalization (non-affine, exactly
  the official "use_norm" Non-stationary-Transformer normalization) wraps the
  model — it is integral to iTransformer's reported performance.

This model is NOT channel-independent: it sees all C channels jointly (that is
its whole point), so the embedding cost is per-variate-token and the attention
cost is O(C^2). For large-C datasets (traffic C=862, electricity C=321) the
per-dataset batch size in configs/default.yaml keeps it inside the 4 GB budget.

Small config for the 4 GB budget: d_model=128, n_heads=4, e_layers=2, d_ff=128,
dropout=0.1 (set via model_overrides in configs/default.yaml).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .revin import RevIN


class ITransformer(nn.Module):
    def __init__(
        self,
        L: int,
        H: int,
        C: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.L, self.H, self.C = L, H, C

        # Official iTransformer normalization is the non-affine instance norm
        # (subtract per-variate lookback mean, divide by std, re-apply at the
        # end). Our RevIN module with affine=False is exactly that.
        self.revin = RevIN(eps=eps, affine=False)

        # Inverted embedding: one token per variate, embedding its length-L series.
        self.embed = nn.Linear(L, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,  # post-norm, as in the official encoder
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        # instance-normalize each variate's lookback (RevIN statistics per (B,C))
        s = x.permute(0, 2, 1).reshape(B * C, L)  # (B*C, L)
        s = self.revin.normalize(s)

        tokens = s.reshape(B, C, L)               # one length-L token per variate
        z = self.drop(self.embed(tokens))         # (B, C, d_model)
        z = self.encoder(z)                       # attention ACROSS variates
        out = self.head(z)                        # (B, C, H)

        out = self.revin.denormalize(out.reshape(B * C, self.H))
        return out.reshape(B, C, self.H).permute(0, 2, 1).contiguous()  # (B,H,C)
