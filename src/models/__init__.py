"""FreqLite models and baselines.

A single ``build_model(name, L, H, C, cfg)`` factory returns any model by name
so the training harness stays model-agnostic.
"""

from __future__ import annotations

from .baselines import DLinear, FITS, NLinear, RLinear
from .freqlite import FreqLite
from .naive import NaiveRepeatLast

_REGISTRY = {
    "naive": NaiveRepeatLast,
    "nlinear": NLinear,
    "dlinear": DLinear,
    "rlinear": RLinear,
    "fits": FITS,
    "freqlite": FreqLite,
}

# PatchTST is optional (heavier). Import lazily so missing/edge configs don't
# break the linear suite.
try:  # pragma: no cover - optional
    from .patchtst import PatchTST  # noqa: F401

    _REGISTRY["patchtst"] = PatchTST
except Exception:  # noqa: BLE001
    pass


def build_model(name: str, L: int, H: int, C: int, cfg: dict | None = None):
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(_REGISTRY)}")
    cfg = cfg or {}
    return _REGISTRY[name](L=L, H=H, C=C, **cfg)


def available_models():
    return sorted(_REGISTRY)
