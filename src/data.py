"""Data pipeline for long-term forecasting, matching the standard LTSF protocol.

Conventions (Informer/Autoformer/DLinear codebase), so our numbers are directly
comparable to published baselines:

* ETTh1/ETTh2: split by *months* — 12 months train, 4 val, 4 test
  (boundaries at row 12*30*24, +4*30*24, +4*30*24; hourly).
* ETTm1/ETTm2: same months at 15-min resolution → multiply by 4
  (12*30*24*4, etc.).
* Weather / Electricity (and any other): 70 / 10 / 20 % chronological split,
  with the val/test windows offset back by the lookback `L` so the first
  prediction target of each split is the official boundary (standard
  ``border1 = boundary - L``).
* Normalization: per-channel z-score using **train-split statistics only**
  (fit on the train rows, applied to all splits). This is the LTSF convention;
  loss/metrics are reported in this train-z-scored space.
* Windowing: sliding windows of length ``L + H`` with stride 1; input is the
  first ``L`` steps, target the last ``H``.
* Multivariate -> we forecast all channels (M->M). Channel-independence is
  handled inside the models, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Datasets that use the ETT month-based split rather than 70/10/20.
_ETT_HOUR = {"ETTh1", "ETTh2"}
_ETT_MIN = {"ETTm1", "ETTm2"}


@dataclass
class Splits:
    """Inclusive-exclusive row ranges [start, end) for each split, in the raw
    (un-windowed) series index. val/test starts are already offset by -L."""

    borders: dict  # {"train": (s,e), "val": (s,e), "test": (s,e)}


def _compute_borders(n_rows: int, name: str, L: int) -> Splits:
    if name in _ETT_HOUR or name in _ETT_MIN:
        mult = 4 if name in _ETT_MIN else 1
        train_end = 12 * 30 * 24 * mult
        val_end = train_end + 4 * 30 * 24 * mult
        test_end = val_end + 4 * 30 * 24 * mult
        # The canonical ETT setup uses exactly these fixed boundaries.
        b = {
            "train": (0, train_end),
            "val": (train_end - L, val_end),
            "test": (val_end - L, test_end),
        }
    else:
        train_end = int(n_rows * 0.7)
        val_end = int(n_rows * 0.8)  # 70% train, 10% val, 20% test
        test_end = n_rows
        b = {
            "train": (0, train_end),
            "val": (train_end - L, val_end),
            "test": (val_end - L, test_end),
        }
    return Splits(borders=b)


class ForecastDataset(Dataset):
    """Sliding-window dataset over one split of one CSV.

    Returns (x, y) with x: (L, C) float32, y: (H, C) float32, already z-scored
    by train statistics.
    """

    def __init__(
        self,
        values: np.ndarray,  # (split_rows, C) already normalized
        L: int,
        H: int,
    ) -> None:
        self.values = values.astype(np.float32)
        self.L = L
        self.H = H
        self.n = len(self.values) - L - H + 1
        if self.n <= 0:
            raise ValueError(
                f"split too short: rows={len(self.values)} for L={L}, H={H}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = self.values[i : i + self.L]
        y = self.values[i + self.L : i + self.L + self.H]
        return torch.from_numpy(x), torch.from_numpy(y)


def _load_csv(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )
    df = pd.read_csv(path)
    # First column is the 'date' index; the rest are channels.
    data = df.iloc[:, 1:].values.astype(np.float64)
    return data


def make_loaders(
    name: str,
    L: int,
    H: int,
    batch_size: int,
    num_workers: int = 0,
):
    """Build train/val/test DataLoaders for a dataset following LTSF protocol.

    Returns (loaders dict, meta dict). meta has C (channels), train mean/std
    (per channel) and the dataset name.
    """
    raw = _load_csv(name)  # (N, C)
    n_rows, C = raw.shape
    splits = _compute_borders(n_rows, name, L)

    tr_s, tr_e = splits.borders["train"]
    train_slice = raw[tr_s:tr_e]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # guard constant channels

    norm = (raw - mean) / std  # z-score whole series with TRAIN stats

    loaders = {}
    shuffles = {"train": True, "val": False, "test": False}
    for split in ("train", "val", "test"):
        s, e = splits.borders[split]
        ds = ForecastDataset(norm[s:e], L, H)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffles[split],
            num_workers=num_workers,
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available(),
        )

    meta = {
        "name": name,
        "C": C,
        "n_rows": n_rows,
        "train_mean": mean,
        "train_std": std,
        "borders": splits.borders,
        "sizes": {k: len(loaders[k].dataset) for k in loaders},
    }
    return loaders, meta
