"""Ablation runner — produces results/ablations.csv (method_spec Sec. 9).

Variants (all FreqLite configurations, isolating each contribution):
  A0 Full FreqLite (K=2, learnable masks, A-RevIN)
  A1 - A-RevIN -> plain RevIN (adaptive off)
  A2 - learnable decomposition -> fixed DLinear-style MA split + A-RevIN
  A3 K=1 (no decomposition) + A-RevIN          (= "A-RevIN-Linear")
  A4 K=1 + plain RevIN                          (= RLinear, degenerate FreqLite)
  A5 K=3 and K=4 bands
  A6 learnable gate recombination
  A7 freeze decomposition at init (no grad on cutoff/sharpness)
  A8 A-RevIN without lambda (drift propagation)

Subset grid (cost control): {ETTh1, ETTm2, weather} x H in {96,720} x 3 seeds, L=336.

Note A2 (fixed MA split): the spec asks for a DLinear-style time-domain moving
average split feeding two heads, while keeping A-RevIN. We realize this as a
dedicated variant model in src (FreqLiteFixedMA) so the comparison is apples to
apples (same A-RevIN, same two-head structure, only the split differs).

Usage: .venv\\Scripts\\python.exe scripts\\run_ablations.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data import make_loaders
from src.engine import evaluate, train_model
from src.models.freqlite import FreqLite
from src.models.ablation import FreqLiteFixedMA
from src.utils import get_device, set_seed

FIELDS = [
    "variant", "dataset", "lookback", "horizon", "seed",
    "test_mse", "test_mae", "val_mse", "params", "epochs_run",
]

# variant -> (builder_kind, kwargs). builder_kind in {"freqlite","fixedma"}.
def variants(base_decomp: dict, base_arevin: dict):
    full_arevin = dict(base_arevin)
    no_adapt = dict(base_arevin, adaptive=False)
    no_lambda = dict(base_arevin, use_lambda=False)
    frozen_decomp = dict(base_decomp, learnable=False)

    return {
        "A0_full":        ("freqlite", dict(K=2, decomposition=base_decomp, arevin=full_arevin, recombination="sum")),
        "A1_revin":       ("freqlite", dict(K=2, decomposition=base_decomp, arevin=no_adapt, recombination="sum")),
        "A2_fixedMA":     ("fixedma",  dict(arevin=full_arevin)),
        "A3_K1_arevin":   ("freqlite", dict(K=1, decomposition=base_decomp, arevin=full_arevin, recombination="sum")),
        "A4_K1_revin":    ("freqlite", dict(K=1, decomposition=base_decomp, arevin=no_adapt, recombination="sum")),
        "A5_K3":          ("freqlite", dict(K=3, decomposition=base_decomp, arevin=full_arevin, recombination="sum")),
        "A5_K4":          ("freqlite", dict(K=4, decomposition=base_decomp, arevin=full_arevin, recombination="sum")),
        "A6_gate":        ("freqlite", dict(K=2, decomposition=base_decomp, arevin=full_arevin, recombination="gate")),
        "A7_frozen":      ("freqlite", dict(K=2, decomposition=frozen_decomp, arevin=full_arevin, recombination="sum")),
        "A8_no_lambda":   ("freqlite", dict(K=2, decomposition=base_decomp, arevin=no_lambda, recombination="sum")),
    }


def build(kind: str, L: int, H: int, C: int, kw: dict):
    if kind == "freqlite":
        return FreqLite(L=L, H=H, C=C, **kw)
    if kind == "fixedma":
        return FreqLiteFixedMA(L=L, H=H, C=C, **kw)
    raise ValueError(kind)


def read_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["variant"], r["dataset"], int(r["lookback"]),
                          int(r["horizon"]), int(r["seed"])))
    return done


def append_row(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/ablations.csv")
    ap.add_argument("--datasets", default="ETTh1,ETTm2,weather")
    ap.add_argument("--horizons", default="96,720")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--variants", default=None, help="comma list to subset")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tr, exp = cfg["train"], cfg["experiment"]
    L = args.lookback or exp["lookback"]
    datasets = args.datasets.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else exp["seeds"])
    max_epochs = args.max_epochs if args.max_epochs is not None else tr["max_epochs"]

    var_map = variants(cfg["model"]["decomposition"], cfg["model"]["arevin"])
    if args.variants:
        keep = set(args.variants.split(","))
        var_map = {k: v for k, v in var_map.items() if k in keep}

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out_path)
    device = get_device()
    print(f"device={device}  variants={list(var_map)}  L={L}")

    total = 0
    t0 = time.perf_counter()
    for dataset in datasets:
        for H in horizons:
            loaders, meta = make_loaders(dataset, L=L, H=H, batch_size=tr["batch_size"])
            C = meta["C"]
            for vname, (kind, kw) in var_map.items():
                for seed in seeds:
                    key = (vname, dataset, L, H, seed)
                    if key in done:
                        continue
                    set_seed(seed)
                    model = build(kind, L, H, C, kw)
                    res = train_model(model, loaders, device, lr=tr["lr"],
                                      weight_decay=tr["weight_decay"],
                                      max_epochs=max_epochs, patience=tr["patience"],
                                      grad_clip_norm=tr["grad_clip_norm"],
                                      lr_schedule=tr["lr_schedule"], L=L, H=H)
                    tmse, tmae = evaluate(model, loaders["test"], device)
                    append_row(out_path, {
                        "variant": vname, "dataset": dataset, "lookback": L,
                        "horizon": H, "seed": seed,
                        "test_mse": f"{tmse:.6f}", "test_mae": f"{tmae:.6f}",
                        "val_mse": f"{res.best_val:.6f}", "params": res.params,
                        "epochs_run": res.epochs_run,
                    })
                    total += 1
                    print(f"  [{total}] {vname:14s} {dataset:6s} H={H:3d} s={seed} "
                          f"mse={tmse:.4f} mae={tmae:.4f} p={res.params}")
    print(f"\nDONE: {total} cells in {(time.perf_counter()-t0)/60:.1f} min -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
