"""Main experiment runner — produces results/main_results.csv.

Iterates the (dataset x horizon x seed x model) grid from a config, trains each
cell under the standard LTSF protocol, evaluates on the test split, and appends
one row per cell with metrics + efficiency to a CSV. Resumable: a cell already
present in the CSV is skipped.

Usage:
  .venv\\Scripts\\python.exe scripts\\run_experiments.py --config configs/default.yaml
  ... --lookback 96                 # override lookback
  ... --models freqlite,rlinear     # subset of models
  ... --datasets ETTh1,weather      # subset of datasets
  ... --horizons 96,720             # subset of horizons
  ... --seeds 2021                  # subset of seeds
  ... --out results/main_results.csv
  ... --max-epochs 20               # override (smoke runs use small values)
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
from src.models import build_model
from src.utils import get_device, set_seed

FIELDS = [
    "model", "dataset", "lookback", "horizon", "seed",
    "test_mse", "test_mae", "val_mse",
    "params", "flops_per_series", "sec_per_epoch", "peak_gpu_mem_mb",
    "epochs_run", "lookback_label",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_cfg(cfg: dict, model_name: str) -> dict:
    """Assemble the per-model build kwargs from default model cfg + overrides."""
    if model_name == "freqlite":
        base = {
            "K": cfg["model"].get("K", 2),
            "recombination": cfg["model"].get("recombination", "sum"),
            "decomposition": cfg["model"].get("decomposition", {}),
            "arevin": cfg["model"].get("arevin", {}),
        }
    else:
        base = {}
    base.update(cfg.get("model_overrides", {}).get(model_name, {}))
    return base


def read_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((
                    row["model"], row["dataset"],
                    int(row["lookback"]), int(row["horizon"]), int(row["seed"]),
                ))
    return done


def append_row(out_path: Path, row: dict) -> None:
    exists = out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/main_results.csv")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--mem-abort-mb", type=float, default=3500.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    tr = cfg["train"]

    L = args.lookback if args.lookback is not None else exp["lookback"]
    models = (args.models.split(",") if args.models else exp["models"])
    datasets = (args.datasets.split(",") if args.datasets else exp["datasets"])
    horizons = ([int(h) for h in args.horizons.split(",")] if args.horizons
                else exp["horizons"])
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else exp["seeds"])
    max_epochs = args.max_epochs if args.max_epochs is not None else tr["max_epochs"]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out_path)

    device = get_device()
    print(f"device={device} cuda={torch.cuda.is_available()}  out={out_path}")
    print(f"grid: {len(models)} models x {len(datasets)} datasets x "
          f"{len(horizons)} horizons x {len(seeds)} seeds, L={L}")

    total = 0
    t_start = time.perf_counter()
    for dataset in datasets:
        bs = cfg.get("data_overrides", {}).get(dataset, {}).get(
            "batch_size", tr["batch_size"])
        for H in horizons:
            # build loaders once per (dataset,H); reuse across models+seeds
            loaders, meta = make_loaders(dataset, L=L, H=H, batch_size=bs)
            C = meta["C"]
            for model_name in models:
                mcfg = build_model_cfg(cfg, model_name)
                for seed in seeds:
                    key = (model_name, dataset, L, H, seed)
                    if key in done:
                        continue
                    set_seed(seed, deterministic=tr.get("deterministic", True)
                             if "deterministic" in tr else True)
                    model = build_model(model_name, L=L, H=H, C=C, cfg=mcfg)
                    res = train_model(
                        model, loaders, device,
                        lr=tr["lr"], weight_decay=tr["weight_decay"],
                        max_epochs=max_epochs, patience=tr["patience"],
                        grad_clip_norm=tr["grad_clip_norm"],
                        lr_schedule=tr["lr_schedule"], L=L, H=H,
                    )
                    if res.peak_gpu_mem_mb > args.mem_abort_mb:
                        print(f"  WARNING mem {res.peak_gpu_mem_mb:.0f}MB > "
                              f"{args.mem_abort_mb} for {key}")
                    test_mse, test_mae = evaluate(model, loaders["test"], device)
                    row = {
                        "model": model_name, "dataset": dataset,
                        "lookback": L, "horizon": H, "seed": seed,
                        "test_mse": f"{test_mse:.6f}", "test_mae": f"{test_mae:.6f}",
                        "val_mse": f"{res.best_val:.6f}",
                        "params": res.params,
                        "flops_per_series": res.flops_per_series,
                        "sec_per_epoch": f"{res.sec_per_epoch:.4f}",
                        "peak_gpu_mem_mb": f"{res.peak_gpu_mem_mb:.1f}",
                        "epochs_run": res.epochs_run,
                        "lookback_label": f"L{L}",
                    }
                    append_row(out_path, row)
                    total += 1
                    print(f"  [{total}] {model_name:9s} {dataset:6s} H={H:3d} "
                          f"s={seed} mse={test_mse:.4f} mae={test_mae:.4f} "
                          f"p={res.params} mem={res.peak_gpu_mem_mb:.0f}MB "
                          f"{res.sec_per_epoch:.2f}s/ep")
    dt = time.perf_counter() - t_start
    print(f"\nDONE: {total} new cells in {dt/60:.1f} min -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
