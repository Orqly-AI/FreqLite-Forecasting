"""Download the long-term forecasting benchmark datasets used by FreqLite.

All files are the *canonical wide-format* CSVs (column ``date`` followed by the
channel columns) used by the established LTSF literature (Informer / Autoformer /
DLinear / FITS), so our numbers are directly comparable to published baselines.

Source mirror (public, no auth): AutonLab "Timeseries-PILE" on HuggingFace,
``forecasting/autoformer/`` — these are byte-for-byte the Autoformer release
files. ETT sizes match the official zhouhaoyi/ETDataset repo exactly.

Run:
    .venv\\Scripts\\python.exe scripts\\download_data.py            # core datasets
    .venv\\Scripts\\python.exe scripts\\download_data.py --with-ecl # + electricity

Idempotent: a file already present with the right size is skipped. After a
successful download the SHA-256 of every file is written to
``data/checksums.txt`` for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = (
    "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/"
    "resolve/main/forecasting/autoformer/"
)

# name -> (expected size in bytes, expected number of channel columns excl. date).
# Sizes are the full-GET byte counts observed on download (the mirror's ranged
# Content-Range total can differ by a few bytes, so size is a soft check only;
# the hard check is that the file parses as the expected wide-format CSV).
CORE = {
    "ETTh1.csv": (2589657, 7),
    "ETTh2.csv": (2417960, 7),
    "ETTm1.csv": (10360719, 7),
    "ETTm2.csv": (9677236, 7),
    "weather.csv": (7182728, 21),
}
OPTIONAL = {
    "electricity.csv": (None, 321),  # ECL — large; only with --with-ecl
}
# Genuinely non-stationary datasets for the A-RevIN analysis (only with --nonstationary).
NONSTATIONARY = {
    "exchange_rate.csv": (None, 8),
    "national_illness.csv": (None, 7),
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _download(name: str, dest: Path, expected_size: int | None, n_cols: int) -> None:
    url = BASE + name
    print(f"  downloading {name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = tmp.stat().st_size
    if expected_size is not None and abs(size - expected_size) > 4096:
        # tolerate a few bytes of mirror variance; flag large discrepancies
        print(f"    WARNING: {name} is {size} bytes, expected ~{expected_size}")
    # hard check: must parse as a wide CSV whose first column is 'date'
    import pandas as pd

    head = pd.read_csv(tmp, nrows=5)
    cols = list(head.columns)
    if cols[0].lower() != "date":
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: first column is {cols[0]!r}, expected 'date'")
    if len(cols) - 1 != n_cols:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: {len(cols) - 1} channel columns, expected {n_cols}"
        )
    tmp.replace(dest)
    print(f"    saved {dest} ({size / 1e6:.2f} MB, {n_cols} channels)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ecl", action="store_true", help="also fetch electricity.csv")
    ap.add_argument("--nonstationary", action="store_true",
                    help="also fetch exchange_rate.csv and national_illness.csv (A-RevIN study)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = dict(CORE)
    if args.with_ecl:
        targets.update(OPTIONAL)
    if args.nonstationary:
        targets.update(NONSTATIONARY)

    for name, (exp, n_cols) in targets.items():
        dest = DATA_DIR / name
        if dest.exists() and (exp is None or abs(dest.stat().st_size - exp) <= 4096):
            print(f"  [skip] {name} already present")
            continue
        _download(name, dest, exp, n_cols)

    # write checksums for every csv currently in data/
    lines = []
    for csv in sorted(DATA_DIR.glob("*.csv")):
        lines.append(f"{_sha256(csv)}  {csv.name}")
    (DATA_DIR / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nchecksums written to data/checksums.txt:")
    print("\n".join("  " + line for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
