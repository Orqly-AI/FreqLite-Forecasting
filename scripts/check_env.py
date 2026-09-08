"""Environment & GPU sanity check for FreqLite.

Run:  .venv\\Scripts\\python.exe scripts\\check_env.py

Prints versions, CUDA availability, the device name, and total/usable VRAM.
Exits non-zero if any required import fails so CI / smoke runs catch it early.
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "matplotlib",
    "tqdm",
]


def main() -> int:
    print("=" * 60)
    print("FreqLite environment check")
    print("=" * 60)
    print(f"Python      : {platform.python_version()}  ({sys.executable})")
    print(f"Platform    : {platform.platform()}")

    ok = True
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok] {mod:<12} {ver}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {mod:<12} {exc}")

    if not ok:
        print("\nMissing dependencies. Run: pip install -r requirements.txt")
        return 1

    import torch

    print("-" * 60)
    print(f"torch       : {torch.__version__}")
    print(f"torch CUDA  : {torch.version.cuda}")
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"device      : {props.name}")
        print(f"total VRAM  : {props.total_memory / 1e9:.3f} GB")
        # quick GPU op to confirm the runtime works end to end
        x = torch.randn(2048, 2048, device="cuda")
        _ = x @ x
        torch.cuda.synchronize()
        print(f"matmul OK   : peak alloc {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    else:
        print("WARNING: CUDA not available — runs will fall back to CPU.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
