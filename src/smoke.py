"""Smoke test: all three corpora load, window and stratify sanely.

The data/_raw_{name}.npz caches are tried first, so a machine that only has the
caches can still self-check. A missing cache falls back to the raw directories,
and a missing corpus raises something actionable instead of numpy's
"need at least one array to stack".
"""
import sys
from collections import Counter

import numpy as np

from . import config
from .datasets import load_cwru, load_mfpt, load_pu
from .run_experiment import get_windows

LOADER = {"cwru": load_cwru, "mfpt": load_mfpt, "pu": load_pu}
HINT = {
    "cwru": "data/cwru/CWRU-dataset-main/  (see data/README.md)",
    "mfpt": "data/mfpt/**/*.mat            (see data/README.md)",
    "pu":   "data/pu/{K001,KA04,KI04}/     (python -m src.fetch_pu K001 KA04 KI04)",
}


def report(name, X, y, runs):
    cnt = Counter(config.CLASSES[i] for i in y)
    print(f"\n=== {name.upper()} ===")
    print(f"windows={len(X)}  shape={X.shape}  runs={len(set(runs))}  "
          f"fs={config.FS_TARGET} Hz")
    print("class distribution:", {k: cnt.get(k, 0) for k in config.CLASSES})
    print(f"window RMS: median={np.median(np.sqrt((X ** 2).mean(1))):.4f}")
    assert len(X) > 0, "no windows"
    assert len(set(y)) == 3, f"expected 3 classes, got {len(set(y))}"
    assert not np.isnan(X).any(), "NaN in windows"


if __name__ == "__main__":
    failed = []
    for name in ("cwru", "mfpt", "pu"):
        try:
            X, y, runs, _ = get_windows(name)
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
            print(f"\n=== {name.upper()} ===\n  unavailable: {type(exc).__name__}: {exc}")
            print(f"  expected layout: {HINT[name]}")
            failed.append(name)
            continue
        report(name, X, y, runs)

    if failed:
        print(f"\n{len(failed)} of 3 corpora unavailable: {', '.join(failed)}")
        sys.exit(1)
    print("\nOK: all three corpora load, window and pass the class checks.")
