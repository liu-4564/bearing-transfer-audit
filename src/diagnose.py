"""Compare per-class signal statistics between two corpora, to locate the
physical reason a cross-corpus transfer fails."""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from . import config
from .run_experiment import get_windows


def stats(X):
    rms = np.sqrt((X ** 2).mean(1))
    kurt = kurtosis(X, axis=1)
    env_kurt = np.array([kurtosis(np.abs(hilbert(x))) for x in X])
    # dominant frequency: the spectral peak within 0-2000 Hz
    S = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], 1 / config.FS_TARGET)
    mask = freqs <= 2000
    peak_f = freqs[mask][S[:, mask].argmax(1)]
    return dict(rms=rms, kurt=kurt, env_kurt=env_kurt, peak_f=peak_f)


def main():
    for ds in ("cwru", "mfpt"):
        X, y, runs, _ = get_windows(ds)
        st = stats(X)
        print(f"\n===== {ds.upper()}  (windows={len(X)}, runs={len(set(runs))}) =====")
        print(f"{'class':10s} {'n':>5s} {'RMS(med)':>10s} {'kurt(med)':>10s} "
              f"{'envKurt(med)':>12s} {'peakF(med)':>11s}")
        for c in np.unique(y):
            m = y == c
            print(f"{config.CLASSES[c]:10s} {m.sum():5d} {np.median(st['rms'][m]):10.4f} "
                  f"{np.median(st['kurt'][m]):10.2f} {np.median(st['env_kurt'][m]):12.2f} "
                  f"{np.median(st['peak_f'][m]):11.0f}")

    # are the nominal classes comparable across corpora? Is the healthy class in
    # MFPT already more fault-like to begin with?
    cw_X, cw_y, _, _ = get_windows("cwru")
    mf_X, mf_y, _, _ = get_windows("mfpt")
    cs, ms = stats(cw_X), stats(mf_X)
    print("\n----- healthy class, across corpora -----")
    for name, st, yy in (("CWRU", cs, cw_y), ("MFPT", ms, mf_y)):
        m = yy == 0
        print(f"{name} healthy: RMS={np.median(st['rms'][m]):.4f}  "
              f"kurt={np.median(st['kurt'][m]):.2f}  "
              f"envKurt={np.median(st['env_kurt'][m]):.2f}")
    print("\n----- fault classes, across corpora -----")
    for name, st, yy in (("CWRU", cs, cw_y), ("MFPT", ms, mf_y)):
        m = yy > 0
        print(f"{name} fault  : RMS={np.median(st['rms'][m]):.4f}  "
              f"kurt={np.median(st['kurt'][m]):.2f}  "
              f"envKurt={np.median(st['env_kurt'][m]):.2f}")


if __name__ == "__main__":
    main()
