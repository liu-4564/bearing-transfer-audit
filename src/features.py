"""Hand-crafted features: time-domain statistics, band energies and envelope terms.

Dataset-agnostic: no bearing geometry is required, so the same extractor serves
all three corpora.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from . import config


def extract_features(X, fs=config.FS_TARGET):
    """X: (n, window) normalised signals -> (n, n_features)"""
    n = X.shape[0]
    freqs = np.fft.rfftfreq(X.shape[1], 1 / fs)
    feats = []
    for i in range(n):
        x = X[i]
        # time domain
        rms = np.sqrt(np.mean(x ** 2))
        f = [rms, kurtosis(x), skew(x), np.max(np.abs(x)),
             np.max(np.abs(x)) / (rms + 1e-12),                    # crest factor
             np.ptp(x), np.mean(np.abs(x)), np.std(x)]
        # frequency domain: spectral centroid, spectral entropy, 8 equal-width band ratios
        S = np.abs(np.fft.rfft(x)) ** 2
        p = S / (S.sum() + 1e-12)
        f.append(np.sum(freqs * p) / fs)                            # spectral centroid
        f.append(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))   # spectral entropy
        bands = np.array_split(S, 8)
        f.extend([b.sum() / (S.sum() + 1e-12) for b in bands])
        # envelope spectrum: tallest low-frequency peak and its location
        env = np.abs(hilbert(x))
        env = env - env.mean()
        ES = np.abs(np.fft.rfft(env)) ** 2
        ef = np.fft.rfftfreq(len(x), 1 / fs)
        low = ef <= 500
        idx = np.argmax(ES[low])
        f.extend([ES[low][idx] / (ES.sum() + 1e-12), ef[low][idx] / 500.0,
                  ES[low].sum() / (ES.sum() + 1e-12)])
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = (["rms", "kurtosis", "skew", "peak", "crest", "p2p", "mean_abs", "std",
                  "spec_centroid", "spec_entropy"] +
                 [f"band{i}" for i in range(8)] +
                 ["env_peak_ratio", "env_peak_freq", "env_lowband_ratio"])
