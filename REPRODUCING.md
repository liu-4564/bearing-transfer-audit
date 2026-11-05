# Reproducing the numbers

This file records what we actually checked, on which machine, and where the
reproduction stops being exact. It is deliberately specific: a claim of
reproducibility is only useful if the places it fails are named.

## What was verified before release

All checks were run on the machine the experiments were produced on:

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11 (`D:\conda_envs\pytorch_new`) |
| GPU | single NVIDIA GPU, CUDA 12.6 |
| numpy / scipy / scikit-learn | 1.26.4 / 1.13.1 / 1.6.1 |
| torch | 2.8.0+cu126 |
| xgboost | 2.1.4 |
| matplotlib | 3.9.4 |

Four things were checked, in order of increasing strictness.

**1. The data loader.** `python -m src.smoke` passes on all three corpora and
reports the window counts the paper assumes:

```
CWRU  4942 windows  32 runs   [healthy 1237, inner 2468, outer 1237]
MFPT   890 windows  20 runs   [healthy 260, inner 315, outer 315]
PU    2709 windows  60 runs   [healthy 903, inner 903, outer 903]
```

**2. The tables.** `python -m src.summarize` regenerates `results/tables.md`
from the 436 configuration files. The output is byte-identical to the committed
one, including the two correlation coefficients and their p-values.

**3. The figures.** `python -m src.make_figures` regenerates six of the nine
figures from `results/` alone; the other three are drawn from the waveforms and
need the corpora. Of the six, both the PNGs and the PDFs come out byte-identical
to the committed ones, checked by hash. The PDFs carry no creation date —
`plotstyle.save` suppresses it — precisely so that two people regenerating the
same figure get the same file.

**4. A training run end to end.** `xgb` re-run on CWRU→MFPT, global
normalisation, k=0 gives balanced accuracy `0.3321`, against `0.3321` in the
archive. Bit-identical, as expected: XGBoost on 21 deterministic features has
no nondeterministic step.

## Caveat 1 — the deep models are not bit-reproducible by default

The 1D-CNN and 1D-ResNet numbers in `results/` were produced under PyTorch's
default cuDNN behaviour (`benchmark = True`), which lets cuDNN pick a
convolution algorithm at runtime and does not guarantee the same reduction
order twice. Re-running the same configuration on the same GPU therefore
drifts slightly:

| Configuration | Archive | Two `BITEXACT_TORCH=1` re-runs | Δ vs archive |
|---|---|---|---|
| CWRU→MFPT, global, 1D-CNN | 0.068054 | 0.065136 / 0.065136 | 0.0029 |
| CWRU→MFPT, global, 1D-ResNet | 0.042707 | 0.039455 / 0.039455 | 0.0033 |

Two points follow. The two re-runs are **identical to each other to six
decimals**, so the non-determinism is a one-off difference between the archive
and any later run, not scatter between runs. And the archive differs by about
0.003, which is small enough not to move any conclusion here — the effects the
paper reports are 0.2 to 0.9 in size — but large enough that a reader
re-running the grid will not match the last digit.

**To get bit-identical deep-model re-runs**, set the environment variable

```bash
BITEXACT_TORCH=1 python -m src.run_experiment --core
```

which switches `cudnn.benchmark` off, `cudnn.deterministic` on, and enables
`torch.use_deterministic_algorithms(warn_only=True)`. The cost is a slower
convolution path. The numbers you get will be reproducible but will sit about
0.003 from the archived ones, because the archive predates the switch.

CUDA 10.2 and later will print a `CUBLAS_WORKSPACE_CONFIG` warning in this
mode. It is harmless here; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` to silence it.

## Caveat 2 — noise runs are not clean runs

Result files whose names contain `_snr` come from evaluating on a test set with
added white Gaussian noise at that SNR. They belong to the noise sweep only.

They are **excluded from every clean baseline** in the paper. This matters
because the filter is easy to get wrong: the four cross-dataset CNN files
`cross_*_cnn_global_snr{8,4,0,-4}.json` are cross-dataset runs (so they carry
`k = 0` in their config) *and* noised (so they should not count as clean). A
filter on `k == 0` alone silently averages noised accuracy into the clean
cross-dataset baseline and moves the headline numbers by up to 0.05.

`src/summarize.py` and `src/make_figures.py` both filter on
`record["snr"] is None`. If you write your own analysis over `results/`, do the
same.

## Caveat 3 — how runs are pooled

A configuration is repeated over seeds, and the repeats are not perfectly
balanced. At `k = 0`, `k = 5` and `k = 10` every direction carries the same
number of runs for each model, so pooling flat and pooling per direction give
the same answer, and the correlation in Table 1 is unaffected either way.

At `k = 20` two directions (CWRU→MFPT and MFPT→CWRU) carry a fourth seed for
the feature model, so it has 20 runs where the networks have 18. The paper
therefore averages **per direction first, then across the six directions**
(`model_family_means` in `src/make_figures.py`, imported by `src/summarize.py`).
Flat pooling of the 20 runs gives 0.876 for the feature model at `k = 20`
instead of the 0.887 quoted in §3.5, because the two over-sampled directions
happen to be the hardest ones. Both numbers come from the same files; the
difference is which runs get counted twice.

## What is not reproducible without the raw data

`results/` is complete, so the tables and six of the nine figures rebuild from
the repository alone. Both were checked from a checkout with no corpora in it:

- `python -m src.summarize` regenerates `results/tables.md`, byte-identical,
  including the full-strength correlation of -0.956.
- `python -m src.make_figures` regenerates figures 1, 3, 4, S2, S3 and the
  graphical abstract. All six PNGs match the committed ones exactly. Figures 2,
  5 and S1 are skipped with a message.

The one statistic that is not stored as a result file is the rig distance of
§2.5, which is computed from the waveforms. Its values are cached in
`results/rig_distances.json`, and `rig_distances()` falls back to that file when
the corpora are absent; the cache is only read in that case, so regenerating
with the data present always overwrites it. That is what lets the correlation in
Table 1 survive a data-free checkout.

Three figures are drawn from the signals themselves and do need the corpora:
Fig. 2 (waveforms, envelope spectra, spectrograms), Fig. S1 (per-window RMS and
kurtosis distributions) and Fig. 5 (t-SNE of per-window descriptors). They are
committed as PDF and PNG, so nothing is lost for a reader; only redrawing them
requires the data.

Retraining needs all three corpora, which are third-party and not redistributed
here — see `data/README.md` for the layout the loader expects. `src/fetch_pu.py`
downloads the Paderborn archive; CWRU and MFPT have to be fetched by hand from
their own sites because neither offers a stable direct-download URL.

So a data-free checkout behaves like this:

```bash
python -m src.summarize      # OK, writes results/tables.md
python -m src.make_figures   # OK, writes 6 of 9 figures, skips 2, 6, 7
python -m src.smoke          # reports all three corpora unavailable
                             # and prints the expected layout for each
```
