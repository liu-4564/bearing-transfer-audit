# How much of "99% accuracy" survives a change of machine?

A three-rig audit of rolling-element bearing fault benchmarks.

This repository contains the code, the cached result files and the figures for a
paper that asks a deliberately narrow question: when a fault classifier is moved
from the machine it was trained on to a different machine, how much of its
reported accuracy survives, and does the size of the loss follow from anything
measurable *before* deployment?

**Short answer.** Within a dataset the benchmark is saturated (0.999–1.000) and
says nothing about diagnostic skill. Across machines, balanced accuracy falls
anywhere between 0.039 and 0.612 depending on the pair of rigs, and the ordering
follows the distance between the two rigs in a standardised descriptor space
(Spearman ρ = −0.956, p = 0.003 over six directions) rather than the choice of
architecture.

## What is in here

| Path | Contents |
|---|---|
| `src/` | All code. `datasets.py` loads and harmonises the three corpora; `run_experiment.py` runs the grid; `make_figures.py` builds the figures; `summarize.py` writes `results/tables.md`. |
| `results/` | One JSON per configuration (436 of them, plus `summary.json` and the distance cache `rig_distances.json`) and the aggregated tables. These are the numbers in the paper; you do not need the raw data to check them. See `results/README.md` for the naming scheme. |
| `figures/` | Nine figures, vector PDF and 600 dpi PNG. |
| `manuscript/` | The submitted text, figure captions and the pre-submission enquiry. |
| `data/` | Empty except for a README. The three corpora are third-party and are not redistributed here. |
| `REPRODUCING.md` | What was verified before release, and the three places where an exact re-run will not match (deep-model nondeterminism, noise runs, aggregation order). Read this before you compare your numbers with ours. |

```
bearing-transfer-audit/
├── src/                  14 files, 2,230 lines
│   ├── config.py         class labels, sampling rates, window length, paths
│   ├── datasets.py       loading, resampling, windowing, normalisation
│   ├── features.py       the 21 descriptors
│   ├── models.py         XGBoost, 1D-CNN (WDCNN), 1D-ResNet
│   ├── train.py          fit/predict wrappers, BITEXACT_TORCH switch
│   ├── run_experiment.py the grid and the two sweeps
│   ├── summarize.py      results/*.json -> results/tables.md
│   ├── make_figures.py   results/*.json -> figures/* (half the code base)
│   ├── illus.py          schematic primitives for Fig. 1
│   ├── plotstyle.py      rcParams, Crameri maps, Okabe-Ito palette
│   ├── diagnose.py       class-conditional signal statistics behind Fig. S1
│   ├── fetch_pu.py       downloader for the Paderborn archive
│   └── smoke.py          loader self-check
├── results/              436 configuration JSON + summary.json + tables.md
├── figures/              9 × {pdf, png}
├── manuscript/           draft_submission.md, figure_captions.md, presubmission_enquiry.md
├── data/                 README.md only
├── REPRODUCING.md
├── CITATION.cff
├── LICENSE               MIT for code, CC BY 4.0 for text and figures
├── requirements.txt
└── Makefile
```

## Reproducing it

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. fetch the three corpora (see data/README.md), then:
python -m src.smoke                 # checks that all three load and window

# 2. the main grid: 3 datasets x 2 split protocols x 3 models x 2 normalisations,
#    plus all six cross-dataset directions. ~40 min on one GPU, longer on CPU.
python -m src.run_experiment --core

# 3. few-shot adaptation sweep (k = 5, 10, 20 per class; three seeds)
python -m src.run_experiment --sweep adapt

# 4. noise robustness sweep (SNR = 8, 4, 0, -4 dB)
python -m src.run_experiment --sweep snr

# 5. tables and figures
python -m src.summarize
python -m src.make_figures
```

`make` wraps the same steps (`make setup`, `make smoke`, `make core`, `make
figures`, `make all`).

Every run is keyed by configuration and cached: re-running a step skips
configurations whose JSON already exists. Pass `force=True` in code, or delete
the file, to recompute one.

## Design choices worth knowing before you read the numbers

- **Three classes only** (healthy, inner race, outer race). MFPT has no
  rolling-element faults, so keeping them would break the shared label space that
  transfer requires.
- **The run is the evaluation unit**, not the window. A run is one bearing
  condition under one load; windows from the same run share bearing, mounting and
  often the same instant of time.
- **Two normalisations.** Per-window z-scoring removes absolute amplitude; global
  normalisation preserves relative energy. They are not interchangeable and the
  paper reports both, defaulting to global for transfer.
- **Aggregation is by model family.** A configuration is first averaged over its
  seeds, then the three families (feature+GBM, 1D-CNN, 1D-ResNet) are averaged
  with equal weight, so a model that happened to be run more often does not
  outvote the others.
- **Noise runs are excluded from clean baselines.** Files carrying `_snr` come
  from a noised test set and are used only by the noise sweep.

## Data

All three corpora are public and third-party; none were generated here.

- **CWRU** — Case Western Reserve University Bearing Data Center,
  <https://engineering.case.edu/bearingdatacenter>
- **MFPT** — Machinery Failure Prevention Technology Society benchmark,
  distributed with the Society's fault-dataset package
- **PU** — Paderborn University Bearing Data Center,
  <https://groups.uni-paderborn.de/kat/BearingDataCenter/>

Each retains its own licence; check the source before redistributing. See
`data/README.md` for the exact layout this code expects.

## Citation

See `CITATION.cff`. If the paper is not yet published, cite the repository.

## Licence

Code: MIT (see `LICENSE`). Text and figures: CC BY 4.0.
