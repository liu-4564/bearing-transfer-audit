# Data

The three corpora are third-party and are **not** redistributed in this
repository. Each keeps its own licence and citation requirement. What follows is
the layout the code expects once you have them.

Everything goes under `data/`, which is git-ignored except for this file.

## CWRU — Case Western Reserve University Bearing Data Center

Source: <https://engineering.case.edu/bearingdatacenter>

`src/datasets.py` reads the 12 kHz drive-end records from a mirror tree laid out
as:

```
data/cwru/CWRU-dataset-main/
├── Normal/
│   └── {id}_Normal_{load}.mat
└── 12k_Drive_End_Bearing_Fault_Data/
    ├── B/{size}/{id}_{load}.mat
    ├── IR/{size}/{id}_{load}.mat
    └── OR/{size}/[@{pos}/]{id}[@pos]_{load}.mat
```

Only the outer-race records taken at the 6 o'clock position are used, which is
the convention in most of the literature. If your copy uses different file names,
adjust `load_cwru()` in `src/datasets.py`; the file-id map is in `src/config.py`.

## MFPT — Machinery Failure Prevention Technology Society benchmark

Source: the Society's fault-dataset package, widely mirrored. The code walks
`data/mfpt/` recursively for `*.mat` files and infers the class from the file
name (`src/datasets._mfpt_label`):

```
data/mfpt/
└── RollingElementBearingFaultDiagnosis-Data-master/
    └── .../*.mat
```

Baseline records are sampled at 97,656 Hz and fault records at 48,828 Hz; both
are resampled to 12 kHz on load.

## PU — Paderborn University Bearing Data Center

Source: <https://groups.uni-paderborn.de/kat/BearingDataCenter/>

The downloader fetches the archives directly and unpacks the three bearings used
in the paper (one healthy, one inner-race, one outer-race, at the
N15_M07_F10 operating point):

```bash
python -m src.fetch_pu K001 KA04 KI04
```

The archives land in `data/pu/` as `K001.rar`, `KA04.rar`, `KI04.rar`; extract
them in place so that `data/pu/K001/`, `data/pu/KA04/` and `data/pu/KI04/`
exist. Vibration channel 1 is used, sampled at 64 kHz.

The downloader uses HTTP range requests over several connections because a
single connection to that host is throttled to roughly 11 kB/s.

## Cached window tensors

On first load, `src/datasets.build_windows` writes `data/_raw_{name}.npz` (a few
megabytes each) so that later runs skip parsing. Delete them to force a rebuild.

## Sanity check

```bash
python -m src.smoke
```

should print, for each corpus, the number of runs and windows and the class
balance. If a corpus is missing, the script says so rather than failing silently.
