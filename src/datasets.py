"""Dataset loading: CWRU, MFPT and Paderborn -> one uniform record.

Record fields:
  signal : float32 1-D vibration signal (not resampled)
  fs     : native sampling rate
  label  : int (see config.CLASSES)
  run_id : unique run identifier; one .mat file is one run, which is what makes
           run-wise splitting possible
  dataset: 'cwru' | 'mfpt'
"""
import os
import numpy as np
from scipy.io import loadmat

from . import config


def _resample(x, fs_in, fs_out=config.FS_TARGET):
    if fs_in == fs_out:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(fs_in), int(fs_out))
    y = resample_poly(x, fs_out // g, fs_in // g)
    return y.astype(np.float32)


def load_cwru(root=None, include_ball=False):
    """Load the 12 kHz drive-end data from a CWRU-dataset-main tree.

    Layout:
      Normal/{id}_Normal_{load}.mat
      12k_Drive_End_Bearing_Fault_Data/{B,IR,OR}/{size}/[@{pos}/]{id}[@pos]_{load}.mat
    Only the @6:00 outer-race files are used, at the centre of the load zone,
    which is what most of the literature reports.
    """
    root = root or os.path.join(config.DATA_DIR, "cwru", "CWRU-dataset-main")
    recs = []

    def add(path, cls, size, load, tag):
        d = loadmat(path)
        key = [k for k in d if k.endswith("DE_time")]
        if not key:
            key = [k for k in d if k.endswith("FE_time")]
        if not key:
            return
        sig = d[key[0]].ravel()
        recs.append(dict(signal=_resample(sig, 12000), fs=config.FS_TARGET,
                         label=config.LABEL2ID[cls], run_id=f"cwru_{tag}",
                         dataset="cwru", meta=dict(size=size, load=load)))

    norm_dir = os.path.join(root, "Normal")
    if os.path.isdir(norm_dir):
        for fn in sorted(os.listdir(norm_dir)):
            if fn.endswith(".mat"):
                fid = fn.split("_")[0]
                load = fn.rsplit("_", 1)[-1].replace(".mat", "")
                add(os.path.join(norm_dir, fn), "healthy", 0, load, f"N{fid}")

    fault_root = os.path.join(root, "12k_Drive_End_Bearing_Fault_Data")
    cls_of = {"IR": "inner", "OR": "outer", "B": "ball"}
    if os.path.isdir(fault_root):
        for cdir in sorted(os.listdir(fault_root)):
            cls = cls_of.get(cdir)
            if cls is None or (cls == "ball" and not include_ball):
                continue
            for dirpath, _, files in os.walk(os.path.join(fault_root, cdir)):
                for fn in sorted(files):
                    if not fn.endswith(".mat"):
                        continue
                    if cdir == "OR" and not ("@6" in fn or os.path.basename(dirpath) == "@6"):
                        continue
                    size = os.path.basename(os.path.dirname(dirpath if "@" in dirpath else dirpath))
                    fid = fn.split("_")[0].split("@")[0]
                    load = fn.rsplit("_", 1)[-1].replace(".mat", "")
                    add(os.path.join(dirpath, fn), cls, size, load, f"{cdir}{fid}")
    return recs


def _mfpt_label(fname):
    f = fname.lower()
    if "baseline" in f:
        return "healthy"
    if "innerrace" in f:
        return "inner"
    if "outerrace" in f:
        return "outer"
    return None


def load_mfpt(root=None):
    root = root or os.path.join(config.DATA_DIR, "mfpt")
    recs = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".mat"):
                continue
            cls = _mfpt_label(fn)
            if cls is None:
                continue
            d = loadmat(os.path.join(dirpath, fn), squeeze_me=True, struct_as_record=False)
            b = d["bearing"]
            sig = np.asarray(b.gs, dtype=np.float64).ravel()
            fs = int(np.asarray(b.sr).ravel()[0])
            run = f"mfpt_{os.path.splitext(fn)[0]}"
            recs.append(dict(signal=_resample(sig, fs), fs=config.FS_TARGET,
                             label=config.LABEL2ID[cls], run_id=run,
                             dataset="mfpt", meta=dict(file=fn)))
    return recs


def build_windows(recs, window=config.WINDOW, overlap=config.OVERLAP, max_per_run=None):
    """Cut every record into fixed-length windows; return raw X, y, run_ids, ds_ids.

    Normalisation is deliberately left to the experiment layer, because per-window
    versus global scaling is one of the core variables here.
    """
    if not recs:
        raise FileNotFoundError(
            "no records matched: the corpus directory is empty or laid out "
            "differently from data/README.md")
    step = int(window * (1 - overlap))
    X, y, runs, dss = [], [], [], []
    for r in recs:
        sig = r["signal"]
        n = (len(sig) - window) // step + 1
        if max_per_run:
            n = min(n, max_per_run)
        for i in range(max(n, 0)):
            w = sig[i * step: i * step + window].astype(np.float32)
            if w.std() < 1e-8:
                continue
            X.append(w)
            y.append(r["label"])
            runs.append(r["run_id"])
            dss.append(r["dataset"])
    return (np.stack(X), np.asarray(y), np.asarray(runs), np.asarray(dss))


class Normalizer:
    """The two normalisation strategies, one of the core variables of this study:

    window : per-window z-score -- removes absolute amplitude, keeps waveform shape,
             which is the usual practice in the literature
    global : one mean and s.d. from the whole training set -- preserves relative
             energy between windows
    """

    def __init__(self, mode="window"):
        assert mode in ("window", "global")
        self.mode = mode
        self.mu = 0.0
        self.sd = 1.0

    def fit(self, X):
        if self.mode == "global":
            Xc = X - X.mean(axis=1, keepdims=True)   # remove the DC offset before estimating the scale
            self.mu, self.sd = 0.0, float(Xc.std() + 1e-8)
        return self

    def transform(self, X):
        Xc = X - X.mean(axis=1, keepdims=True)       # both modes remove the DC offset
        if self.mode == "window":
            s = Xc.std(axis=1, keepdims=True) + 1e-8
        else:
            s = self.sd
        return (Xc / s).astype(np.float32)


def add_noise(X, snr_db, seed=config.SEED):
    """Add white Gaussian noise at a given SNR, for test-time robustness."""
    if snr_db is None:
        return X
    rng = np.random.default_rng(seed)
    p_sig = np.mean(X ** 2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (X + rng.normal(0, np.sqrt(p_noise), X.shape)).astype(np.float32)

# ---------------- Paderborn (PU) ----------------
PU_CLASS = {"K": "healthy", "KA": "outer", "KI": "inner", "KB": "outer"}
PU_FS = 64000


def load_pu(root=None, condition="N15_M07_F10", channel="vibration_1"):
    """Paderborn: every .mat file is one record, i.e. one run.

    Layout: data/pu/extracted/{bearing}/{condition}_{bearing}_{i}.mat
    Structure: the top-level Y holds the measurement channels and Name labels them
    (force / speed / torque / phase_current / vibration_1 ...). X is only the time
    grid shared by the channels and carries no measurements.
    Sampling: 64 kHz on the vibration channel, 4 s per record.
    Naming: K = healthy, KA = outer-race damage, KI = inner-race damage,
    KB = combined inner and outer damage.
    """
    root = root or os.path.join(config.DATA_DIR, "pu", "extracted")
    recs = []
    if not os.path.isdir(root):
        return recs
    for bearing in sorted(os.listdir(root)):
        bdir = os.path.join(root, bearing)
        if not os.path.isdir(bdir):
            continue
        prefix = "KA" if bearing.startswith("KA") else ("KI" if bearing.startswith("KI")
                 else ("KB" if bearing.startswith("KB") else "K"))
        cls = PU_CLASS.get(prefix)
        if cls is None:
            continue
        for fn in sorted(os.listdir(bdir)):
            if not fn.endswith(".mat") or condition not in fn:
                continue
            d = loadmat(os.path.join(bdir, fn), squeeze_me=True, struct_as_record=False)
            rec = d[[k for k in d if not k.startswith("__")][0]]
            chans = np.atleast_1d(rec.Y)
            pick = [c for c in chans if str(c.Name) == channel] or \
                   [c for c in chans if str(c.Name).startswith("vibration")]
            if not pick:
                continue
            sig = np.asarray(pick[0].Data, dtype=np.float64).ravel()
            recs.append(dict(signal=_resample(sig, PU_FS), fs=config.FS_TARGET,
                             label=config.LABEL2ID[cls],
                             run_id=f"pu_{bearing}_{fn.rsplit('_', 1)[-1][:-4]}",
                             dataset="pu", meta=dict(file=fn, bearing=bearing)))
    return recs
