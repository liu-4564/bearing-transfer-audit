"""Experiment orchestration.

Protocols:
  random : window-level random split, the literature default, kept as the
           inflated baseline
  run    : split by run; windows of one .mat/run never straddle the split
  cross  : cross-dataset transfer, trained on the source and tested on the target

Sweepable dimensions:
  --norm window|global   per-window versus training-set global normalisation
  --adapt K              fine-tune on K samples per class in the target (0 = none)
  --snr DB               add white Gaussian noise at this SNR at test time
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config
from .datasets import (Normalizer, add_noise, build_windows, load_cwru, load_mfpt,
                       load_pu)
from .train import (finetune_torch, fit_torch, fit_xgb, metrics, predict_torch,
                    predict_xgb)


# ---------------- data cache ----------------
def get_windows(name):
    """Return raw (unnormalised) windows, cached per corpus."""
    path = os.path.join(config.DATA_DIR, f"_raw_{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["X"], z["y"], z["runs"], z["ds"]
    recs = {"cwru": load_cwru, "mfpt": load_mfpt, "pu": load_pu}[name]()
    X, y, runs, ds = build_windows(recs)
    np.savez_compressed(path, X=X, y=y, runs=runs, ds=ds)
    return X, y, runs, ds


# ---------------- splits ----------------
def split_random(y, runs, seed=None):
    seed = config.SEED if seed is None else seed
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)


def split_by_run(y, runs, seed=None, test_size=0.3):
    seed = config.SEED if seed is None else seed
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr, te), = gss.split(np.zeros(len(y)), y, groups=runs)
    return tr, te


def _sample_k_per_class(X, y, k, seed=None):
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    sel = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        sel.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    sel = np.concatenate(sel)
    rng.shuffle(sel)
    return X[sel], y[sel]


# ---------------- a single run ----------------
def _fit_predict(model, Xtr, ytr, Xte, yte, adapt_pack=None):
    """Fit and predict; with adapt_pack=(Xa, ya), fine-tune on a few target samples."""
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr)
    tr2, va = train_test_split(np.arange(len(ytr)), test_size=0.15,
                               random_state=config.SEED, stratify=ytr)
    if model == "xgb":
        if adapt_pack is not None:
            Xtr = np.concatenate([Xtr, adapt_pack[0]])
            ytr = np.concatenate([ytr, adapt_pack[1]])
        clf = fit_xgb(Xtr, ytr)
        pred = predict_xgb(clf, Xte)
    else:
        net = fit_torch(model, Xtr[tr2], ytr[tr2], Xtr[va], ytr[va])
        if adapt_pack is not None and len(adapt_pack[1]):
            net = finetune_torch(net, *adapt_pack)
        pred = predict_torch(net, Xte)
    return metrics(yte, pred)


def run_one(dataset=None, protocol="random", model="cnn", norm="window",
            snr=None, adapt=0, train_on=None, test_on=None, tag="", force=False):
    t0 = time.time()
    if protocol == "cross":
        key = f"cross_{train_on}2{test_on}_{model}_{norm}"
    else:
        key = f"{protocol}_{dataset}_{model}_{norm}"
    if adapt:
        key += f"_k{adapt}"
    if snr is not None:
        key += f"_snr{snr}"
    if tag:
        key += f"_{tag}"

    path = os.path.join(config.RESULTS_DIR, f"{key}.json")
    if os.path.exists(path) and not force:
        return key, None            # idempotent: skip what is already done, so a sweep can be resumed

    if protocol == "cross":
        Xs, ys, _, _ = get_windows(train_on)
        Xt, yt, runs_t, _ = get_windows(test_on)
        normz = Normalizer(norm).fit(Xs)
        Xs, Xt = normz.transform(Xs), normz.transform(Xt)
        # split the target by run into an adapt pool and a test pool, so that no
        # fine-tuning sample is ever evaluated
        ad_idx, te_idx = split_by_run(yt, runs_t, test_size=0.6)
        adapt_pack = (_sample_k_per_class(Xt[ad_idx], yt[ad_idx], adapt)
                      if adapt > 0 else None)
        res = _fit_predict(model, Xs, ys, add_noise(Xt[te_idx], snr),
                           yt[te_idx], adapt_pack)
    else:
        X, y, runs, _ = get_windows(dataset)
        normz = Normalizer(norm).fit(X)
        X = normz.transform(X)
        tr, te = (split_random(y, runs) if protocol == "random"
                  else split_by_run(y, runs))
        res = _fit_predict(model, X[tr], y[tr], add_noise(X[te], snr), y[te])

    res["elapsed_s"] = round(time.time() - t0, 1)
    res["model"] = model
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{key}] acc={res['acc']:.3f} bacc={res['bacc']:.3f} ({res['elapsed_s']}s)",
          flush=True)
    return key, res


# ---------------- batch ----------------
def _save_summary(summary_file="summary.json"):
    out = {}
    for fn in sorted(os.listdir(config.RESULTS_DIR)):
        if not fn.endswith(".json") or fn.startswith("summary"):
            continue
        with open(os.path.join(config.RESULTS_DIR, fn)) as f:
            d = json.load(f)
        out[fn[:-5]] = {k: d[k] for k in ("acc", "bacc", "f1")}
    with open(os.path.join(config.RESULTS_DIR, summary_file), "w") as f:
        json.dump(out, f, indent=2)
    return out


ALL_DS = ("cwru", "mfpt", "pu")


def run_core(norms=("window", "global"), datasets=ALL_DS):
    for norm in norms:
        for ds in datasets:
            for proto in ("random", "run"):
                for model in ("xgb", "cnn", "resnet"):
                    run_one(ds, proto, model, norm=norm)
        for tr_ds in datasets:
            for te_ds in datasets:
                if tr_ds == te_ds:
                    continue
                for model in ("xgb", "cnn", "resnet"):
                    run_one(protocol="cross", model=model, norm=norm,
                            train_on=tr_ds, test_on=te_ds)
    _save_summary()


def run_adapt(ks=(5, 10, 20), norms=("window", "global"), seeds=(42, 7, 2024),
              datasets=ALL_DS):
    """The few-shot sweep. Training on a handful of samples is noisy, so seeds are
averaged and the spread is reported as one standard deviation."""
    base_seed = config.SEED
    try:
        for seed in seeds:
            config.SEED = seed
            for norm in norms:
                for tr_ds in datasets:
                    for te_ds in datasets:
                        if tr_ds == te_ds:
                            continue
                        for model in ("xgb", "cnn", "resnet"):
                            for k in ks:
                                run_one(protocol="cross", model=model, norm=norm,
                                        adapt=k, train_on=tr_ds, test_on=te_ds,
                                        tag=f"s{seed}")
    finally:
        config.SEED = base_seed
    _save_summary()


def run_snr(snrs=(8, 4, 0, -4), norm="global", model="cnn"):
    for ds in ALL_DS:
        for snr in snrs:
            run_one(ds, "run", model, norm=norm, snr=snr)
    for tr_ds in ALL_DS:
        for te_ds in ALL_DS:
            if tr_ds == te_ds:
                continue
            for snr in snrs:
                run_one(protocol="cross", model=model, norm=norm, snr=snr,
                        train_on=tr_ds, test_on=te_ds)
    _save_summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(ALL_DS))
    ap.add_argument("--protocol", choices=["random", "run", "cross"], default="random")
    ap.add_argument("--model", choices=["xgb", "cnn", "resnet"], default="cnn")
    ap.add_argument("--norm", choices=["window", "global"], default="window")
    ap.add_argument("--adapt", type=int, default=0)
    ap.add_argument("--snr", type=float)
    ap.add_argument("--train-on", dest="train_on", choices=list(ALL_DS))
    ap.add_argument("--test-on", dest="test_on", choices=list(ALL_DS))
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--sweep", choices=["adapt", "snr"])
    a = ap.parse_args()
    print("device:", config.DEVICE, flush=True)
    if a.core:
        run_core()
    elif a.sweep == "adapt":
        run_adapt()
    elif a.sweep == "snr":
        run_snr()
    elif a.protocol == "cross":
        run_one(protocol="cross", model=a.model, norm=a.norm, adapt=a.adapt,
                snr=a.snr, train_on=a.train_on, test_on=a.test_on)
    else:
        run_one(a.dataset, a.protocol, a.model, norm=a.norm, adapt=a.adapt, snr=a.snr)
