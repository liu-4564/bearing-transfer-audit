"""Paper figures, journal-grade and evidence-led.

Fig 1  the three evaluation protocols (study design)
Fig 2  signal triplet: waveform / envelope spectrum / time-frequency map
Fig 3  within-corpus protocol gap: paired forest plot with effect sizes
Fig 4  cross-corpus transfer matrices: absolute level at k=0, and gain from tuning
Fig 5  few-shot recovery: six small multiples with ±1 s.d. bands
Fig 6  why transfer fails: class-conditional ridges / kurtosis ECDF / distance
Fig 7  feature-space embedding and confusion matrices
Fig S1 noise robustness heatmap (supplementary)

Consistent throughout: Times faces, Okabe-Ito colours, 7-8 pt, inward ticks, no
top or right spine, vector PDF plus 600 dpi PNG.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from scipy.signal import hilbert, spectrogram
from scipy.stats import gaussian_kde

from . import config
from .plotstyle import (CM_ALT, CM_SEQ, CM_SEQ2, DOUBLE, MODEL_C, ONE_HALF,
                        SINGLE, C, apply, panel, save, text_on)

apply()

PAT = re.compile(
    r"^(?:cross_(?P<src>cwru|mfpt|pu)2(?P<tgt>cwru|mfpt|pu)|(?P<proto>random|run)_(?P<ds>cwru|mfpt|pu))"
    r"_(?P<model>xgb|cnn|resnet)_(?P<norm>window|global)"
    r"(?:_k(?P<k>\d+))?(?:_snr(?P<snr>-?\d+))?(?:_s(?P<seed>\d+))?$")

MODEL_LABEL = {"xgb": "Feature + GBM", "cnn": "1D-CNN", "resnet": "1D-ResNet"}
DS_LABEL = {"cwru": "CWRU", "mfpt": "MFPT", "pu": "PU"}
DS_C = {"cwru": C["blue"], "mfpt": C["vermillion"], "pu": C["green"]}
CLS_MARK = ["o", "s", "^"]
CLASS_LABEL = ["Healthy", "Inner race", "Outer race"]
CLASS_C = [C["grey"], C["orange"], C["purple"]]

# transfer directions: those that do not involve PU first, then those that do,
# which is the order Fig. 5 expects
DIR_ORDER = [("cwru", "mfpt"), ("mfpt", "cwru"),
             ("cwru", "pu"), ("pu", "cwru"),
             ("mfpt", "pu"), ("pu", "mfpt")]


def load_results():
    out = []
    for fn in os.listdir(config.RESULTS_DIR):
        if not fn.endswith(".json") or fn.startswith("summary"):
            continue
        m = PAT.match(fn[:-5])
        if not m:
            continue
        d = m.groupdict()
        d["k"] = int(d["k"]) if d["k"] else 0
        d["seed"] = int(d["seed"]) if d["seed"] else config.SEED
        d["snr"] = float(d["snr"]) if d["snr"] else None
        with open(os.path.join(config.RESULTS_DIR, fn)) as f:
            r = json.load(f)
        d.update(acc=r["acc"], bacc=r["bacc"], f1=r["f1"], cm=r["cm"])
        out.append(d)
    return out


def datasets_present(rows):
    ds = set()
    for r in rows:
        for key in ("src", "tgt", "ds"):
            if r.get(key):
                ds.add(r[key])
    return [d for d in ("cwru", "mfpt", "pu") if d in ds]


def get_cell(rows, src, tgt, model="resnet", norm="global", k=0):
    if src == tgt:
        sub = [r["bacc"] for r in rows if r.get("ds") == src and r.get("proto") == "run"
               and r["model"] == model and r["norm"] == norm and r.get("snr") is None]
    else:
        sub = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
               and r["model"] == model and r["norm"] == norm and r["k"] == k
               and r.get("snr") is None]
    return float(np.mean(sub)) if sub else np.nan


def _ms(vals):
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=0))


def model_family_means(rows, **filt):
    """Average within each model family first, then weight the families equally.

    Averaging over every result file instead would over-weight whichever model was
    run most often -- the CNN has five seed and noise copies -- quietly turning
    "three models" into "mostly the CNN".
    """
    per = []
    for m in ("xgb", "cnn", "resnet"):
        v = [r["bacc"] for r in rows
             if r["model"] == m and r.get("snr") is None
             and all(r.get(k) == val for k, val in filt.items())]
        if v:
            per.append(float(np.mean(v)))
    if not per:
        return np.nan, np.nan
    return float(np.mean(per)), float(np.std(per, ddof=0))


def rig_distances(max_per_ds=1600, seed=0):
    """Descriptor-space distance between two corpora: standardise the 21 features
    globally, then take the RMS of the Euclidean distances between same-class means.

    One definition, shared by Fig. 6c and the tables, so that the prose and the
    figures cannot drift apart.
    """
    import json

    from .features import extract_features
    from .run_experiment import get_windows

    cache = os.path.join(config.RESULTS_DIR, "rig_distances.json")
    has_raw = all(os.path.exists(os.path.join(config.DATA_DIR, f"_raw_{d}.npz"))
                  for d in ("cwru", "mfpt", "pu"))
    if not has_raw and os.path.exists(cache):
        # with no raw data, fall back to the committed cache so that tables.md and
            # Fig. 6c still rebuild
        with open(cache) as f:
            rec = json.load(f)
        dist = {tuple(k.split("|")): v for k, v in rec["distances"].items()}
        return list(rec["datasets"]), dist

    ds_list = [d for d in ("cwru", "mfpt", "pu")
               if os.path.exists(os.path.join(config.DATA_DIR, f"_raw_{d}.npz"))]
    rng = np.random.default_rng(seed)
    feats = {}
    for name in ds_list:
        X, y, _, _ = get_windows(name)
        idx = rng.choice(len(X), min(max_per_ds, len(X)), replace=False)
        feats[name] = (extract_features(X[idx]), y[idx])
    Z = np.vstack([feats[d][0] for d in ds_list])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    mvec = {}
    for d in ds_list:
        F, y = feats[d]
        Fz = (F - mu) / sd
        mvec[d] = np.vstack([Fz[y == c].mean(0) for c in range(3)])
    dist = {}
    for a in ds_list:
        for b in ds_list:
            if a < b:
                dist[(a, b)] = float(np.sqrt(((mvec[a] - mvec[b]) ** 2).sum(1).mean()))
                dist[(b, a)] = dist[(a, b)]
    with open(cache, "w") as f:
        json.dump({"datasets": ds_list,
                   "distances": {f"{a}|{b}": v for (a, b), v in dist.items()},
                   "max_per_ds": max_per_ds, "seed": seed}, f, indent=2)
    return ds_list, dist


def bw_for(v, target=0.055):
    """Convert an absolute bandwidth (dex) into the relative one gaussian_kde wants,
    so that narrow distributions do not collapse into needles."""
    s = float(np.std(v)) + 1e-9
    return float(np.clip(target / s, 0.12, 6.0))


# ---------------------------------------------------------------- Fig 1
def fig_protocol():
    """Study design: from the physical object to the evaluation protocol."""
    from .illus import (ACC, BLUE, GREEN, GREY, SKY, STEEL, arrow, bearing, card,
                        chip, rig)

    W, H = 69.0, 46.0
    fig = plt.figure(figsize=(6.9, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    # ---- top row: object -> sensor -> signal -> windows
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrowPatch

    R0, CX, CY = 5.9, 7.6, 37.5
    bearing(ax, CX, CY, R0, defect="inner", defect_angle=-90)
    dy = CY - 0.37 * R0                       # the leader runs through the gap between two rolling elements, not across them
    lead = FancyArrowPatch((CX, 30.45), (CX, dy), arrowstyle="-", mutation_scale=1,
                           lw=0.45, color=ACC, zorder=20, shrinkA=0, shrinkB=0)
    lead.set_path_effects([pe.withStroke(linewidth=1.3, foreground="white")])
    ax.add_patch(lead)
    ax.plot([CX], [dy], "o", ms=1.5, color=ACC, zorder=21, mec="white", mew=0.3)
    ax.text(CX, 29.85, "inner-race defect", fontsize=6.2, color=ACC, ha="center")
    ax.text(CX, 44.15, "one bearing", fontsize=6.4, color=GREY, ha="center")

    arrow(ax, (14.2, 37.5), (18.2, 37.5))
    rig(ax, 25.0, 37.0, 11.0, 7.4)
    ax.text(25.0, 31.0, "one test rig", fontsize=6.4, color=GREY, ha="center")

    arrow(ax, (31.4, 37.5), (34.4, 37.5))
    axw = fig.add_axes([34.8 / W, 34.8 / H, 15.6 / W, 6.0 / H])
    t = np.linspace(0, 1, 2400)
    imp = np.zeros_like(t)
    for c in np.arange(0.05, 1.0, 0.105):
        imp += np.exp(-((t - c) / 0.008) ** 2)
    sig = 0.32 * np.sin(2 * np.pi * 33 * t) + 1.35 * imp
    axw.plot(t, sig, lw=0.35, color="#222222")
    for x0 in (0.16, 0.235, 0.31):
        axw.axvspan(x0, x0 + 0.135, color=SKY, alpha=0.30, lw=0)
        axw.axvspan(x0, x0 + 0.135, fill=False, ec=BLUE, lw=0.5)
    axw.set_xlim(0, 1); axw.set_ylim(-1.1, 1.75)
    axw.set_xticks([]); axw.set_yticks([])
    for sp in axw.spines.values():
        sp.set_linewidth(0.5); sp.set_color(STEEL["dark"])
    axw.spines["top"].set_visible(False); axw.spines["right"].set_visible(False)
    ax.text(42.6, 31.0, "vibration record, 171 ms shown", fontsize=6.4, color=GREY,
            ha="center")

    arrow(ax, (51.0, 37.5), (54.0, 37.5))
    for k, yy in enumerate((41.4, 37.9, 34.4)):
        ax.add_patch(plt.Rectangle((55.4 + 0.9 * k, yy - 1.5), 12.4, 3.0,
                                   fc="#EAF2F8", ec=BLUE, lw=0.5, zorder=4))
        xs = np.linspace(0, 1, 120)
        ax.plot(55.4 + 0.9 * k + 0.6 + xs * 11.2,
                yy + 0.55 * np.sin(9 * xs) * np.exp(-2 * ((xs - 0.45) / 0.5) ** 2),
                lw=0.4, color="#444444", zorder=5)
    ax.text(61.6, 31.0, "windows, 50% overlap", fontsize=6.4, color=GREY, ha="center")

    ax.plot([1.0, 68.0], [28.6, 28.6], lw=0.5, color="#E1E5E9", zorder=1)

    # ---- legend, in the band between the rule and the cards
    for lx, lab, col in ((20.0, "training", BLUE), (33.0, "calibration", SKY),
                         (46.6, "held-out test", "#E69F00")):
        ax.add_patch(plt.Rectangle((lx, 26.6), 1.5, 1.3, fc=col, ec="none", zorder=5))
        ax.text(lx + 2.0, 27.25, lab, fontsize=6.4, color="#333333", va="center")

    # ---- bottom row: the three protocols
    CW, CH, GAP = 21.2, 24.0, 2.3
    xs0 = [1.0, 1.0 + CW + GAP, 1.0 + 2 * (CW + GAP)]
    titles = ["Random window split", "Split by run", "Cross-dataset transfer"]
    letters = ["a", "b", "c"]
    cw, ch, step = 1.30, 1.90, 1.38
    cell0 = 4.2
    rows_y = [17.8, 14.2, 10.6, 7.0]

    for ci in range(3):
        x0 = xs0[ci]
        card(ax, x0, 1.6, CW, CH)
        ax.text(x0 + 3.6, 1.6 + CH - 2.1, titles[ci], fontsize=7.0, color="#1A1A1A",
                va="center", ha="left")
        ax.text(x0 + 1.5, 1.6 + CH - 1.95, f"({letters[ci]})", fontsize=7.6,
                fontweight="bold", color="#111111", va="center", ha="left")

        for ri, yy in enumerate(rows_y):
            bearing(ax, x0 + 2.0, yy + ch / 2, 1.15, lw=0.3, z=4)
            for cj in range(10):
                xx = x0 + cell0 + cj * step
                if ci == 0:
                    # exactly 5:5 per row: the panel is about one recording being cut in two,
                    # and a nearly monochrome row would look as if it had not been cut
                    if cj == 0:
                        _pick = set(np.random.default_rng(ri * 97 + 11)
                                    .choice(10, 5, replace=False).tolist())
                    fc = BLUE if cj in _pick else "#E69F00"
                elif ci == 1:
                    fc = BLUE if ri < 3 else "#E69F00"
                else:
                    fc = BLUE if ri < 2 else (SKY if ri == 2 else "#E69F00")
                ax.add_patch(plt.Rectangle((xx, yy), cw, ch, fc=fc, ec="white", lw=0.5,
                                           zorder=5))

        if ci == 0:
            ax.add_patch(FancyBboxPatch((x0 + 1.1, rows_y[3] - 0.85),
                                        CW - 3.4, rows_y[0] - rows_y[3] + ch + 1.7,
                                        boxstyle="round,pad=0,rounding_size=0.8",
                                        fill=False, ec=ACC, lw=0.6,
                                        ls=(0, (2.2, 1.6)), zorder=6))
            ax.text(x0 + CW / 2, 4.2,
                    "each row is one run;\nits windows fall on both sides",
                    fontsize=5.9, color=ACC, ha="center", va="top", linespacing=1.4)
        elif ci == 1:
            for ri, yy in enumerate(rows_y):
                ax.text(x0 + cell0 + 10 * step + 0.35, yy + ch / 2,
                        "train" if ri < 3 else "test",
                        fontsize=6.0, color=GREY, va="center", ha="left")
            ax.text(x0 + CW / 2, 4.2, "all windows of a recording\nstay on one side",
                    fontsize=6.0, color=GREY, ha="center", va="top", linespacing=1.4)
        else:
            ax.text(x0 + cell0, rows_y[0] + ch + 0.50, "source rig", fontsize=6.0,
                    color=GREY, va="bottom")
            ax.text(x0 + cell0, rows_y[2] + ch + 0.50, "target rig", fontsize=6.0,
                    color=GREY, va="bottom")
            for ri, yy in enumerate(rows_y):
                if ri < 2:
                    lab, col = "train", GREY
                elif ri == 2:
                    lab, col = "calib.", "#3C3C3C"
                else:
                    lab, col = "test", "#3C3C3C"
                ax.text(x0 + cell0 + 10 * step + 0.35, yy + ch / 2, lab,
                        fontsize=6.0, color=col, va="center", ha="left")
            ax.text(x0 + CW / 2, 4.2, "no target label reaches training;\nadaptation uses calib. only",
                    fontsize=6.0, color=GREY, ha="center", va="top", linespacing=1.4)

    save(fig, "fig1_protocol")


# ---------------------------------------------------------------- Graphical abstract
def fig_graphical(rows, norm="global"):
    """Graphical abstract: same model, another machine -- how much survives, and what decides it."""
    from .illus import ACC, BLUE, GREY, SKY, STEEL, arrow, bearing, chip, rig

    W, H = 69.0, 22.0
    fig = plt.figure(figsize=(6.9, 2.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    # left: the training rig, neutral grey
    rig(ax, 10.6, 13.4, 11.6, 6.6)
    chip(ax, 10.6, 6.0, "balanced accuracy 1.000", fc="#E7F2EA", tc="#1E6B3A", fs=6.0)
    ax.text(10.6, 3.0, "train on rig A", fontsize=6.4, color=GREY, ha="center")

    arrow(ax, (16.9, 13.2), (21.3, 13.2), color="#333333", lw=1.0, ms=9)
    ax.text(19.1, 14.9, "change\nthe machine", fontsize=6.2, color="#333333",
            ha="center", va="bottom", linespacing=1.3)

    # centre: the target rig, warmer grey, unmistakably another machine
    rig(ax, 27.2, 13.4, 11.6, 6.6,
        colors={"motor": "#C9C2BB", "house": "#F2EEE9"})
    chip(ax, 27.2, 6.0, "balanced accuracy 0.043", fc="#FBEAE2", tc="#A33F0B", fs=6.0)
    ax.text(27.2, 3.0, "test on rig B", fontsize=6.4, color=GREY, ha="center")

    arrow(ax, (33.5, 13.2), (37.2, 13.2), color="#333333", lw=1.0, ms=9)

    # right: why
    axs = fig.add_axes([41.6 / W, 5.4 / H, 25.2 / W, 12.8 / H])
    _, DIST = rig_distances()
    xs, ys_, pk = [], [], []
    for src, tgt in DIR_ORDER:
        if (src, tgt) not in DIST:
            continue
        v = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
             and r["norm"] == norm and r["k"] == 0]
        if not v:
            continue
        xs.append(DIST[(src, tgt)]); ys_.append(float(np.mean(v)))
        pk.append((src, tgt))
    xs, ys_ = np.asarray(xs), np.asarray(ys_)
    if len(xs) >= 3:
        p = np.polyfit(xs, ys_, 1)
        gx = np.linspace(xs.min() - 0.35, xs.max() + 0.35, 40)
        axs.plot(gx, np.polyval(p, gx), lw=0.8, color=GREY, ls=(0, (4, 2)), zorder=1)
    worst = int(np.argmin(ys_)) if len(ys_) else -1
    axs.scatter(xs, ys_, s=26, c=[ACC if i == worst else BLUE for i in range(len(xs))],
                edgecolor="white", linewidth=0.6, zorder=3)
    axs.axhline(1 / 3, ls=(0, (2, 2)), lw=0.6, color="#AAAAAA", zorder=0)
    axs.set_xlabel("discrepancy between the two rigs (a.u.)", fontsize=6.4, labelpad=2)
    axs.set_ylabel("transfer accuracy", fontsize=6.4)
    axs.set_ylim(0, 1.0)
    axs.set_xlim(xs.min() - 0.75, xs.max() + 0.85)
    axs.tick_params(labelsize=6.0, length=2)
    axs.annotate("$\\rho$ = $-$0.96", xy=(0.04, 0.10), xycoords="axes fraction",
                 fontsize=7.6, color="#333333")
    ax.text(54.2, 1.15, "the more two rigs differ, the more is lost",
            fontsize=6.4, color=GREY, ha="center")

    save(fig, "figGA")


# ---------------------------------------------------------------- Fig 2
def fig_signals():
    """The signal triplet: waveform with envelope and impulse spacing, envelope
    spectrum with theoretical orders against the measured peak, time-frequency map.

    The waveform panel shows only the first 40 ms of the window: across the full
    171 ms a single impulse spans 0.3% of the width and is invisible. The envelope
    spectrum and the time-frequency map still use the whole 171 ms window.
    """
    from scipy.signal import find_peaks

    from .run_experiment import get_windows
    Xc, yc, _, _ = get_windows("cwru")
    picks = [("Healthy", 0), ("Inner race", 1), ("Outer race", 2)]
    fs = config.FS_TARGET
    fr = 1797 / 60.0
    bpfo, bpfi = 3.585 * fr, 5.415 * fr
    TZ = 40.0                                    # displayed span of the waveform (ms)
    nz = int(TZ / 1000 * fs)

    sig = [Xc[yc == cls][0] for _, cls in picks]
    spec = []
    for x in sig:
        f2, t2, Sxx = spectrogram(x, fs=fs, nperseg=128, noverlap=96)
        m2 = f2 <= 3000
        spec.append((t2, f2[m2], 10 * np.log10(Sxx[m2] + 1e-12)))
    dbmax = max(s[2].max() for s in spec)        # one colour scale for all three rows, not three independent normalisations
    vmin, vmax = dbmax - 38, dbmax

    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE, 4.4))
    t = np.arange(config.WINDOW) / fs * 1000
    mesh = None
    for i, (name, cls) in enumerate(picks):
        x = sig[i]
        env = np.abs(hilbert(x - x.mean()))
        xn = x / (np.abs(x).max() + 1e-12)
        en = env / (env.max() + 1e-12)

        # ---- (a) waveform and envelope
        ax = axes[i, 0]
        ax.plot(t, xn, lw=0.30, color=C["black"], zorder=2)
        ax.plot(t, en, lw=0.55, color=C["blue"], zorder=3)
        ax.plot(t, -en, lw=0.55, color=C["blue"], zorder=3)
        ax.set_xlim(0, TZ)
        ax.set_ylim(-1.30, 1.55)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=6.8, rotation=0, ha="right", va="center",
                      labelpad=30)

        # impulse period from the first peak of the envelope autocorrelation between
        # 3 and 20 ms; a healthy bearing has no such peak
        e = en[:nz] - en[:nz].mean()
        ac = np.correlate(e, e, "full")[len(e) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lag_lo, lag_hi = int(0.003 * fs), int(0.020 * fs)
        apk, _ = find_peaks(ac[lag_lo:lag_hi], height=0.25)
        loc, _ = find_peaks(en[:nz], height=0.45)
        if apk.size:
            per = (apk[0] + lag_lo) / fs * 1000
            cand = loc[(t[loc] > 2.5) & (t[loc] < TZ - per - 1.0)]
            p0 = cand[np.argmax(en[cand])]
            near = loc[np.abs(t[loc] - (t[p0] + per)) < 2.2]
            p1 = near[np.argmin(np.abs(t[near] - (t[p0] + per)))] if near.size else None
            if p1 is not None:
                ax.annotate("", xy=(t[p1], 1.20), xytext=(t[p0], 1.20),
                            arrowprops=dict(arrowstyle="<|-|>", lw=0.55,
                                            color=C["vermillion"], mutation_scale=4.5,
                                            shrinkA=0, shrinkB=0), zorder=6)
                ax.text((t[p0] + t[p1]) / 2, 1.28, "%.1f ms" % (t[p1] - t[p0]),
                        fontsize=6.2, color=C["vermillion"], ha="center", va="bottom")
                for pp in (p0, p1):
                    ax.annotate("", xy=(t[pp], en[pp] + 0.04), xytext=(t[pp], 1.13),
                                arrowprops=dict(arrowstyle="-", lw=0.35,
                                                color="#B0B0B0", shrinkA=0, shrinkB=0))
        if not apk.size:
            ax.text(TZ * 0.5, 1.14, "no periodic impulses", fontsize=6.2,
                    color="#6E6E6E", ha="center", va="bottom", style="italic")
        if i == 0:
            ax.text(TZ * 0.985, 1.02, "envelope", fontsize=6.2, color=C["blue"],
                    ha="right", va="bottom")
        ax.set_xlabel("Time (ms)" if i == 2 else "", fontsize=7)
        if i != 2:
            ax.tick_params(labelbottom=False)

        # ---- (b) envelope spectrum
        ax = axes[i, 1]
        S = np.abs(np.fft.rfft(env - env.mean())) ** 2
        f = np.fft.rfftfreq(len(x), 1 / fs)
        m = f <= 400
        f, S = f[m], S[m] / S[m].max()
        ax.plot(f, S, lw=0.55, color=C["black"], zorder=3)
        ax.fill_between(f, 0, S, color=C["black"], alpha=0.10, lw=0, zorder=2)
        for fv, lab in ((bpfo, "BPFO"), (bpfi, "BPFI")):
            ax.axvline(fv, ls=(0, (2.5, 1.6)), lw=0.6, color="#8A8A8A", zorder=1)
            if i == 0:                           # label once, on the first row only, so that the peak itself is not covered
                ax.text(fv, 1.215, lab, fontsize=6.0, color="#5E5E5E", ha="center",
                        va="top", zorder=6,
                        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 1.26)
        ax.set_yticks([])
        ax.set_xlabel("Frequency (Hz)" if i == 2 else "", fontsize=7)
        if i != 2:
            ax.tick_params(labelbottom=False)

        # attribute the main peak: within 25% of a theoretical order it is that order,
            # otherwise it is counted as a shaft harmonic
        j = int(np.argmax(S))
        d_o, d_i = abs(f[j] - bpfo) / bpfo, abs(f[j] - bpfi) / bpfi
        if d_o < 0.25:
            lab = "BPFO %.0f Hz" % f[j]
        elif d_i < 0.25:
            lab = "BPFI %.0f Hz" % f[j]
        else:
            lab = "shaft harmonic"
        ha = "left" if f[j] < 80 else ("right" if f[j] > 330 else "center")
        ax.plot(f[j], S[j], "o", ms=2.2, mfc=C["vermillion"], mec="white",
                mew=0.4, zorder=7)
        if lab == "shaft harmonic":              # put shaft harmonics down and right of the peak, leaving the top free for labels
            txt_xy, ha = (f[j] + 48, S[j] - 0.26), "left"
        else:
            txt_xy = (f[j], S[j] + 0.115)
        ax.annotate(lab, xy=(f[j], S[j] + 0.02), xytext=txt_xy,
                    fontsize=6.2, color=C["vermillion"], ha=ha, va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.45, color="#B0B0B0",
                                    shrinkA=1.5, shrinkB=2.5), zorder=8)
        # second harmonic
        if d_o < 0.25:
            k = np.where((f > 1.85 * bpfo) & (f < 2.15 * bpfo))[0]
            if k.size and S[k].max() > 0.25:
                kk = k[int(np.argmax(S[k]))]
                ax.plot(f[kk], S[kk], "o", ms=2.0, mfc="white", mec=C["vermillion"],
                        mew=0.7, zorder=7)
                ax.annotate("2$\\times$BPFO", xy=(f[kk], S[kk] + 0.02),
                            xytext=(f[kk], S[kk] + 0.115), fontsize=6.0,
                            color=C["vermillion"], ha="center", va="bottom",
                            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85),
                            arrowprops=dict(arrowstyle="-", lw=0.45,
                                            color="#B0B0B0", shrinkA=1.5, shrinkB=2.5))

        # ---- (c) time-frequency map
        ax = axes[i, 2]
        t2, fy, zz = spec[i]
        mesh = ax.pcolormesh(t2 * 1000, fy / 1000, zz, cmap=CM_ALT, shading="auto",
                             rasterized=True, vmin=vmin, vmax=vmax)
        ax.set_yticks([0, 1, 2, 3])
        if i == 2:
            ax.set_xlabel("Time (ms)", fontsize=7)
            ax.set_ylabel("kHz", fontsize=6.2)
        else:
            ax.tick_params(labelbottom=False, labelleft=False)

    for j, title in enumerate(("Waveform", "Envelope spectrum", "Spectrogram")):
        axes[0, j].set_title(title, fontsize=7.4, pad=5)
    for j in range(3):
        panel(axes[0, j], "abc"[j], dx=-0.015, dy=1.02)

    cax = fig.add_axes([0.928, 0.155, 0.0095, 0.60])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Power (dB)", fontsize=6.6, labelpad=2)
    cb.ax.tick_params(labelsize=6.0, length=1.8, width=0.5)
    cb.outline.set_linewidth(0.5)

    fig.subplots_adjust(hspace=0.13, wspace=0.15, left=0.105, right=0.905)
    save(fig, "fig2_signals")


# ---------------------------------------------------------------- Fig 3
def fig_forest(rows, norm="global"):
    """Paired forest plot with an effect-size column on the right: random versus
    run-wise split accuracy for the same (corpus, model)."""
    ds = datasets_present(rows)
    models = ("xgb", "cnn", "resnet")
    nrows = len(ds) * (len(models) + 1)
    fig = plt.figure(figsize=(DOUBLE, 0.155 * nrows + 1.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.65, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axe = fig.add_subplot(gs[0, 1], sharey=ax)

    ypos, labels, bands, yy = [], [], [], 0
    for gi, d in enumerate(ds):
        start = yy
        for m in models:
            sub = [r for r in rows if r.get("ds") == d and r["model"] == m
                   and r["norm"] == norm and r["snr"] is None]
            a_mu, a_sd = _ms([r["acc"] for r in sub if r["proto"] == "random"])
            b_mu, b_sd = _ms([r["acc"] for r in sub if r["proto"] == "run"])
            ypos.append((yy, a_mu, a_sd, b_mu, b_sd, d, m))
            labels.append(MODEL_LABEL[m])
            yy += 1
        bands.append((d, start - 0.5, yy - 0.5))
        yy += 0.9

    dy = 0.185
    for yy_, a_mu, a_sd, b_mu, b_sd, d, m in ypos:
        ax.plot([b_mu, a_mu], [yy_ + dy, yy_ - dy], lw=0.8, color=C["grey"], zorder=1)
        ax.errorbar([a_mu], [yy_ - dy], xerr=[a_sd], fmt="o", ms=3.8, mfc="white",
                    mew=0.9, color=C["vermillion"], ecolor=C["vermillion"],
                    elinewidth=0.6, capsize=1.4, zorder=4)
        ax.errorbar([b_mu], [yy_ + dy], xerr=[b_sd], fmt="o", ms=3.8,
                    color=C["blue"], ecolor=C["blue"], elinewidth=0.6,
                    capsize=1.4, zorder=4)
        dlt = (a_mu - b_mu) * 100
        if abs(dlt) < 1:
            # a null effect is drawn as a tick at zero, so that it does not read as missing data
            axe.plot([0], [yy_], marker="|", ms=5.5, mew=0.9,
                     color=C["black"], zorder=3)
        else:
            axe.barh(yy_, dlt, height=0.44, color=DS_C[d], alpha=0.75,
                     edgecolor="none")
        axe.text(dlt + 1.0 if dlt > 2 else 1.0, yy_,
                 f"{dlt:+.0f}" if abs(dlt) >= 1 else "0",
                 va="center", ha="left", fontsize=6.6, color=C["black"])

    for d, lo, hi in bands:
        ax.axhspan(lo, hi, color=DS_C[d], alpha=0.055, lw=0, zorder=0)
        axe.axhspan(lo, hi, color=DS_C[d], alpha=0.055, lw=0, zorder=0)
        ax.text(0.292, lo + 0.12, DS_LABEL[d], fontsize=6.8, color=DS_C[d],
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.4, alpha=0.92))

    ax.axvline(1 / 3, ls=(0, (1, 1.8)), lw=0.6, color=C["grey"], zorder=0)
    ax.annotate("chance", xy=(1 / 3, 0.005), xycoords=("data", "axes fraction"),
                xytext=(2.5, 0), textcoords="offset points", fontsize=6.4,
                color=C["grey"], va="bottom")
    ax.set_yticks([y[0] for y in ypos], labels, fontsize=6.6)
    ax.set_ylim(-1.1, yy - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0.28, 1.03)
    ax.set_xlabel("Accuracy (mean $\\pm$ s.d. over runs)", fontsize=7.2)
    ax.tick_params(axis="y", length=0)

    max_d = max((a - b) * 100 for _, a, _, b, _, _, _ in ypos)
    axe.set_xlim(-1.5, max_d * 1.45)
    axe.axvline(0, lw=0.5, color=C["black"])
    axe.set_xlabel("$\\Delta$ (pp)", fontsize=7.2)
    axe.tick_params(axis="y", left=False, labelleft=False)
    axis_off_right(axe)

    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=C["vermillion"],
                               color=C["vermillion"], ms=4, label="Random window split"),
                        Line2D([], [], marker="o", ls="", color=C["blue"], ms=4,
                               label="Split by run")],
               loc="lower center", bbox_to_anchor=(0.42, -0.055), ncol=2, fontsize=6.4,
               handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(left=0.135, bottom=0.16, top=0.985)
    save(fig, "fig3_protocol_gap")


def axis_off_right(ax):
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ---------------------------------------------------------------- Fig 4
def fig_matrix(rows, model="resnet", norm="global"):
    """Left: the absolute level at k=0. Right: the gain from tuning, annotated with
    the level it reaches."""
    ds = datasets_present(rows)
    n = len(ds)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.55))
    norm0 = Normalize(0.0, 1.0)
    cmap0 = CM_SEQ

    ax = axes[0]
    M0 = np.array([[get_cell(rows, s, t, model, norm, 0) for t in ds] for s in ds])
    ax.imshow(M0, cmap=cmap0, vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if np.isnan(M0[i, j]):
                continue
            ax.text(j, i, f"{M0[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                    color=text_on(cmap0, M0[i, j], 0.0, 1.0),
                    fontweight="bold" if i == j else "normal")

    ax = axes[1]
    D = np.zeros((n, n))
    for i, s in enumerate(ds):
        for j, t in enumerate(ds):
            a, b = get_cell(rows, s, t, model, norm, 0), get_cell(rows, s, t, model, norm, 20)
            D[i, j] = (b - a) if (a == a and b == b) else np.nan
    Dm = np.ma.masked_invalid(D)
    Dm = np.ma.masked_where(np.eye(n, dtype=bool), Dm)   # the diagonal is saturated; a gain there means nothing
    cmap_d = CM_SEQ2.copy()
    cmap_d.set_bad("0.88")
    im = ax.imshow(Dm, cmap=cmap_d, vmin=0.0, vmax=max(0.45, float(np.nanmax(D))))
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "saturated\nat $k$=0", ha="center", va="center",
                        fontsize=6.2, color="0.42", style="italic")
                continue
            if np.isnan(D[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="0.42")
                continue
            b = get_cell(rows, ds[i], ds[j], model, norm, 20)
            ax.text(j, i - 0.13, f"+{D[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if D[i, j] > 0.28 else C["black"])
            ax.text(j, i + 0.20, f"$\\rightarrow${b:.2f}", ha="center", va="center",
                    fontsize=6.4, color="white" if D[i, j] > 0.28 else C["grey"])

    for kk, ax in enumerate(axes):
        ax.set_xticks(range(n), [DS_LABEL[d] for d in ds], fontsize=7)
        ax.set_yticks(range(n), [DS_LABEL[d] for d in ds], fontsize=7)
        ax.set_xlabel("test domain", fontsize=7)
        if kk == 0:
            ax.set_ylabel("train domain", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel(ax, "ab"[kk], dx=-0.16, dy=1.02)

    axes[0].set_title("Frozen source model ($k$=0)", fontsize=7.4, pad=6)
    axes[1].set_title("Gain from $k$=20 adaptation", fontsize=7.4, pad=6)

    cb0 = fig.colorbar(plt.cm.ScalarMappable(norm=norm0, cmap=cmap0),
                       ax=axes[0], shrink=0.72, pad=0.03)
    cb0.set_label("Balanced accuracy", fontsize=6.4, labelpad=9)
    cb1 = fig.colorbar(im, ax=axes[1], shrink=0.72, pad=0.03)
    cb1.set_label("$\\Delta$ balanced accuracy", fontsize=6.4, labelpad=9)
    for cb in (cb0, cb1):
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig4_matrix")


# ---------------------------------------------------------------- Fig 5
def fig_recovery(rows, norm="global"):
    """Six small multiples: recovery from k = 0 to 20, with ±1 s.d. bands."""
    dirs = [(s, t) for s, t in DIR_ORDER
            if any(r.get("src") == s and r.get("tgt") == t for r in rows)]
    if not dirs:
        return
    ncol = 3
    nrow = int(np.ceil(len(dirs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.42 * nrow + 0.55),
                             sharex=True, sharey=True, squeeze=False)
    ks = [0, 5, 10, 20]

    # the target ceiling: mean accuracy of a run-wise split within that corpus
    def ceiling(d, m):
        v = [r["acc"] for r in rows if r.get("ds") == d and r.get("proto") == "run"
             and r["model"] == m and r["norm"] == norm]
        return float(np.mean(v)) if v else np.nan

    for idx, (src, tgt) in enumerate(dirs):
        ax = axes[idx // ncol][idx % ncol]
        for m in ("xgb", "cnn", "resnet"):
            mu, sd = [], []
            for k in ks:
                sub = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
                       and r["model"] == m and r["norm"] == norm and r["k"] == k]
                a, b = _ms(sub)
                mu.append(a); sd.append(b)
            mu, sd = np.array(mu), np.array(sd)
            ax.fill_between(ks, mu - sd, mu + sd, color=MODEL_C[m], alpha=0.16, lw=0)
            ax.plot(ks, mu, color=MODEL_C[m], lw=1.0,
                    marker="o" if m == "cnn" else ("D" if m == "resnet" else "s"),
                    ms=2.6, mfc="white" if m == "xgb" else MODEL_C[m], mew=0.7)
            c = ceiling(tgt, m)
            if c == c:
                ax.axhline(c, ls=(0, (1, 1.6)), lw=0.6, color=MODEL_C[m], alpha=0.55)
        ax.axhline(1 / 3, ls=(0, (3, 2)), lw=0.7, color=C["black"], alpha=0.45)
        ax.set_xlim(-1.2, 21.2)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(ks, ["0", "5", "10", "20"])
        ax.set_title(f"({ 'abcdef'[idx] })  {DS_LABEL[src]} $\\rightarrow$ {DS_LABEL[tgt]}",
                     fontsize=7.2, pad=3, color=DS_C[tgt], loc="left")
        if idx % ncol == 0:
            ax.set_ylabel("Balanced accuracy", fontsize=7)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("Labelled target windows per class, $k$", fontsize=6.8)

    for idx in range(len(dirs), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    handles = [Line2D([], [], color=MODEL_C[m], lw=1.1, marker="o", ms=2.8,
                      label=MODEL_LABEL[m]) for m in ("xgb", "cnn", "resnet")]
    handles += [Line2D([], [], color=C["black"], lw=0.8, ls=(0, (3, 2)), alpha=0.5,
                       label="chance"),
                Line2D([], [], color=C["grey"], lw=0.8, ls=(0, (1, 1.6)),
                       label="target-domain ceiling")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=6.0, handletextpad=0.4, columnspacing=1.1)
    fig.subplots_adjust(hspace=0.30, wspace=0.18, bottom=0.14, top=0.94)
    save(fig, "figS2_recovery")


# ---------------------------------------------------------------- Fig 6
def fig_mechanism(rows, norm="global"):
    """Why transfer fails: class-conditional shift, impulsiveness, distance vs accuracy."""
    from .run_experiment import get_windows

    ds_list = [d for d in ("cwru", "mfpt", "pu")
               if os.path.exists(os.path.join(config.DATA_DIR, f"_raw_{d}.npz"))]
    rng = np.random.default_rng(0)
    prof = {}
    for name in ds_list:
        X, y, _, _ = get_windows(name)
        idx = rng.choice(len(X), min(1600, len(X)), replace=False)
        Xs, ys = X[idx], y[idx]
        Xc = Xs - Xs.mean(axis=1, keepdims=True)
        rms = np.sqrt((Xc ** 2).mean(axis=1))
        kurt = ((Xc ** 4).mean(axis=1) / (Xc ** 2).mean(axis=1) ** 2) - 3.0
        prof[name] = (rms, kurt, ys)

    fig = plt.figure(figsize=(DOUBLE, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.92, 0.92, 0.92, 1.35, 1.45],
                          wspace=0.30)

    # (a) class-conditional RMS ridges, one group per corpus
    ax0 = None
    for gi, name in enumerate(ds_list):
        ax = fig.add_subplot(gs[0, gi], sharex=ax0, sharey=ax0)
        ax0 = ax if ax0 is None else ax0
        rms, _, ys = prof[name]
        lr = np.log10(rms)
        lo, hi = -2.0, 0.6
        grid = np.linspace(lo, hi, 220)
        for ci in (2, 1, 0):          # stack bottom-up, so that later classes sit on top
            m = ys == ci
            if m.sum() < 5:
                continue
            v = lr[m]
            v = v[(v > lo) & (v < hi)]
            if len(v) < 5:
                continue
            d = gaussian_kde(v, bw_method=bw_for(v))(grid)
            d = d / d.max() * 0.72
            base = 2 - ci
            ax.fill_between(grid, base, base + d, color=CLASS_C[ci], alpha=0.92,
                            lw=0.5, edgecolor="white", zorder=3 + ci)
            ax.plot(grid, np.full_like(grid, base), lw=0.4, color=C["black"], alpha=0.4,
                    zorder=2)
            rug = v if len(v) <= 140 else rng.choice(v, 140, replace=False)
            ax.plot(rug, np.full_like(rug, base - 0.05), "|", ms=1.8, mew=0.45,
                    color=CLASS_C[ci], alpha=0.55, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.25, 3.0)
        ax.set_yticks([2, 1, 0], ["H", "IR", "OR"], fontsize=6.6)
        ax.set_xticks([-2, -1, 0], ["0.01", "0.1", "1"])
        ax.tick_params(axis="y", length=0)
        ax.set_title(DS_LABEL[name], fontsize=7.2, pad=3, color=DS_C[name],
                     fontweight="bold")
        if gi == 0:
            ax.set_ylabel("density", fontsize=6.2)
        ax.set_xlabel("window RMS", fontsize=6.4)
        if gi == 0:
            panel(ax, "a", dx=-0.34, dy=1.04)

    # (b) kurtosis ECDF for the inner-race class
    ax = fig.add_subplot(gs[0, 3])
    for name in ds_list:
        _, kurt, ys = prof[name]
        v = np.sort(kurt[ys == 1])
        v = v[(v > 0.3)]
        if len(v) < 10:
            continue
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=1.0,
                color=DS_C[name], label=DS_LABEL[name])
        med = np.median(v)
        ax.plot([med], [0.5], marker="o", ms=3.0, color=DS_C[name], mec="white", mew=0.5,
                zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 400)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Inner-race excess kurtosis", fontsize=6.8)
    ax.set_ylabel("Cumulative fraction", fontsize=6.8)
    ax.axhline(0.5, ls=(0, (1, 2)), lw=0.5, color=C["grey"], zorder=0)
    ax.legend(fontsize=6.6, loc="lower right", handlelength=1.4, labelspacing=0.25)
    panel(ax, "b", dx=-0.26, dy=1.04)

    # (c) descriptor-space distance vs transfer accuracy
    ax = fig.add_subplot(gs[0, 4])
    _, DIST = rig_distances()
    xs, ys_, es, labs, cols, pair_keys = [], [], [], [], [], []
    for src, tgt in DIR_ORDER:
        if (src, tgt) not in DIST:
            continue
        dist = DIST[(src, tgt)]
        a, b = model_family_means(rows, src=src, tgt=tgt, norm=norm, k=0)
        if a != a:
            continue
        xs.append(dist); ys_.append(a); es.append(b)
        labs.append(f"{DS_LABEL[src]}$\\to${DS_LABEL[tgt]}")
        pair_keys.append((src, tgt))
        cols.append(DS_C[tgt])
    xs, ys_, es, pk = np.asarray(xs), np.asarray(ys_), np.asarray(es), pair_keys
    if len(xs) >= 3:
        p = np.polyfit(xs, ys_, 1)
        gx = np.linspace(xs.min() * 0.94, xs.max() * 1.04, 50)
        ax.plot(gx, np.polyval(p, gx), lw=0.8, color=C["grey"], ls=(0, (4, 2)), zorder=1)
    # manual offset: two points share each distance, so nudge them apart
    OFF = {("cwru", "mfpt"): (0.15, -0.050, "left"),
           ("mfpt", "cwru"): (0.15, 0.050, "left"),
           ("cwru", "pu"): (-0.15, 0.032, "right"),
           ("pu", "cwru"): (-0.15, -0.032, "right"),
           ("mfpt", "pu"): (0.15, -0.060, "left"),
           ("pu", "mfpt"): (0.15, 0.060, "left")}
    code = {"cwru": "C", "mfpt": "M", "pu": "P"}
    for x_, y_, e_, c_ in zip(xs, ys_, es, cols):
        ax.errorbar([x_], [y_], yerr=[e_], fmt="o", ms=4.4, color=c_, mec="white",
                    mew=0.6, ecolor=c_, elinewidth=0.6, capsize=1.8, zorder=3)
    for x_, y_, (s_, t_) in zip(xs, ys_, pk):
        dx_, dy_, ha_ = OFF[(s_, t_)]
        ax.annotate(code[s_] + "$\\to$" + code[t_], xy=(x_, y_),
                    xytext=(x_ + dx_, y_ + dy_), fontsize=6.2, color="#3A3A3A",
                    ha=ha_, va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#B0B0B0",
                                    shrinkA=1.0, shrinkB=2.5))
    ax.axhline(1 / 3, ls=(0, (3, 2)), lw=0.7, color=C["black"], alpha=0.45)
    ax.set_xlabel("Rig distance (a.u.)", fontsize=6.8)
    ax.set_ylabel("Transfer balanced accuracy", fontsize=6.8)
    ax.set_ylim(0, 1.0)
    if len(xs):
        ax.set_xlim(xs.min() - 1.22, xs.max() + 0.92)
        ax.text(xs.min() - 1.17, 1 / 3 - 0.030, "chance", fontsize=6.2, color=C["grey"],
                ha="left", va="top")
        ax.text(0.015, 0.965, "C CWRU   M MFPT   P PU", transform=ax.transAxes,
                fontsize=6.0, color="#5E5E5E", va="top",
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    panel(ax, "c", dx=-0.30, dy=1.04)

    save(fig, "figS1_mechanism")
    # print the numbers the prose quotes
    from scipy.stats import pearsonr, spearmanr
    for i, l_ in enumerate(labs):
        print(f"  {l_:<12} d={xs[i]:.2f}  bacc={ys_[i]:.3f}±{es[i]:.3f}")
    if len(xs) >= 4:
        rho, prho = spearmanr(xs, ys_)
        r, pr = pearsonr(xs, ys_)
        print(f"  distance vs bacc: Spearman rho={rho:.3f} (p={prho:.3f}), "
              f"Pearson r={r:.3f} (p={pr:.3f}), n={len(xs)}")


# ---------------------------------------------------------------- Fig 7
def fig_embedding(rows):
    """(a) t-SNE of the feature space; (b), (c) confusion matrices before and after transfer."""
    from sklearn.manifold import TSNE
    from .datasets import Normalizer
    from .features import extract_features
    from .run_experiment import get_windows

    ds_list = [d for d in ("cwru", "mfpt", "pu")
               if os.path.exists(os.path.join(config.DATA_DIR, f"_raw_{d}.npz"))]
    Z, Y, D = [], [], []
    rng = np.random.default_rng(0)
    for name in ds_list:
        X, y, _, _ = get_windows(name)
        idx = rng.choice(len(X), min(700, len(X)), replace=False)
        Z.append(extract_features(X[idx])); Y.append(y[idx])
        D.append(np.full(len(idx), ds_list.index(name)))
    Z, Y, D = np.vstack(Z), np.concatenate(Y), np.concatenate(D)
    Z = Normalizer("global").fit(Z).transform(Z)
    print("t-SNE on", Z.shape)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
               learning_rate="auto").fit_transform(Z)

    fig = plt.figure(figsize=(DOUBLE, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    for di, name in enumerate(ds_list):
        for ci, mk in enumerate(CLS_MARK):
            m = (D == di) & (Y == ci)
            ax.scatter(emb[m, 0], emb[m, 1], s=3.2, marker=mk, color=DS_C[name],
                       alpha=0.7 if ci == 0 else 0.42, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=6.6); ax.set_ylabel("t-SNE 2", fontsize=6.6)
    ax.set_title("Feature space", fontsize=7.4, pad=4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=DS_C[d], ms=4,
                              label=DS_LABEL[d]) for d in ds_list],
              fontsize=6.4, loc="best", handletextpad=0.2, borderpad=0.2)
    panel(ax, "a", dx=-0.05, dy=1.0)

    def pick(src, tgt, k):
        sub = [r for r in rows if r.get("src") == src and r.get("tgt") == tgt
               and r["model"] == "resnet" and r["norm"] == "global" and r["k"] == k
               and r["seed"] == 42]
        return sub[0]["cm"] if sub else None

    im = None
    for gi, (src, tgt, k, title, letter) in enumerate(
            [("cwru", "mfpt", 0, "CWRU$\\to$MFPT, $k$=0", "b"),
             ("cwru", "mfpt", 20, "CWRU$\\to$MFPT, $k$=20", "c")]):
        axx = fig.add_subplot(gs[0, gi + 1])
        cm = pick(src, tgt, k)
        if cm is None:
            continue
        cmn = np.array(cm, dtype=float)
        cmn = cmn / cmn.sum(axis=1, keepdims=True)
        im = axx.imshow(cmn, cmap=CM_SEQ, vmin=0, vmax=1)   # the same colour scale in both panels, so that they can be read against each other
        axx.set_xticks(range(3), ["H", "IR", "OR"], fontsize=6)
        axx.set_yticks(range(3), ["H", "IR", "OR"], fontsize=6)
        axx.set_xlabel("Predicted", fontsize=6.4)
        if gi == 0:
            axx.set_ylabel("True", fontsize=6.4)
        for i in range(3):
            for j in range(3):
                axx.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                         color=text_on(CM_SEQ, cmn[i, j], 0, 1))
        for s in axx.spines.values():
            s.set_visible(False)
        axx.tick_params(length=0)
        axx.set_title(title, fontsize=6.5, pad=4)
        panel(axx, letter, dx=-0.26, dy=1.0)
    if im is not None:
        cb = fig.colorbar(im, ax=fig.axes[1:], shrink=0.72, pad=0.025)
        cb.set_label("P(predicted $\\mid$ true)", fontsize=6.2, labelpad=2)
        cb.ax.tick_params(labelsize=6.0, length=1.8, width=0.5)
        cb.outline.set_linewidth(0.4)
    save(fig, "fig5_embedding")


# ---------------------------------------------------------------- Fig S1
def fig_snr(rows, norm="global"):
    rowsn = [r for r in rows if r["model"] == "cnn" and r["norm"] == norm]
    snrs = [None, 8, 4, 0, -4]
    conds = []
    for d in datasets_present(rows):
        conds.append((f"within {DS_LABEL[d]}", ("ds", d)))
    for src, tgt in DIR_ORDER:
        if any(r.get("src") == src and r.get("tgt") == tgt for r in rows):
            conds.append((f"{DS_LABEL[src]}$\\rightarrow${DS_LABEL[tgt]}",
                          ("cross", (src, tgt))))
    M = np.full((len(conds), len(snrs)), np.nan)
    for i, (_, key) in enumerate(conds):
        for j, snr in enumerate(snrs):
            if key[0] == "ds":
                sub = [r["bacc"] for r in rowsn if r.get("ds") == key[1]
                       and r["proto"] == "run" and r["snr"] == snr]
            else:
                sub = [r["bacc"] for r in rowsn if r.get("src") == key[1][0]
                       and r.get("tgt") == key[1][1] and r["k"] == 0 and r["snr"] == snr]
            if sub:
                M[i, j] = np.mean(sub)
    fig, ax = plt.subplots(figsize=(ONE_HALF + 1.0, 2.2))
    im = ax.imshow(M, cmap=CM_SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(snrs)), ["clean"] + [f"{s:+d}" for s in snrs[1:]], fontsize=6.4)
    ax.set_yticks(range(len(conds)), [c[0] for c in conds], fontsize=6.2)
    ax.set_xlabel("Test SNR (dB)", fontsize=6.8)
    for i in range(len(conds)):
        for j in range(len(snrs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                        color=text_on(CM_SEQ, M[i, j], 0, 1))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Balanced accuracy", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.6)
    cb.outline.set_linewidth(0.4)
    save(fig, "figS3_snr")


if __name__ == "__main__":
    rows = load_results()
    print(f"{len(rows)} result files; datasets: {datasets_present(rows)}")
    # figs 2, 6 and 7 need the raw waveforms; skip them when absent, rebuild the rest
    has_raw = all(os.path.exists(os.path.join(config.DATA_DIR, f"_raw_{d}.npz"))
                  for d in ("cwru", "mfpt", "pu"))
    jobs = [("fig1", lambda: fig_protocol(), False),
            ("figGA", lambda: fig_graphical(rows), False),
            ("fig2", lambda: fig_signals(), True),
            ("fig3", lambda: fig_forest(rows), False),
            ("fig4", lambda: fig_matrix(rows), False),
            ("fig5", lambda: fig_recovery(rows), False),
            ("fig6", lambda: fig_mechanism(rows), True),
            ("fig7", lambda: fig_embedding(rows), True),
            ("figS1", lambda: fig_snr(rows), False)]
    skipped = []
    for name, fn, needs_data in jobs:
        if needs_data and not has_raw:
            print(f"  {name}: skipped, drawn from the waveforms", flush=True)
            skipped.append(name)
            continue
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  {name}: skipped ({exc})", flush=True)
            skipped.append(name)
    if skipped:
        print(f"skipped {', '.join(skipped)}; the committed PDF and PNG in "
              f"figures/ remain the reference versions")
