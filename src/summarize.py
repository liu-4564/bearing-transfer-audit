"""Aggregate results/*.json into the tables used in the paper (markdown)."""
import os
from collections import defaultdict

import numpy as np

from . import config
from .make_figures import (MODEL_LABEL, load_results, model_family_means,
                           rig_distances)


def fmt(vals):
    if not vals:
        return "—"
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.3f}" if s < 1e-6 else f"{m:.3f} ± {s:.3f}"


def main():
    rows = load_results()
    lines = []

    # Table 1: within-corpus performance
    lines.append("## Table 1  Within-dataset accuracy (mean over seeds where available)\n")
    lines.append("| Dataset | Split | Norm | " + " | ".join(MODEL_LABEL[m] for m in
                 ["xgb", "cnn", "resnet"]) + " |")
    lines.append("|---|---|---|" + "---|" * 3)
    for ds in ("cwru", "mfpt", "pu"):
        for proto in ("random", "run"):
            for norm in ("window", "global"):
                cells = []
                for m in ("xgb", "cnn", "resnet"):
                    sub = [r["acc"] for r in rows if r.get("ds") == ds and
                           r.get("proto") == proto and r["model"] == m and
                           r["norm"] == norm and r["k"] == 0 and r["snr"] is None]
                    cells.append(fmt(sub))
                lines.append(f"| {ds.upper()} | {proto} | {norm} | " + " | ".join(cells) + " |")

    # Table 2: cross-corpus, with few-shot adaptation
    lines.append("\n## Table 2  Cross-dataset balanced accuracy vs. target samples per class k\n")
    lines.append("| Direction | Norm | Model | k=0 | k=5 | k=10 | k=20 |")
    lines.append("|---|---|---|---|---|---|---|")
    DS = ("cwru", "mfpt", "pu")
    DIRS = [(a, b) for a in DS for b in DS if a != b]
    for src, tgt in DIRS:
        for norm in ("window", "global"):
            for m in ("xgb", "cnn", "resnet"):
                cells = []
                for k in (0, 5, 10, 20):
                    sub = [r["bacc"] for r in rows if r.get("src") == src and
                           r.get("tgt") == tgt and r["model"] == m and
                           r["norm"] == norm and r["k"] == k and r["snr"] is None]
                    cells.append(fmt(sub))
                lines.append(f"| {src.upper()} → {tgt.upper()} | {norm} | "
                             f"{MODEL_LABEL[m]} | " + " | ".join(cells) + " |")

    # Table 3: XGBoost across corpora (no fine-tuning; the classical reference point)
    lines.append("\n## Table 3  Feature-based baseline (XGBoost) cross-dataset accuracy\n")
    lines.append("| Direction | Norm | k=0 (no adaptation) | k=20 (refit with target samples) |")
    lines.append("|---|---|---|---|")
    DS = ("cwru", "mfpt", "pu")
    DIRS = [(a, b) for a in DS for b in DS if a != b]
    for src, tgt in DIRS:
        for norm in ("window", "global"):
            a = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
                 and r["model"] == "xgb" and r["norm"] == norm and r["k"] == 0
                 and r["snr"] is None]
            b = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
                 and r["model"] == "xgb" and r["norm"] == norm and r["k"] == 20]
            lines.append(f"| {src.upper()} → {tgt.upper()} | {norm} | {fmt(a)} | {fmt(b)} |")

    # Table 4: descriptor-space distance vs transfer accuracy
    lines.append("\n## Table 4  Rig distance vs. transfer accuracy (no adaptation, global norm)\n")
    lines.append("| Direction | Descriptor distance | Models combined | 1D-ResNet |")
    lines.append("|---|---|---|---|")
    _, DIST = rig_distances()
    for src, tgt in DIRS:
        mu, sd = model_family_means(rows, src=src, tgt=tgt, norm="global", k=0)
        v_rn = [r["bacc"] for r in rows if r.get("src") == src and r.get("tgt") == tgt
                and r["norm"] == "global" and r["k"] == 0 and r["model"] == "resnet"
                and r.get("snr") is None]
        if mu != mu:
            continue
        lines.append(f"| {src.upper()} → {tgt.upper()} | {DIST.get((src, tgt), float('nan')):.2f} "
                     f"| {mu:.3f} ± {sd:.3f} | {fmt(v_rn)} |")

    from scipy.stats import pearsonr, spearmanr
    dx, dy = [], []
    for src, tgt in DIRS:
        mu, _ = model_family_means(rows, src=src, tgt=tgt, norm="global", k=0)
        if mu == mu and (src, tgt) in DIST:
            dx.append(DIST[(src, tgt)]); dy.append(mu)
    if len(dx) >= 4:
        rho, prho = spearmanr(dx, dy)
        r, pr = pearsonr(dx, dy)
        lines.append(f"\nSpearman rho = {rho:.3f} (p = {prho:.3f}); "
                     f"Pearson r = {r:.3f} (p = {pr:.3f}); n = {len(dx)} directions.\n")

    text = "\n".join(lines)
    out = os.path.join(config.RESULTS_DIR, "tables.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
