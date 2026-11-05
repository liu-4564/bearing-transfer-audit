"""Shared configuration: class labels, sampling rates, windowing parameters, paths."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

CLASSES = ["healthy", "inner", "outer"]  # MFPT has no ball class, so the benchmark uses three classes throughout
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

FS_TARGET = 12000      # every corpus is resampled to 12 kHz
WINDOW = 2048          # ~0.17 s
OVERLAP = 0.5

SEED = 42

DEVICE = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
except Exception:
    pass

# CWRU file id -> (class, load in hp, defect size)
CWRU_MAP = {}
for i, fid in enumerate([97, 98, 99, 100]):
    CWRU_MAP[fid] = ("healthy", i, 0)
for base, size in [(105, 7), (169, 14), (209, 21)]:      # inner race
    for j, fid in enumerate(range(base, base + 4)):
        CWRU_MAP[fid] = ("inner", j, size)
for base, size in [(130, 7), (197, 14), (234, 21)]:      # outer race @6:00
    for j, fid in enumerate(range(base, base + 4)):
        CWRU_MAP[fid] = ("outer", j, size)
