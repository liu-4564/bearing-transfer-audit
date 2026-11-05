"""Training and evaluation primitives: fit, predict, metrics, few-shot fine-tuning. GPU-aware."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

from . import config
from .features import extract_features
from .models import CNN1D, ResNet1D

DEV = config.DEVICE

# cuDNN picks convolution algorithms heuristically, so two runs of the same seed
# can differ by ~0.02 balanced accuracy. That is far below the between-seed spread
# reported in the paper, so the default is kept for speed; the archived results
# were produced in this mode. Set BITEXACT_TORCH=1 for bit-level reproducibility:
# slower, but the same seed then returns exactly the same numbers.
if os.environ.get("BITEXACT_TORCH") == "1":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
else:
    torch.backends.cudnn.benchmark = True


def _seed():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def _t(X, y=None):
    xb = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEV).unsqueeze(1)
    if y is None:
        return xb
    return xb, torch.tensor(np.asarray(y), dtype=torch.long, device=DEV)


def build_model(model_name, n_classes):
    return (CNN1D(n_classes) if model_name == "cnn" else ResNet1D(n_classes)).to(DEV)


def fit_torch(model_name, Xtr, ytr, Xva, yva, epochs=40, bs=128, lr=1e-3):
    _seed()
    n_classes = len(set(np.asarray(ytr).tolist()) | set(np.asarray(yva).tolist()))
    model = build_model(model_name, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    lossf = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*_t(Xtr, ytr)), batch_size=bs, shuffle=True)
    Xva_t, yva_t = _t(Xva, yva)
    best_state, best_va, bad = None, -1.0, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1).cpu().numpy()
        va = balanced_accuracy_score(yva_t.cpu().numpy(), pred)
        if va > best_va:
            best_va, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def finetune_torch(model, Xa, ya, epochs=15, bs=32, lr=3e-4):
    """Few-shot fine-tuning: keep training on k labelled samples per class in the
target domain, mimicking an on-site calibration."""
    _seed()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(*_t(Xa, ya))
    loader = torch.utils.data.DataLoader(ds, batch_size=min(bs, len(ya)), shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_torch(model, X, bs=512):
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(_t(X[i:i + bs])).argmax(1).cpu().numpy())
    return np.concatenate(outs)


def fit_xgb(Xtr, ytr):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                        random_state=config.SEED, n_jobs=8)
    clf.fit(extract_features(Xtr), ytr)
    return clf


def predict_xgb(clf, X):
    return clf.predict(extract_features(X))


def metrics(y_true, y_pred):
    return dict(acc=float(accuracy_score(y_true, y_pred)),
                bacc=float(balanced_accuracy_score(y_true, y_pred)),
                f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                cm=confusion_matrix(y_true, y_pred,
                                    labels=list(range(len(config.CLASSES)))).tolist())
