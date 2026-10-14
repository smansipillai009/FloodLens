"""Segmentation metrics (numpy only so they can be tested without torch)."""
from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict:
    pred, truth = pred.astype(bool), truth.astype(bool)
    if valid is not None:
        pred, truth = pred[valid], truth[valid]
    return {"tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()),
            "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())}


def scores(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"iou": d(tp, tp + fp + fn), "f1": d(2 * tp, 2 * tp + fp + fn),
            "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
            "area_diff_pct": d(100.0 * ((tp + fp) - (tp + fn)), tp + fn)}


def add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}
