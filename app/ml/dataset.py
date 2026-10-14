"""
Patch dataset for flood segmentation.

Expected on-disk format (produced by scripts/prepare_kurosiwo.py — you write the adapter after
checking how Kuro Siwo ships its data on Day 1):
    <root>/<split>/<event_id>/<patch>.npz  with arrays
        sar  float32 or float16 (4, H, W)  = pre_VV, pre_VH, post_VV, post_VH in **dB**
        y    uint8   (H, W)     = 1 flood water, 0 other   (permanent water -> 0 or ignore)
        valid uint8  (H, W)     = 1 usable pixel
Split by EVENT (never random patches): train / val / test(held-out events). The Himalayan
check-only reference events are NEVER in train/val.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.ml.model import make_input


class PatchDataset:
    def __init__(self, root: Path, split: str, norm: dict | None = None, augment: bool = False):
        self.files = sorted((Path(root) / split).rglob("*.npz"))
        self.norm, self.augment = norm, augment
        if not self.files:
            raise FileNotFoundError(f"No .npz patches under {Path(root) / split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        x = make_input(*z["sar"].astype("float32"))   # patches may be stored as float16 to save disk
        if self.norm:
            x = (x - np.array(self.norm["mean"], "float32")[:, None, None]) / np.array(self.norm["std"], "float32")[:, None, None]
        y, v = z["y"].astype("float32"), z["valid"].astype("float32")
        if self.augment:
            if np.random.rand() < .5: x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
            if np.random.rand() < .5: x, y, v = x[:, ::-1], y[::-1], v[::-1]
        return (np.ascontiguousarray(x), np.ascontiguousarray(y)[None], np.ascontiguousarray(v)[None])


def compute_norm_stats(ds: PatchDataset, max_patches: int = 500) -> dict:
    acc, acc2, n = 0.0, 0.0, 0
    for i in np.linspace(0, len(ds) - 1, min(len(ds), max_patches)).astype(int):
        x, _, v = ds[int(i)]
        m = v[0] > 0
        xs = x[:, m]
        acc, acc2, n = acc + xs.sum(1), acc2 + (xs ** 2).sum(1), n + xs.shape[1]
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-6))
    return {"mean": mean.tolist(), "std": std.tolist()}
