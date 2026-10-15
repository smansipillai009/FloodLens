"""Tiled U-Net inference on the S1 analysis grid -> water probability GeoTIFF."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import ROOT, get_config
from app.ml.model import build_model, make_input

logger = logging.getLogger(__name__)


def checkpoint_available() -> bool:
    c = get_config()["ml"]
    return (ROOT / c["checkpoint"]).exists() and (ROOT / c["norm_stats"]).exists()


def predict_water_probability(before_vv: Path, before_vh: Path, after_vv: Path, after_vh: Path,
                              out_path: Path) -> Optional[Path]:
    import rasterio
    import torch
    cfg = get_config()["ml"]
    if not checkpoint_available():
        logger.warning("No trained checkpoint: AI mode unavailable.")
        return None
    norm = json.loads((ROOT / cfg["norm_stats"]).read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], "float32").reshape(-1)[:, None, None]; std = np.asarray(norm["std"], "float32").reshape(-1)[:, None, None]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / cfg["checkpoint"], map_location=dev)
    model = build_model(ck.get("encoder", cfg["encoder"])).to(dev); model.load_state_dict(ck["state"]); model.eval()

    arrs = []
    for p in (before_vv, before_vh, after_vv, after_vh):
        with rasterio.open(p) as s:
            a = s.read(1).astype("float32"); meta = s.meta.copy()
        arrs.append(np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-10)), np.nan))
    valid = np.all([np.isfinite(a) for a in arrs], axis=0)
    # nodata -> typical low dB so tiles stay finite; masked out afterwards
    x = make_input(*[np.where(np.isfinite(a), a, -25.0) for a in arrs])
    x = (x - mean) / std
    H, W = x.shape[1:]
    T, O = cfg["tile_size"], cfg["tile_overlap"]
    step = T - 2 * O
    acc, wsum = np.zeros((H, W), "float32"), np.zeros((H, W), "float32")
    with torch.no_grad():
        for r in range(0, max(H - 2 * O, 1), step):
            for c in range(0, max(W - 2 * O, 1), step):
                r0, c0 = min(r, max(H - T, 0)), min(c, max(W - T, 0))
                tile = x[:, r0:r0 + T, c0:c0 + T]
                ph, pw = T - tile.shape[1], T - tile.shape[2]
                tile = np.pad(tile, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(tile.shape[1:]) > 1 else "edge")
                pr = torch.sigmoid(model(torch.tensor(tile[None]).to(dev)))[0, 0].cpu().numpy()
                pr = pr[:T - ph, :T - pw]
                acc[r0:r0 + pr.shape[0], c0:c0 + pr.shape[1]] += pr
                wsum[r0:r0 + pr.shape[0], c0:c0 + pr.shape[1]] += 1
    prob = np.where(valid & (wsum > 0), acc / np.maximum(wsum, 1), 0).astype("float32")
    meta.update(dtype="float32", nodata=0.0, count=1, compress="deflate")
    with rasterio.open(out_path, "w", **meta) as d:
        d.write(prob, 1)
    return out_path


def ensemble_allowed() -> bool:
    cfg = get_config()["ml"]
    if not cfg["ensemble_requires_validation"]:
        return True
    mf = ROOT / cfg["metrics_file"]
    return mf.exists() and bool(json.loads(mf.read_text(encoding="utf-8")).get("ensemble_validated"))
