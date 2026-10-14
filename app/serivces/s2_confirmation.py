"""
FloodLens Nepal — Sentinel-2 optical confirmation (clear pixels only)
=====================================================================
Replaces the old "cloud reconstruction" idea: a generative/reconstructed pixel is not an
observation and the plan forbids using it for any count. Here every SAR candidate pixel gets
one of four optical states:

  0 = not a candidate
  1 = SAR-only       (optical not clear: cloud / cloud shadow / topographic shadow / snow / no data)
  2 = confirmed      (clear optical agrees)
  3 = contradicted   (clear optical disagrees)

Water  : MNDWI = (B03 - B11) / (B03 + B11) > threshold.
Debris : NDVI drop vs pre-event clear S2 if supplied, else a weaker 'bare/sparse in post' test.
Snow/ice (SCL 11) is excluded from 'clear' because snow has high MNDWI and would fake water.
Note: S2 is acquired on a different day than S1 -> disagreement is not proof of error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation

from app.config import get_config

logger = logging.getLogger(__name__)
SCL_CLEAR = (4, 5, 6, 7)            # vegetation, bare soil, water, unclassified
SCL_CLOUDY = (3, 8, 9, 10)          # cloud shadow, cloud med/high, thin cirrus


def cloud_and_clear_masks(scl: np.ndarray, data_ok: np.ndarray, dilation_px: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.isin(scl, SCL_CLOUDY)
    if dilation_px > 0:
        cloud = binary_dilation(cloud, iterations=dilation_px)
    clear = np.isin(scl, SCL_CLEAR) & ~cloud & data_ok
    return cloud, clear


def _safe_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    return np.divide(a - b, den, out=np.full(a.shape, np.nan, dtype=np.float32), where=np.abs(den) > 1e-6)


def classify_optical_agreement(
    sar_class: np.ndarray, clear: np.ndarray, mndwi: np.ndarray,
    ndvi_post: np.ndarray, ndvi_pre: Optional[np.ndarray],
    mndwi_thr: float, ndvi_drop_thr: float, bare_ndvi_max: float = 0.2,
) -> np.ndarray:
    """sar_class: 1 water, 2 debris (from flood_class.tif). Returns optical state array (0-3)."""
    out = np.zeros(sar_class.shape, dtype=np.uint8)
    water, debris = sar_class == 1, sar_class == 2
    cand = water | debris
    out[cand] = 1                                       # default: SAR-only
    w_conf = water & clear & (mndwi > mndwi_thr)
    w_contra = water & clear & (mndwi <= mndwi_thr)
    if ndvi_pre is not None:
        d_evidence = (ndvi_pre - ndvi_post) > ndvi_drop_thr
    else:
        d_evidence = ndvi_post < bare_ndvi_max
    d_conf = debris & clear & d_evidence & ~(mndwi > mndwi_thr)
    d_contra = debris & clear & ~d_evidence
    out[w_conf | d_conf] = 2
    out[w_contra | d_contra] = 3
    return out


def run_s2_confirmation(s2_tif: Path, flood_class_path: Path, run_dir: Path,
                        s2_pre_tif: Optional[Path] = None) -> dict:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    cfg = get_config()["sentinel2"]

    def on_grid(path: Path, band: int, ref: rasterio.DatasetReader, method) -> np.ndarray:
        dst = np.zeros((ref.height, ref.width), dtype=np.float32)
        with rasterio.open(path) as s:
            reproject(rasterio.band(s, band), dst, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=ref.transform, dst_crs=ref.crs, resampling=method)
        return dst

    with rasterio.open(flood_class_path) as ref:
        sar_class = ref.read(1)
        meta, px = ref.meta.copy(), abs(ref.transform.a)
        b03, b04, b08, b11 = (on_grid(s2_tif, i, ref, Resampling.bilinear) for i in (1, 2, 3, 4))
        scl = on_grid(s2_tif, 5, ref, Resampling.nearest).astype(np.uint8)
        ok = on_grid(s2_tif, 6, ref, Resampling.nearest) > 0
        pre_ndvi = None
        if s2_pre_tif and Path(s2_pre_tif).exists():
            pr, pn = (on_grid(s2_pre_tif, i, ref, Resampling.bilinear) for i in (2, 3))
            pre_ndvi = _safe_nd(pn, pr)

    cloud, clear = cloud_and_clear_masks(scl, ok, cfg["cloud_dilation_pixels"])
    mndwi, ndvi = _safe_nd(b03, b11), _safe_nd(b08, b04)
    opt = classify_optical_agreement(sar_class, clear, mndwi, ndvi, pre_ndvi,
                                     cfg["mndwi_water_threshold"], cfg["ndvi_drop_threshold"])

    m = meta.copy(); m.update(dtype="uint8", nodata=255, count=1)
    for name, arr in (("optical_class", opt), ("cloud_mask", cloud.astype(np.uint8))):
        with rasterio.open(run_dir / f"{name}.tif", "w", **m) as d:
            d.write(arr, 1)

    n_cand = int((sar_class > 0).sum())
    conf, contra = int((opt == 2).sum()), int((opt == 3).sum())
    clear_cand = conf + contra
    cloud_pct = 100.0 * float(cloud[ok].mean()) if ok.any() else None
    enough = n_cand > 0 and (clear_cand / n_cand) >= cfg["min_clear_fraction_for_confirmation"]
    agreement = 100.0 * conf / clear_cand if (enough and clear_cand) else None
    return {
        "optical_cloud_pct_in_aoi": cloud_pct,
        "optical_agreement_pct": agreement,
        "optical_confirmed_km2": round(conf * px * px / 1e6, 4),
        "optical_contradicted_km2": round(contra * px * px / 1e6, 4),
        "optical_clear_fraction_of_candidates": (clear_cand / n_cand) if n_cand else None,
        "optical_class_path": run_dir / "optical_class.tif",
        "insufficient_optical": not enough,
    }
