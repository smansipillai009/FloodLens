"""
FloodLens Nepal — Sentinel-1 preprocessing on a common analysis grid
====================================================================
Input : terrain-corrected sigma0 GeoTIFFs (bands VV, VH[, dataMask]) from
        app/services/cdse_fetch.py (preferred) or from SNAP/pyroSAR/openEO.
        Raw GRD .SAFE zips are NOT accepted here: they need real
        orbit/thermal-noise/calibration/terrain correction (the old code only
        reprojected and squared DNs, which is not terrain correction).
Output: s1_{before,after}_{vv,vh}.tif  — linear sigma0, speckle-filtered,
        identical grid + metric CRS, nodata = 0, clipped to the AOI polygon.

Design rules
  * Everything is linear power until the very end; dB only for thresholds/display.
  * Each scene is resampled ONCE, straight onto the analysis grid (no chained warps).
  * Several same-track pre-event scenes are combined by per-pixel median.
  * Co-registration quality is MEASURED (phase correlation), never assumed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from app.config import get_config, utm_crs_for_geom

logger = logging.getLogger(__name__)
EPSILON = 1e-10
ENL_GRDH = 4.4          # equivalent number of looks of IW GRDH (used by the Lee filter)


# ── unit handling (explicit, no guessing) ─────────────────────────────────────

def calibrate_to_sigma0(raw_band: np.ndarray, units: str = "linear_sigma0") -> np.ndarray:
    """Return LINEAR sigma0 with nodata = 0.
    units: 'linear_sigma0' (already linear), 'db', or 'dn' (amplitude DN, sigma0 = DN^2 — only valid
    if the product is already radiometrically calibrated amplitude)."""
    arr = raw_band.astype(np.float32).copy()
    arr[~np.isfinite(arr)] = 0
    if units == "db":
        valid = arr != 0
        out = np.zeros_like(arr)
        out[valid] = np.power(10.0, arr[valid] / 10.0)
        return out
    if units == "dn":
        return np.where(arr > 0, arr ** 2, 0).astype(np.float32)
    if units == "linear_sigma0":
        return np.where(arr > 0, arr, 0).astype(np.float32)
    raise ValueError(f"Unknown units '{units}'")


# ── speckle filters (nodata-aware) ────────────────────────────────────────────

def _nan_mean(img: np.ndarray, valid: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    v = valid.astype(np.float64)
    cnt = uniform_filter(v, size=window)
    s = uniform_filter(np.where(valid, img, 0.0), size=window)
    return np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0), cnt


def boxcar_filter(band: np.ndarray, window: int = 5) -> np.ndarray:
    valid = band != 0
    mean, _ = _nan_mean(band.astype(np.float64), valid, window)
    return np.where(valid, mean, 0).astype(np.float32)


def lee_filter(band: np.ndarray, window: int = 5, enl: float = ENL_GRDH) -> np.ndarray:
    """Classic Lee (1980) filter in linear power. (The previous 'refined_lee' was
    not Refined Lee and used the wrong weight formula; this is honest about being Lee.)"""
    valid = band != 0
    img = band.astype(np.float64)
    mean, _ = _nan_mean(img, valid, window)
    mean2, _ = _nan_mean(img ** 2, valid, window)
    var = np.maximum(mean2 - mean ** 2, 0)
    cu2 = 1.0 / enl
    var_x = np.maximum((var - mean ** 2 * cu2) / (1.0 + cu2), 0)
    w = np.divide(var_x, var, out=np.zeros_like(var), where=var > 0)
    out = mean + w * (img - mean)
    return np.where(valid, np.maximum(out, 0), 0).astype(np.float32)


# ── analysis grid ─────────────────────────────────────────────────────────────

def make_grid(aoi_geojson: dict, metric_crs: str, res_m: float, pad_m: float = 100.0) -> dict:
    from pyproj import Transformer
    geom = shape(aoi_geojson)
    t = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    g_utm = shp_transform(t.transform, geom)
    minx, miny, maxx, maxy = g_utm.bounds
    minx, miny = np.floor((minx - pad_m) / res_m) * res_m, np.floor((miny - pad_m) / res_m) * res_m
    maxx, maxy = np.ceil((maxx + pad_m) / res_m) * res_m, np.ceil((maxy + pad_m) / res_m) * res_m
    w, h = int((maxx - minx) / res_m), int((maxy - miny) / res_m)
    tr = from_origin(minx, maxy, res_m, res_m)
    inside = features.rasterize([(g_utm, 1)], out_shape=(h, w), transform=tr, fill=0,
                                dtype="uint8", all_touched=True).astype(bool)
    return {"crs": metric_crs, "transform": tr, "width": w, "height": h,
            "res_m": res_m, "aoi_mask": inside}


def warp_band_to_grid(src_path: Path, band: int, grid: dict) -> np.ndarray:
    dst = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
    with rasterio.open(src_path) as src:
        if band > src.count:
            raise ValueError(f"{src_path.name} has {src.count} band(s); band {band} requested.")
        reproject(source=rasterio.band(src, band), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=CRS.from_string(grid["crs"]),
                  resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    return dst


def _write(path: Path, arr: np.ndarray, grid: dict) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"], width=grid["width"],
                       count=1, dtype="float32", crs=grid["crs"], transform=grid["transform"],
                       nodata=0.0, compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def multilook(arr: np.ndarray, k: int) -> np.ndarray:
    """Block-average k x k pixels of a LINEAR-power image (nodata = 0, ignored). A block needs > half valid pixels.
    Speckle is multiplicative noise: averaging N independent looks cuts its std by ~sqrt(N), which is what makes a -3 dB threshold usable."""
    if k <= 1:
        return arr
    h, w = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
    blocks = arr[:h, :w].reshape(h // k, k, w // k, k)
    valid = blocks > 0
    cnt = valid.sum(axis=(1, 3))
    tot = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    out = np.where(cnt > (k * k) // 2, tot / np.maximum(cnt, 1), 0.0)
    return out.astype(np.float32)


def estimate_coregistration_shift_m(ref: np.ndarray, mov: np.ndarray, res_m: float) -> Optional[float]:
    """Residual shift between pre and post VV (metres) via sub-pixel phase correlation."""
    try:
        from skimage.registration import phase_cross_correlation
    except Exception:  # noqa: BLE001
        logger.warning("scikit-image missing: alignment not measured.")
        return None
    valid = (ref > 0) & (mov > 0)
    if valid.mean() < 0.3:
        return None
    a = np.where(valid, 10 * np.log10(ref + EPSILON), 0)
    b = np.where(valid, 10 * np.log10(mov + EPSILON), 0)
    a, b = a - a[valid].mean(), b - b[valid].mean()
    a, b = np.where(valid, a, 0), np.where(valid, b, 0)
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
    return float(np.hypot(*shift) * res_m)


# ── public API ────────────────────────────────────────────────────────────────

def preprocess_s1_pair(
    before_tif: Path | Sequence[Path],
    after_tif: Path,
    run_dir: Path,
    aoi_geojson: dict,
    metric_crs: Optional[str] = None,
    resolution_m: Optional[float] = None,
    speckle_method: Optional[str] = None,
    input_units: Optional[str] = None,
    multilook_factor: Optional[int] = None,
) -> dict:
    """before_tif may be ONE path or several same-track pre-event paths (median reference).
    multilook_factor overrides config sentinel1.multilook_factor (the U-Net is trained on 10 m patches, so AI mode passes 1)."""
    cfg = get_config()["sentinel1"]
    metric_crs = metric_crs or utm_crs_for_geom(aoi_geojson)
    resolution_m = resolution_m or cfg["resolution_m"]
    speckle_method = speckle_method or cfg["speckle_filter"]
    units = input_units or cfg["input_units"]
    befores = [Path(before_tif)] if isinstance(before_tif, (str, Path)) else [Path(p) for p in before_tif]

    grid = make_grid(aoi_geojson, metric_crs, resolution_m)
    logger.info("Analysis grid: %dx%d px @ %.0f m, %s", grid["width"], grid["height"], resolution_m, metric_crs)

    def load(path: Path, band: int) -> np.ndarray:
        return calibrate_to_sigma0(warp_band_to_grid(path, band, grid), units)

    def speckle(a: np.ndarray) -> np.ndarray:
        if speckle_method == "boxcar":
            return boxcar_filter(a, cfg["speckle_window"])
        return lee_filter(a, cfg["speckle_window"])

    out: dict = {}
    arrays: dict = {}
    for pol, band in (("vv", 1), ("vh", 2)):
        stack = np.stack([load(p, band) for p in befores])
        stack = np.where(stack > 0, stack, np.nan)
        with np.errstate(all="ignore"):
            pre = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
        post = load(Path(after_tif), band)
        pre, post = speckle(pre), speckle(post)
        pre[~grid["aoi_mask"]] = 0
        post[~grid["aoi_mask"]] = 0
        arrays[pol] = (pre, post)
        if pol == "vv":
            shift = estimate_coregistration_shift_m(pre, post, resolution_m)   # measured at the native grid

    k = int(cfg.get("multilook_factor", 1)) if multilook_factor is None else int(multilook_factor)
    if k > 1:
        from affine import Affine
        arrays = {p: (multilook(a, k), multilook(b, k)) for p, (a, b) in arrays.items()}
        h, w = arrays["vv"][0].shape
        grid = {**grid, "transform": grid["transform"] @ Affine.scale(k), "width": w, "height": h,
                "aoi_mask": multilook(grid["aoi_mask"].astype(np.float32), k) > 0.5}
        resolution_m = resolution_m * k
        logger.info("Multilook x%d: analysis grid %dx%d px @ %.0f m (speckle reduction)", k, w, h, resolution_m)
    for pol, (pre, post) in arrays.items():
        out[f"s1_before_{pol}"] = _write(run_dir / f"s1_before_{pol}.tif", pre, grid)
        out[f"s1_after_{pol}"] = _write(run_dir / f"s1_after_{pol}.tif", post, grid)

    out["alignment_rmse_m"] = shift
    out["n_pre_event_scenes"] = len(befores)
    out["grid"] = {"crs": metric_crs, "res_m": resolution_m, "width": grid["width"], "height": grid["height"],
                   "transform": list(grid["transform"])[:6]}
    (run_dir / "s1_processing_metadata.json").write_text(json.dumps({
        "before_scenes": [str(p) for p in befores], "after_scene": str(after_tif),
        "metric_crs": metric_crs, "resolution_m": resolution_m, "speckle": speckle_method,
        "input_units": units, "measured_coregistration_shift_m": shift,
        "n_pre_event_scenes": len(befores)}, indent=2), encoding="utf-8")
    if shift is not None and shift > 1.5 * resolution_m:
        logger.warning("Residual co-registration shift %.1f m is > 1.5 px.", shift)
    return out


def to_db(linear_arr: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    arr = linear_arr.astype(np.float32)
    nodata = arr == 0
    result = 10.0 * np.log10(arr + epsilon)
    result[nodata] = np.nan
    return result


def verify_raster_grid_match(path_a: Path, path_b: Path) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        return (a.crs == b.crs and a.width == b.width and a.height == b.height
                and abs(a.transform.c - b.transform.c) < 0.1 and abs(a.transform.f - b.transform.f) < 0.1)
