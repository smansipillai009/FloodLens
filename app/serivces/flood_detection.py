"""
FloodLens Nepal — SAR flood & debris change detector (two-sided baseline)
=========================================================================
Water  : VV backscatter DROPS (change <= water_change_db_vv) AND post-event VV is
         open-water dark (<= water_post_max_db_vv)  AND not permanent water
         (pre-event VV already dark).  The old version used the change test alone, so any
         dark-to-darker pixel (wet soil, shadow) became "flood".
Debris : VH backscatter RISES (>= debris_change_db_vh), near the channel (HAND limit).
Both   : terrain-masked (slope + radar layover/shadow), size-filtered.
All thresholds come from configs/config.yaml (the old code ignored the config).

Outputs in run_dir
  sar_change_vv.tif, sar_change_vh.tif   change in dB
  flood_probability.tif                  heuristic 0-1 score (NOT a calibrated probability)
  water_mask.tif, debris_mask.tif        uint8
  flood_class.tif                        0 none / 1 water / 2 debris
  flood_mask.tif                         union, uint8
  flood_polygons.geojson                 EPSG:4326, property `class` = water|debris
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import binary_dilation, label
from shapely.geometry import mapping, shape as shp_shape

from app.config import get_config

logger = logging.getLogger(__name__)
EPSILON = 1e-10


# ── primitives (signatures kept for the unit tests) ──────────────────────────

def compute_log_ratio(pre_band: np.ndarray, post_band: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """10*log10(post/pre) on LINEAR power. 0 = nodata -> NaN."""
    pre, post = pre_band.astype(np.float64), post_band.astype(np.float64)
    nodata = (pre <= 0) | (post <= 0) | ~np.isfinite(pre) | ~np.isfinite(post)
    out = 10.0 * (np.log10(np.where(nodata, 1, post) + epsilon) - np.log10(np.where(nodata, 1, pre) + epsilon))
    return np.where(nodata, np.nan, out).astype(np.float32)


def threshold_change_map(change_map: np.ndarray, threshold: float, direction: str = "decrease") -> np.ndarray:
    valid = np.isfinite(change_map)
    flagged = valid & (change_map < -abs(threshold)) if direction == "decrease" \
        else valid & (change_map > abs(threshold))
    return flagged.astype(np.uint8)


def apply_terrain_mask(candidate_mask: np.ndarray, terrain_mask: Optional[np.ndarray]) -> np.ndarray:
    if terrain_mask is None:
        return candidate_mask
    if terrain_mask.shape != candidate_mask.shape:
        raise ValueError(f"terrain mask {terrain_mask.shape} != candidate {candidate_mask.shape}; "
                         "resample with terrain.resample_to_reference first.")
    return (candidate_mask.astype(bool) & (terrain_mask == 1)).astype(np.uint8)


def remove_small_patches(mask: np.ndarray, pixel_area_m2: float, min_area_m2: float = 2500) -> np.ndarray:
    min_pixels = max(1, int(round(min_area_m2 / pixel_area_m2)))
    labeled, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    keep_ids = np.flatnonzero(counts >= min_pixels)
    keep_ids = keep_ids[keep_ids != 0]
    keep = np.isin(labeled, keep_ids)
    removed = int(mask.sum()) - int(keep.sum())
    if removed:
        logger.info("Removed %d px in patches < %.0f m2.", removed, min_area_m2)
    return keep.astype(np.uint8)


def compute_flood_probability(change_vv: np.ndarray, change_vh: np.ndarray,
                              vv_threshold: float = -3.0, vh_threshold: float = 3.0) -> np.ndarray:
    """Heuristic evidence score in [0,1] (logistic of distance beyond threshold). Not calibrated."""
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    vv = np.nan_to_num(change_vv, nan=0.0).astype(np.float64)
    vh = np.nan_to_num(change_vh, nan=0.0).astype(np.float64)
    prob = np.maximum(sig((-vv - abs(vv_threshold)) / 2.0), sig((vh - abs(vh_threshold)) / 2.0)).astype(np.float32)
    prob[~(np.isfinite(change_vv) | np.isfinite(change_vh))] = 0.0
    return prob


def mask_to_polygons(class_raster: np.ndarray, transform, crs_epsg: int,
                     simplify_tolerance_m: float = 10.0) -> dict:
    """Vectorise a class raster (1 water, 2 debris). Geometry stays in the raster's metric CRS."""
    names = {1: "water", 2: "debris"}
    feats = []
    for geom_dict, val in rio_shapes(class_raster.astype(np.uint8), mask=class_raster > 0, transform=transform):
        v = int(val)
        if v not in names:
            continue
        g = shp_shape(geom_dict).simplify(simplify_tolerance_m, preserve_topology=True)
        if g.is_empty:
            continue
        feats.append({"type": "Feature",
                      "properties": {"class": names[v], "label": "flood_debris_candidate",
                                     "area_m2": round(g.area, 1), "crs_epsg": crs_epsg},
                      "geometry": mapping(g)})
    return {"type": "FeatureCollection", "features": feats, "crs_epsg": crs_epsg}


def _write_polygons_wgs84(fc_metric: dict, out_path: Path, epsg: int) -> None:
    """BUGFIX: the old file was written in UTM metres with no CRS; geopandas then read it as
    lon/lat and every downstream overlay was wrong. Standard GeoJSON must be EPSG:4326."""
    import geopandas as gpd
    if not fc_metric["features"]:
        out_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return
    gdf = gpd.GeoDataFrame.from_features(fc_metric["features"], crs=f"EPSG:{epsg}").to_crs("EPSG:4326")
    out_path.write_text(gdf.to_json(), encoding="utf-8")


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as s:
        return s.read(1).astype(np.float32)


def _save(path: Path, arr: np.ndarray, meta: dict, dtype: str, nodata) -> None:
    m = meta.copy(); m.update(dtype=dtype, nodata=nodata, count=1, compress="deflate")
    with rasterio.open(path, "w", **m) as dst:
        dst.write(arr.astype(dtype), 1)


# ── main entry ────────────────────────────────────────────────────────────────

def robust_sigma(change: np.ndarray, valid: np.ndarray) -> float:
    """Noise sigma of a change map from the median absolute deviation (insensitive to the few % of real change)."""
    v = change[valid & np.isfinite(change)]
    if v.size < 100:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def adaptive_thresholds(th: dict, s_vv: float, s_vh: float) -> dict:
    """Effective thresholds: never weaker than the configured ones, scaled up to the scene noise."""
    ks, kg = th.get("adaptive_k_seed", 3.0), th.get("adaptive_k_grow", 1.75)
    return {"water_change_db_vv": min(th["water_change_db_vv"], -ks * s_vv),
            "water_grow_change_db_vv": min(th["water_grow_change_db_vv"], -kg * s_vv),
            "debris_change_db_vh": max(th["debris_change_db_vh"], ks * s_vh),
            "debris_grow_change_db_vh": max(th["debris_grow_change_db_vh"], kg * s_vh)}


def remove_regional_trend(change: np.ndarray, valid: np.ndarray, sigma_px: float, clip_db: float = 3.0) -> np.ndarray:
    """Subtract a smooth, outlier-resistant regional mean (normalised Gaussian convolution of the clipped change)."""
    from scipy.ndimage import gaussian_filter
    if sigma_px <= 0:
        return change
    ok = valid & np.isfinite(change)
    c = np.where(ok, np.clip(change, -clip_db, clip_db), 0.0)
    num = gaussian_filter(c, sigma_px, mode="nearest")
    den = gaussian_filter(ok.astype(np.float64), sigma_px, mode="nearest")
    trend = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return np.where(np.isfinite(change), change - trend, change)


def hysteresis_grow(seed: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Connected components (8-neighbourhood) of `weak` that contain at least one `seed` pixel."""
    from scipy.ndimage import label
    weak = weak | seed
    lab, n = label(weak, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return seed.copy()
    keep = np.unique(lab[seed & weak])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def detect_flood(
    before_vv: Path, before_vh: Path, after_vv: Path, after_vh: Path, run_dir: Path,
    terrain: Optional[dict] = None,
    water_prob_path: Optional[Path] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """
    terrain : dict from terrain.prepare_terrain_layers (terrain_mask, hand paths or None)
    water_prob_path : optional U-Net water probability on the SAME grid; if given it replaces the
                      baseline water rule (debris always comes from the baseline).
    """
    from app.services.terrain import resample_to_reference
    cfg = get_config()
    fd, dem_cfg = cfg["flood_detection"], cfg["dem"]
    th = {**fd, **(thresholds or {})}

    with rasterio.open(before_vv) as src:
        meta, transform = src.meta.copy(), src.transform
        px = abs(transform.a)
        epsg = src.crs.to_epsg() or 0
    px_area = px * px
    pre_vv, pre_vh, post_vv, post_vh = (_read(p) for p in (before_vv, before_vh, after_vv, after_vh))
    valid = (pre_vv > 0) & (post_vv > 0) & (pre_vh > 0) & (post_vh > 0)

    change_vv, change_vh = compute_log_ratio(pre_vv, post_vv), compute_log_ratio(pre_vh, post_vh)
    raw_stats = {"vv": change_vv.copy(), "vh": change_vh.copy()}
    sigma_px = th.get("detrend_sigma_m", 0) / px if px > 0 else 0
    if sigma_px > 0:
        change_vv = remove_regional_trend(change_vv, valid, sigma_px, th.get("detrend_clip_db", 3.0)).astype(np.float32)
        change_vh = remove_regional_trend(change_vh, valid, sigma_px, th.get("detrend_clip_db", 3.0)).astype(np.float32)
    pre_db = 10 * np.log10(np.where(pre_vv > 0, pre_vv, 1) + EPSILON)
    post_db = 10 * np.log10(np.where(post_vv > 0, post_vv, 1) + EPSILON)

    fmeta = meta.copy(); fmeta.update(dtype="float32", nodata=np.nan)
    _save(run_dir / "sar_change_vv.tif", change_vv, fmeta, "float32", np.nan)
    _save(run_dir / "sar_change_vh.tif", change_vh, fmeta, "float32", np.nan)

    noise = {"sigma_vv_db": round(robust_sigma(change_vv, valid), 3), "sigma_vh_db": round(robust_sigma(change_vh, valid), 3)}
    if th.get("adaptive_thresholds", False):
        th = {**th, **adaptive_thresholds(th, noise["sigma_vv_db"], noise["sigma_vh_db"])}

    # permanent water (pre-event reference only), slightly dilated
    perm = binary_dilation((pre_db < th["permanent_water_pre_max_db_vv"]) & (pre_vv > 0), iterations=1)

    # terrain + HAND on the S1 grid
    terr = None
    hand = None
    warnings: list[str] = []
    if terrain and terrain.get("terrain_mask"):
        terr = resample_to_reference(terrain["terrain_mask"], before_vv, fill=0)
    else:
        warnings.append("No DEM terrain mask: false positives on steep slopes / radar shadow are NOT filtered.")
    if terrain and terrain.get("hand"):
        from rasterio.enums import Resampling
        hand = resample_to_reference(terrain["hand"], before_vv, Resampling.bilinear, fill=-9999)
        is_proxy = terrain.get("hand_method") != "pysheds_d8_hand"
        lim_w = dem_cfg["hand_proxy_max_m_water"] if is_proxy else dem_cfg["hand_max_m_water"]
        lim_d = dem_cfg["hand_proxy_max_m_debris"] if is_proxy else dem_cfg["hand_max_m_debris"]
        use_hand = (not is_proxy) or bool(dem_cfg.get("hand_proxy_filter", False))
        hand_info = {"method": terrain.get("hand_method"), "used_as_filter": use_hand, "limit_water_m": lim_w, "limit_debris_m": lim_d}
        hand_ok_w = (hand < 0) | (hand <= lim_w) if use_hand else np.ones_like(valid)
        hand_ok_d = (hand < 0) | (hand <= lim_d) if use_hand else np.ones_like(valid)
    else:
        hand_ok_w = hand_ok_d = np.ones_like(valid)

    # baseline rules (every stage is counted so an empty result can be explained: see detection_funnel.json)
    funnel: dict = {"grid_pixels": int(valid.size), "valid_pixels": int(valid.sum())}
    if hand is not None:
        funnel["hand"] = hand_info
    w_chg = threshold_change_map(change_vv, th["water_change_db_vv"], "decrease").astype(bool)
    w_dark = post_db <= th["water_post_max_db_vv"]
    d_chg = threshold_change_map(change_vh, th["debris_change_db_vh"], "increase").astype(bool)
    funnel["water"] = {"1_vv_drop_enough": int((w_chg & valid).sum()),
                       "2_and_post_dark_enough": int((w_chg & w_dark & valid).sum()),
                       "3_and_not_permanent_water": int((w_chg & w_dark & ~perm & valid).sum()),
                       "4_and_HAND_ok": int((w_chg & w_dark & ~perm & valid & hand_ok_w).sum())}
    funnel["debris"] = {"1_vh_rise_enough": int((d_chg & valid).sum()),
                        "2_and_not_permanent_water": int((d_chg & ~perm & valid).sum()),
                        "3_and_HAND_ok": int((d_chg & ~perm & valid & hand_ok_d).sum())}
    if hand is not None:      # where do the candidates sit? (tells whether the HAND limit, not the signal, is the problem)
        def _hp(m):
            v = hand[m & (hand > -9000)]
            return {f"p{q}": round(float(np.percentile(v, q)), 1) for q in (10, 50, 90)} if v.size else {}
        funnel["hand_at_candidates_m"] = {"water": _hp(w_chg & w_dark & ~perm & valid), "debris": _hp(d_chg & ~perm & valid)}
    water = w_chg & w_dark & ~perm & valid & hand_ok_w
    debris = d_chg & ~perm & valid & hand_ok_d
    funnel["seeds_px"] = {"water": int(water.sum()), "debris": int(debris.sum())}
    if th.get("hysteresis", False):
        w_weak = (change_vv <= th["water_grow_change_db_vv"]) & (post_db <= th["water_grow_post_max_db_vv"]) & ~perm & valid & hand_ok_w
        d_weak = (change_vh >= th["debris_grow_change_db_vh"]) & ~perm & valid & hand_ok_d
        water, debris = hysteresis_grow(water, w_weak), hysteresis_grow(debris, d_weak)
        funnel["after_hysteresis_px"] = {"water": int(water.sum()), "debris": int(debris.sum())}
    method = "baseline_log_ratio"
    if water_prob_path is not None and Path(water_prob_path).exists():
        ai_p = _read(water_prob_path)
        water = (ai_p >= get_config()["ml"]["prob_threshold"]) & ~perm & valid & hand_ok_w
        method = "unet_water+baseline_debris"
    debris &= ~water

    w_t = apply_terrain_mask(water.astype(np.uint8), terr)
    d_t = apply_terrain_mask(debris.astype(np.uint8), terr)
    funnel["water"]["5_after_terrain_mask"] = int(w_t.sum())
    funnel["debris"]["4_after_terrain_mask"] = int(d_t.sum())
    water = remove_small_patches(w_t, px_area, th["min_area_m2"])
    debris = remove_small_patches(d_t, px_area, th["min_area_m2"])
    funnel["water"]["6_after_min_patch_size"] = int(water.sum())
    funnel["debris"]["5_after_min_patch_size"] = int(debris.sum())
    pct = lambda a: {f"p{q:02d}": round(float(np.percentile(a[valid], q)), 2) for q in (1, 5, 50, 95, 99)} if valid.any() else {}  # noqa: E731
    funnel["stats_db"] = {"pre_vv_db": pct(pre_db), "post_vv_db": pct(post_db),
                          "log_ratio_vv_db": pct(raw_stats["vv"]), "log_ratio_vh_db": pct(raw_stats["vh"]),
                          "log_ratio_vv_db_after_detrend": pct(change_vv), "log_ratio_vh_db_after_detrend": pct(change_vh)}
    funnel["noise"] = noise
    funnel["thresholds"] = {k: round(float(th[k]), 3) if isinstance(th[k], (int, float)) else th[k] for k in (
        "water_change_db_vv", "water_grow_change_db_vv", "water_post_max_db_vv", "debris_change_db_vh", "debris_grow_change_db_vh",
        "permanent_water_pre_max_db_vv", "min_area_m2") if k in th}
    funnel["thresholds_are_adaptive"] = bool(th.get("adaptive_thresholds", False))
    (run_dir / "detection_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    debris = (debris.astype(bool) & ~water.astype(bool)).astype(np.uint8)
    cls = np.where(water == 1, 1, np.where(debris == 1, 2, 0)).astype(np.uint8)
    union = (cls > 0).astype(np.uint8)

    prob = compute_flood_probability(change_vv, change_vh, th["water_change_db_vv"], th["debris_change_db_vh"])
    prob[~valid] = 0
    _save(run_dir / "flood_probability.tif", prob, fmeta, "float32", np.nan)
    for name, arr in (("water_mask", water), ("debris_mask", debris), ("flood_mask", union), ("flood_class", cls)):
        _save(run_dir / f"{name}.tif", arr, meta, "uint8", 0)

    fc = mask_to_polygons(cls, transform, epsg)
    poly_path = run_dir / "flood_polygons.geojson"
    _write_polygons_wgs84(fc, poly_path, epsg)

    km2 = lambda n: round(float(n) * px_area / 1e6, 4)  # noqa: E731
    res = {
        "flood_mask_path": run_dir / "flood_mask.tif", "flood_class_path": run_dir / "flood_class.tif",
        "flood_polygons_path": poly_path, "flood_probability_path": run_dir / "flood_probability.tif",
        "flood_area_km2": km2(union.sum()), "water_area_km2": km2(water.sum()),
        "debris_area_km2": km2(debris.sum()),
        "water_pixel_count": int(water.sum()), "debris_pixel_count": int(debris.sum()),
        "valid_fraction": float(valid.mean()), "method": method, "warnings": warnings,
        "pixel_size_m": px, "metric_epsg": epsg, "funnel": funnel,
    }
    logger.info("Detection (%s): water %.3f km2, debris %.3f km2 (valid %.0f%% of grid)",
                method, res["water_area_km2"], res["debris_area_km2"], 100 * res["valid_fraction"])
    if res["water_pixel_count"] + res["debris_pixel_count"] == 0:
        logger.warning("Nothing detected. Funnel (pixels): water %s | debris %s | valid %d of %d. See detection_funnel.json",
                       funnel["water"], funnel["debris"], funnel["valid_pixels"], funnel["grid_pixels"])
    return res
