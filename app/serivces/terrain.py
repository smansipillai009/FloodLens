"""
FloodLens Nepal — Copernicus DEM Terrain Processing
=====================================================
Downloads Copernicus DEM GLO-30 tiles, builds a mosaic,
computes slope, HAND approximation, layover/shadow mask,
and combined terrain exclusion mask.

Outputs: dem.tif, slope.tif, hand_approx.tif,
         layover_shadow_mask.tif, terrain_mask.tif
Depends on: rasterio, numpy, scipy, requests, pyproj
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import minimum_filter
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

COP_DEM_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
# Real AWS naming: Copernicus_DSM_COG_10_N27_00_E085_00_DEM  (note the _00 suffixes)
TILE_PATTERN = (
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
    "/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
)
DEFAULT_MAX_SLOPE_DEG = 30.0
DEFAULT_HAND_MAX_M    = 15.0
DEFAULT_RESOLUTION_M  = 30.0


def _dem_tile_name(lat: int, lon: int) -> str:
    lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return TILE_PATTERN.format(lat=lat_str, lon=lon_str)


def _tiles_for_bbox(minx, miny, maxx, maxy):
    return [
        (lat, lon)
        for lat in range(math.floor(miny), math.floor(maxy) + 1)
        for lon in range(math.floor(minx), math.floor(maxx) + 1)
    ]


def download_dem_tile(lat: int, lon: int, cache_dir: Path) -> Optional[Path]:
    tile_name = _dem_tile_name(lat, lon)
    url = f"{COP_DEM_S3_BASE}/{tile_name}"
    out_path = cache_dir / f"dem_{lat}_{lon}.tif"
    if out_path.exists():
        return out_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading DEM tile: %s", tile_name)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 404:
            logger.warning("DEM tile 404 (ocean tile or wrong URL): %s", url)
            return None
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                f.write(chunk)
        return out_path
    except requests.exceptions.RequestException as exc:
        logger.error("DEM tile download failed: %s", exc)
        return None


def _reproject_dem(src_path, out_path, metric_crs, resolution_m):
    # nearest-compatible bilinear resampling of elevation is fine at 30 m
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, CRS.from_string(metric_crs),
            src.width, src.height, *src.bounds,
            resolution=(resolution_m, resolution_m),
        )
        meta = src.meta.copy()
        meta.update(crs=metric_crs, transform=transform,
                    width=width, height=height,
                    dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=metric_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=-9999, dst_nodata=-9999.0,
                )


def build_dem_mosaic(
    aoi_geojson: dict,
    cache_dir: Path,
    out_path: Path,
    metric_crs: str = "EPSG:32645",   # callers pass the AOI-specific UTM zone
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> Optional[Path]:
    geom = shape(aoi_geojson)
    tiles_needed = _tiles_for_bbox(*geom.bounds)
    logger.info("DEM: %d tile(s) needed.", len(tiles_needed))
    tile_paths = [p for lat, lon in tiles_needed
                  if (p := download_dem_tile(lat, lon, cache_dir))]
    if not tile_paths:
        logger.error("No DEM tiles available.")
        return None
    sources = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = rio_merge(sources)
    mosaic_meta = sources[0].meta.copy()
    mosaic_meta.update(height=mosaic_arr.shape[1],
                       width=mosaic_arr.shape[2],
                       transform=mosaic_transform)
    for src in sources:
        src.close()
    tmp_mosaic = cache_dir / "dem_mosaic_tmp.tif"
    with rasterio.open(tmp_mosaic, "w", **mosaic_meta) as dst:
        dst.write(mosaic_arr)
    with rasterio.open(tmp_mosaic) as src:
        clipped, clip_transform = rio_mask(src, [mapping(geom)], crop=True, nodata=-9999)
        clip_meta = src.meta.copy()
        clip_meta.update(height=clipped.shape[1], width=clipped.shape[2],
                         transform=clip_transform)
    tmp_clipped = cache_dir / "dem_clipped_tmp.tif"
    with rasterio.open(tmp_clipped, "w", **clip_meta) as dst:
        dst.write(clipped)
    _reproject_dem(tmp_clipped, out_path, metric_crs, resolution_m)
    logger.info("DEM ready: %s", out_path)
    return out_path


def _horn(dem: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(dem, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, e = p[1:-1, :-2], p[1:-1, 2:]
    f, g, h = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dz_dx = ((c + 2 * e + h) - (a + 2 * d + f)) / (8.0 * cell)   # +x = east
    dz_dy = ((a + 2 * b + c) - (f + 2 * g + h)) / (8.0 * cell)   # d z / d north (row 0 is the north edge)
    return dz_dx, dz_dy


def compute_slope(dem_path: Path, out_path: Path, aspect_path: Optional[Path] = None) -> Path:
    """Horn (1981) slope in degrees; optionally also writes aspect (deg clockwise from north,
    direction the slope FACES / descends towards)."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        cell = abs(src.transform.a)
        meta = src.meta.copy()
        meta.update(dtype="float32", nodata=-9999.0)
    dem = np.where(dem == nodata, np.nan, dem)
    dz_dx, dz_dy = _horn(dem, cell)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    bad = np.isnan(dem) | np.isnan(slope)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(np.where(bad, -9999.0, slope).astype(np.float32), 1)
    if aspect_path is not None:
        # downslope direction: (-dz/dx, -dz/dy) in (east, north)
        asp = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0
        with rasterio.open(aspect_path, "w", **meta) as dst:
            dst.write(np.where(bad, -9999.0, asp).astype(np.float32), 1)
    valid = slope[~bad]
    if valid.size:
        logger.info("Slope: mean=%.1f deg max=%.1f deg", valid.mean(), valid.max())
    return out_path


def compute_hand(dem_path: Path, out_path: Path, window: int = 51) -> tuple[Path, str]:
    """Height Above Nearest Drainage.
    Uses pysheds (true D8 flow-routing HAND) when installed; otherwise falls back to
    a LOCAL-RELIEF proxy (elevation minus the minimum within `window` cells).
    Returns (path, method).  The method is logged in the manifest and shown in the report:
    the proxy is NOT HAND and must not be called HAND."""
    try:
        from pysheds.grid import Grid  # type: ignore
        grid = Grid.from_raster(str(dem_path))
        dem = grid.read_raster(str(dem_path))
        pit = grid.fill_pits(dem); flooded = grid.fill_depressions(pit); inflated = grid.resolve_flats(flooded)
        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)
        stream = acc > max(200, np.nanpercentile(acc, 98))
        hand = grid.compute_hand(fdir, inflated, stream)
        arr = np.asarray(hand, dtype=np.float32)
        with rasterio.open(dem_path) as src:
            meta = src.meta.copy()
        meta.update(dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32), 1)
        return out_path, "pysheds_d8_hand"
    except Exception as exc:  # noqa: BLE001
        logger.warning("pysheds HAND unavailable (%s); using local-relief proxy.", exc)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        meta = src.meta.copy()
        meta.update(dtype="float32", nodata=-9999.0)
    valid = dem != nodata
    filled = np.where(valid, dem, np.inf)
    local_min = minimum_filter(filled, size=window)
    hand = np.where(valid, dem - local_min, -9999.0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(hand.astype(np.float32), 1)
    return out_path, "local_relief_proxy"


# kept for backwards compatibility with older imports/tests
def compute_hand_approximation(dem_path: Path, out_path: Path, window: int = 51) -> Path:
    return compute_hand(dem_path, out_path, window)[0]


def compute_layover_shadow_mask(
    slope_path: Path, aspect_path: Path, out_path: Path,
    incidence_angle_deg: float = 39.0, look_azimuth_deg: float = 80.0,
) -> Path:
    """Radar geometry mask: 1 = usable, 0 = layover/shadow, 255 = nodata.
    eps = slope angle measured in the range direction (positive = facing the sensor).
    Layover if eps >= incidence angle; shadow if eps <= -(90 - incidence angle).
    look_azimuth_deg: direction from satellite to ground (ascending right-looking ~80,
    descending ~280).  The previous version compared raw slope with the incidence angle
    and ignored aspect, i.e. it did not model radar geometry at all."""
    with rasterio.open(slope_path) as src:
        slope = src.read(1); nodata = src.nodata if src.nodata is not None else -9999.0
        meta = src.meta.copy(); meta.update(dtype="uint8", nodata=255)
    with rasterio.open(aspect_path) as src:
        asp = src.read(1)
    valid = (slope != nodata) & (asp != nodata)
    facing = (look_azimuth_deg + 180.0) % 360.0       # slope faces the sensor when aspect ~ this
    eps = np.degrees(np.arctan(np.tan(np.radians(np.where(valid, slope, 0)))
                               * np.cos(np.radians(np.where(valid, asp, 0) - facing))))
    bad = valid & ((eps >= incidence_angle_deg) | (eps <= -(90.0 - incidence_angle_deg)))
    out = np.where(valid, np.where(bad, 0, 1), 255).astype(np.uint8)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out, 1)
    logger.info("Layover/shadow: %.1f%% of pixels masked.", 100.0 * bad.sum() / max(valid.sum(), 1))
    return out_path


def resample_to_reference(src_path: Path, ref_path: Path, resampling=Resampling.nearest,
                          fill: float = 0) -> np.ndarray:
    """Resample any raster onto the S1 analysis grid (fixes the old shape mismatch between
    the 30 m terrain mask and the 10 m SAR rasters, which crashed apply_terrain_mask)."""
    with rasterio.open(ref_path) as ref:
        shape_, tr, crs = (ref.height, ref.width), ref.transform, ref.crs
    out = np.full(shape_, fill, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=tr, dst_crs=crs, resampling=resampling,
                  src_nodata=src.nodata, dst_nodata=fill)
    return out


def build_terrain_exclusion_mask(
    slope_path: Path,
    out_path: Path,
    max_slope_deg: float = DEFAULT_MAX_SLOPE_DEG,
    layover_mask_path: Optional[Path] = None,
) -> Path:
    """terrain_mask: 1 = valid, 0 = excluded (too steep or radar layover/shadow).
    HAND is applied SEPARATELY per class (water vs debris have different limits)."""
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        meta = src.meta.copy(); meta.update(dtype="uint8", nodata=255)
    valid = slope != nodata
    ok = valid & (slope < max_slope_deg)
    if layover_mask_path and Path(layover_mask_path).exists():
        with rasterio.open(layover_mask_path) as src:
            ok &= src.read(1) == 1
    out = np.where(valid, np.where(ok, 1, 0), 255).astype(np.uint8)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out, 1)
    logger.info("Terrain mask: %.1f%% pixels usable.", 100.0 * ok.sum() / max(valid.sum(), 1))
    return out_path


def prepare_terrain_layers(
    aoi_geojson: dict,
    run_dir: Path,
    cache_dir: Path,
    metric_crs: str = "EPSG:32645",
    max_slope_deg: float = DEFAULT_MAX_SLOPE_DEG,
    incidence_angle_deg: float = 39.0,
    look_azimuth_deg: float = 80.0,
    use_layover_shadow: bool = True,
) -> dict:
    """Returns dict of Paths (or None): dem, slope, aspect, hand, layover_shadow, terrain_mask
    plus 'hand_method' (str)."""
    res: dict = {"dem": None, "slope": None, "aspect": None, "hand": None,
                 "layover_shadow": None, "terrain_mask": None, "hand_method": None}
    dem_path = run_dir / "dem.tif"
    dem = build_dem_mosaic(aoi_geojson, cache_dir / "dem", dem_path, metric_crs)
    if dem is None:
        return res
    res["dem"] = dem
    slope_path, aspect_path = run_dir / "slope.tif", run_dir / "aspect.tif"
    compute_slope(dem_path, slope_path, aspect_path)
    res["slope"], res["aspect"] = slope_path, aspect_path
    hand_path = run_dir / "hand.tif"
    _, res["hand_method"] = compute_hand(dem_path, hand_path)
    res["hand"] = hand_path
    ls_path = None
    if use_layover_shadow:
        ls_path = run_dir / "layover_shadow_mask.tif"
        compute_layover_shadow_mask(slope_path, aspect_path, ls_path, incidence_angle_deg, look_azimuth_deg)
        res["layover_shadow"] = ls_path
    mask_path = run_dir / "terrain_mask.tif"
    build_terrain_exclusion_mask(slope_path, mask_path, max_slope_deg, ls_path)
    res["terrain_mask"] = mask_path
    return res
