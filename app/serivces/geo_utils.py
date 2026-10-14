"""Shared geometry helpers (GeoJSON -> metric GeoDataFrame, OSM ids)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


def to_metric_gdf(geojson_data: dict[str, Any] | Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON dict/path into `target_crs`.
    dict input: honours a legacy `crs` member or `crs_epsg`, else assumes EPSG:4326.
    path input: standard GeoJSON => EPSG:4326 (all FloodLens files are written in 4326)."""
    crs_spec = "EPSG:4326"
    if isinstance(geojson_data, Path):
        gdf = gpd.read_file(geojson_data)
    else:
        crs_prop = geojson_data.get("crs", {})
        if isinstance(crs_prop, dict) and "properties" in crs_prop:
            crs_spec = crs_prop["properties"].get("name", crs_spec)
        elif "crs_epsg" in geojson_data:
            crs_spec = f"EPSG:{geojson_data['crs_epsg']}"
        feats = geojson_data.get("features", [])
        gdf = gpd.GeoDataFrame.from_features(feats, crs=crs_spec)
        # keep feature-level "id" (used by ohsome/OSM exports)
        if feats and any("id" in f for f in feats):
            gdf["feature_id"] = [f.get("id") for f in feats]
    if gdf.empty or gdf.geometry.isnull().all():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_spec)
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def osm_id(row, fallback: str) -> str:
    """ohsome puts the id in `@osmId` ('way/123'); older exports use `id`."""
    for k in ("@osmId", "feature_id", "id", "@id"):
        v = row.get(k) if hasattr(row, "get") else None
        if v is not None and str(v) not in ("nan", "None", ""):
            return str(v)
    return fallback
