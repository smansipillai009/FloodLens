"""
FloodLens Nepal — infrastructure impact (OSM x flood/debris candidates)
=======================================================================
* Metric CRS for all geometry work (caller passes the AOI's UTM zone).
* Building "potentially affected" if overlap >= configured share.
* Road/bridge "potentially affected" if the geometry intersects the candidate mask
  (roads: >= min overlap length so single-pixel noise does not count).
* Sensitivity: the same counts at every hazard buffer (default 0/30/60 m); the headline number is
  buffer 0 and the range is reported — the first version only did this for roads.
* Feature-level evidence (OSM id, overlap, confidence class) is written to the GeoJSON files.
A flood/debris mask cannot confirm structural damage: wording stays "potentially affected".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd

from app.api.schemas import InfrastructureImpact
from app.config import get_config
from app.services.geo_utils import osm_id, to_metric_gdf

logger = logging.getLogger(__name__)
_to_metric_gdf = to_metric_gdf   # backwards-compatible alias


def _conf(overlap_pct: float) -> str:
    return "high" if overlap_pct >= 50 else "medium" if overlap_pct >= 25 else "low"


def analyze_infrastructure_impact(
    flood_polygons: dict[str, Any] | Path,
    osm_buildings: Optional[dict[str, Any] | Path],
    osm_roads: Optional[dict[str, Any] | Path],
    osm_bridges: Optional[dict[str, Any] | Path],
    run_dir: Path,
    building_overlap_pct: Optional[float] = None,
    buffer_m_list: Optional[list[float]] = None,
    metric_crs: str = "EPSG:32645",
    min_road_overlap_m: Optional[float] = None,
) -> tuple[InfrastructureImpact, dict[str, Any]]:
    cfg = get_config()
    building_overlap_pct = cfg["osm"]["building_overlap_threshold_pct"] if building_overlap_pct is None else building_overlap_pct
    min_road_overlap_m = cfg["osm"]["min_road_overlap_m"] if min_road_overlap_m is None else min_road_overlap_m
    buffer_m_list = buffer_m_list or cfg["accessibility"]["buffers_m"]

    flood_gdf = to_metric_gdf(flood_polygons, metric_crs)
    impact = InfrastructureImpact()
    details: dict[str, Any] = {"buffers": [], "impacted_road_ids": [], "impacted_bridge_ids": []}
    if flood_gdf.empty:
        logger.warning("No flood polygons: reporting 0 impacted features.")
        details["buffers"] = [{"buffer_m": float(b), "buildings": 0, "road_km": 0.0, "bridges": 0} for b in buffer_m_list]
        return impact, details

    flood_union = flood_gdf.union_all()
    zones = {float(b): (flood_union if b == 0 else flood_union.buffer(b)) for b in buffer_m_list}
    rows = {b: {"buffer_m": b, "buildings": 0, "road_km": 0.0, "bridges": 0} for b in zones}

    # ── buildings ────────────────────────────────────────────────────────────
    if osm_buildings:
        bld = to_metric_gdf(osm_buildings, metric_crs)
        if not bld.empty:
            bld = bld[bld.geometry.area > 0].copy()
            area = bld.geometry.area
            for b, zone in zones.items():
                pct = bld.geometry.intersection(zone).area / area * 100.0
                rows[b]["buildings"] = int((pct >= building_overlap_pct).sum())
            bld["overlap_pct"] = (bld.geometry.intersection(flood_union).area / area * 100.0).round(1)
            bld["osm_id"] = [osm_id(r, f"bld_{i}") for i, r in bld.iterrows()]
            bld["confidence"] = bld["overlap_pct"].map(_conf)
            flagged = bld[bld["overlap_pct"] >= building_overlap_pct]
            impact.potentially_affected_buildings = len(flagged)
            if not flagged.empty:
                p = run_dir / "impacted_buildings.geojson"
                flagged.to_crs("EPSG:4326").to_file(p, driver="GeoJSON")
                impact.impacted_buildings_file = p.name

    # ── roads ────────────────────────────────────────────────────────────────
    if osm_roads:
        road = to_metric_gdf(osm_roads, metric_crs)
        if not road.empty:
            for b, zone in zones.items():
                rl = road.geometry.intersection(zone).length
                rows[b]["road_km"] = round(float(rl[rl >= min_road_overlap_m].sum()) / 1000.0, 3)
            road["cut_length_m"] = road.geometry.intersection(flood_union).length.round(1)
            flagged = road[road["cut_length_m"] >= min_road_overlap_m].copy()
            flagged["osm_id"] = [osm_id(r, f"road_{i}") for i, r in flagged.iterrows()]
            impact.potentially_affected_road_km = round(float(flagged["cut_length_m"].sum()) / 1000.0, 3)
            details["impacted_road_ids"] = flagged["osm_id"].tolist()
            if not flagged.empty:
                p = run_dir / "impacted_roads.geojson"
                flagged.to_crs("EPSG:4326").to_file(p, driver="GeoJSON")
                impact.impacted_roads_file = p.name

    # ── bridges ──────────────────────────────────────────────────────────────
    if osm_bridges:
        br = to_metric_gdf(osm_bridges, metric_crs)
        if not br.empty:
            for b, zone in zones.items():
                rows[b]["bridges"] = int(br.geometry.intersects(zone).sum())
            hit = br[br.geometry.intersects(flood_union)].copy()
            hit["osm_id"] = [osm_id(r, f"br_{i}") for i, r in hit.iterrows()]
            hit["note"] = "intersects mask; structural damage NOT confirmed (requires field verification)"
            impact.potentially_affected_bridges = len(hit)
            details["impacted_bridge_ids"] = hit["osm_id"].tolist()
            if not hit.empty:
                p = run_dir / "impacted_bridges.geojson"
                hit.to_crs("EPSG:4326").to_file(p, driver="GeoJSON")
                impact.impacted_bridges_file = p.name

    details["buffers"] = [rows[b] for b in sorted(rows)]
    logger.info("Impact: %d buildings, %.2f km road, %d bridges (buffer 0)",
                impact.potentially_affected_buildings, impact.potentially_affected_road_km,
                impact.potentially_affected_bridges)
    return impact, details
