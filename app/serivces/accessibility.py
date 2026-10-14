"""
FloodLens Nepal — settlement cut-off analysis (road graph G0 vs G1)
===================================================================
* Motorable roads only (tag filter in configs/config.yaml, applied at fetch time).
* Destinations = hospitals/clinics AND towns/cities (the problem asks for "nearest town or hospital").
  Pharmacies are not destinations.  Villages are the assessed settlements.
* A settlement is snapped to the nearest road node within a tolerance, else `uncertain`.
* Edge blocked if the hazard (buffered by each sensitivity buffer) overlaps it by at least
  min(min_block_length_m, half the segment) — avoids single-pixel noise.
* G0 = pre-event graph, G1_b = G0 minus blocked edges for buffer b.
* Per-settlement class: connected | potentially_cut_off | uncertain, evaluated at EVERY buffer.
  A settlement whose class changes across buffers is `uncertain`; the cut-off count is reported as a range.
* If a settlement cannot reach any destination in G0, that is NEVER called flood isolation.

Performance: one multi-source Dijkstra per graph (not has_path()+shortest_path per
settlement x target as before, and no graph copy per settlement).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, mapping

from app.api.schemas import (
    AccessStatus, BufferSensitivity, FloodConfidence, SettlementAccessResult,
)
from app.config import get_config
from app.services.geo_utils import osm_id, to_metric_gdf

logger = logging.getLogger(__name__)
_to_metric_gdf = to_metric_gdf
_SUPER = "__targets__"


# ── graph construction ───────────────────────────────────────────────────────

def build_road_graph(roads_gdf: gpd.GeoDataFrame):
    """Segment-level undirected graph. Nodes are vertices rounded to 1 m, so OSM ways that share
    an OSM node share a graph node (ways that merely cross do not connect — correct)."""
    G = nx.Graph()
    coord_to_node: dict[tuple[float, float], int] = {}
    edges_meta: list[dict] = []
    counter = 0

    def get_node(pt):
        nonlocal counter
        key = (round(pt[0], 0), round(pt[1], 0))
        if key not in coord_to_node:
            coord_to_node[key] = counter
            G.add_node(counter, x=key[0], y=key[1])
            counter += 1
        return coord_to_node[key]

    edge_id = 0
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if isinstance(geom, LineString) else (list(geom.geoms) if isinstance(geom, MultiLineString) else [])
        way = osm_id(row, f"road_{idx}")
        for line in lines:
            c = list(line.coords)
            for i in range(len(c) - 1):
                u, v = get_node(c[i]), get_node(c[i + 1])
                seg = LineString([c[i], c[i + 1]])
                if u != v and seg.length > 0:
                    G.add_edge(u, v, weight=seg.length, edge_id=edge_id, geom=seg, way=way)
                    edges_meta.append({"edge_id": edge_id, "u": u, "v": v, "weight": seg.length, "geom": seg, "way": way})
                    edge_id += 1
    logger.info("G0: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    return G, coord_to_node, edges_meta


def snap_point_to_graph(point: Point, node_coords: list[tuple[float, float]], node_ids: list[int],
                        max_tolerance_m: float = 1000.0) -> tuple[Optional[int], float]:
    if not node_coords:
        return None, float("inf")
    arr = np.asarray(node_coords)
    d = np.hypot(arr[:, 0] - point.x, arr[:, 1] - point.y)
    i = int(np.argmin(d))
    return (node_ids[i], float(d[i])) if d[i] <= max_tolerance_m else (None, float(d[i]))


# ── pure graph logic (no geometry; unit-tested with networkx only) ────────────

def distance_tree(G: nx.Graph, targets: set[int]) -> tuple[dict, dict]:
    """Distance (m) and predecessor to the NEAREST target for every reachable node."""
    if not targets:
        return {}, {}
    G.add_node(_SUPER)
    try:
        for t in targets:
            G.add_edge(_SUPER, t, weight=0.0, edge_id=-1)
        pred, dist = nx.dijkstra_predecessor_and_distance(G, _SUPER, weight="weight")
    finally:
        G.remove_node(_SUPER)
    dist.pop(_SUPER, None)
    return dist, {n: p[0] for n, p in pred.items() if p and p[0] != _SUPER and n != _SUPER}


def walk_route(pred: dict, node: int) -> list[int]:
    """Nodes from `node` to its nearest target following the predecessor chain."""
    route, seen = [node], {node}
    while node in pred and pred[node] not in seen:
        node = pred[node]
        route.append(node); seen.add(node)
    return route


def classify_settlements(
    G0: nx.Graph, blocked_by_buffer: dict[float, set[int]], snapped: dict[str, tuple[Optional[int], float]],
    targets: set[int], buffers: list[float], high_snap_m: float = 150.0,
) -> dict[str, dict]:
    """Return {sid: {status, by_buffer, reason, d0_km, d1_km, route0, confidence}}."""
    d0, p0 = distance_tree(G0, targets)
    trees: dict[float, tuple[dict, dict]] = {}
    for b in buffers:
        Gb = G0.copy()
        Gb.remove_edges_from([(u, v) for u, v, dd in G0.edges(data=True) if dd.get("edge_id") in blocked_by_buffer[b]])
        trees[b] = distance_tree(Gb, targets)
    primary = buffers[0]
    out: dict[str, dict] = {}
    for sid, (node, snap) in snapped.items():
        rec: dict[str, Any] = {"by_buffer": {}, "reason": None, "d0_km": None, "d1_km": None,
                               "route0": [], "snap": snap}
        if node is None:
            reason = "snapping_failed"
        elif not targets:
            reason = "no_targets"
        elif node not in d0:
            reason = "baseline_unreachable"
        else:
            reason = None
        if reason:
            rec.update(status=AccessStatus.UNCERTAIN, reason=reason, confidence=FloodConfidence.LOW)
            rec["by_buffer"] = {b: AccessStatus.UNCERTAIN for b in buffers}
            out[sid] = rec
            continue
        rec["d0_km"] = d0[node] / 1000.0
        rec["route0"] = walk_route(p0, node)
        for b in buffers:
            rec["by_buffer"][b] = AccessStatus.CONNECTED if node in trees[b][0] else AccessStatus.POTENTIALLY_CUT_OFF
        if node in trees[primary][0]:
            rec["d1_km"] = trees[primary][0][node] / 1000.0
        uniq = set(rec["by_buffer"].values())
        if len(uniq) > 1:
            rec.update(status=AccessStatus.UNCERTAIN, reason="unstable_across_buffers", confidence=FloodConfidence.LOW)
        else:
            rec["status"] = uniq.pop()
            rec["confidence"] = FloodConfidence.HIGH if snap <= high_snap_m else FloodConfidence.MEDIUM
        out[sid] = rec
    return out


# ── geometry-dependent glue ──────────────────────────────────────────────────

def blocked_edges(edges_meta: list[dict], hazard, min_block_length_m: float) -> set[int]:
    import shapely
    if hazard is None or hazard.is_empty or not edges_meta:
        return set()
    geoms = np.array([e["geom"] for e in edges_meta], dtype=object)
    shapely.prepare(hazard)
    hit = shapely.intersects(geoms, hazard)
    blocked: set[int] = set()
    for i in np.flatnonzero(hit):
        seg = geoms[i]
        ln = seg.intersection(hazard).length
        if ln >= min(min_block_length_m, 0.5 * seg.length):
            blocked.add(edges_meta[i]["edge_id"])
    return blocked


def _name(row, fallback: str) -> str:
    for k in ("name:en", "name", "name:ne"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return fallback


def analyze_accessibility(
    roads_data, settlements_data, health_data, flood_polygons_data, run_dir: Path,
    snap_tolerance_m: Optional[float] = None, buffer_m_list: Optional[list[float]] = None,
    metric_crs: str = "EPSG:32645", towns_data=None, target_snap_tolerance_m: Optional[float] = None,
    min_block_length_m: Optional[float] = None,
) -> tuple[list[SettlementAccessResult], dict[str, Any], Path]:
    """Returns (per-settlement results, summary dict, settlement_access.geojson path).
    summary: assessed, cut_off, uncertain, connected, cut_off_min/max, n_targets, buffer_rows."""
    acfg = get_config()["accessibility"]
    snap_tolerance_m = snap_tolerance_m or acfg["settlement_snap_tolerance_m"]
    target_tol = target_snap_tolerance_m or acfg["target_snap_tolerance_m"]
    buffers = [float(b) for b in (buffer_m_list or acfg["buffers_m"])]
    min_block = acfg["min_block_length_m"] if min_block_length_m is None else min_block_length_m
    empty = {"assessed": 0, "cut_off": 0, "uncertain": 0, "connected": 0, "cut_off_min": 0,
             "cut_off_max": 0, "n_targets": 0, "buffer_rows": []}
    access_path = run_dir / "settlement_access.geojson"

    roads = to_metric_gdf(roads_data, metric_crs) if roads_data else gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    sett = to_metric_gdf(settlements_data, metric_crs) if settlements_data else gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    if roads.empty or sett.empty:
        logger.warning("Roads or settlements empty: nothing to assess.")
        access_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return [], empty, access_path
    flood = to_metric_gdf(flood_polygons_data, metric_crs)

    G0, c2n, edges_meta = build_road_graph(roads)
    coords, ids = list(c2n.keys()), list(c2n.values())

    # destinations: hospitals/clinics + towns/cities
    targets: set[int] = set()
    for data in (health_data, towns_data):
        if not data:
            continue
        for _, r in to_metric_gdf(data, metric_crs).iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            nid, _ = snap_point_to_graph(g if g.geom_type == "Point" else g.centroid, coords, ids, target_tol)
            if nid is not None:
                targets.add(nid)
    logger.info("Destination nodes snapped to road graph: %d", len(targets))

    flood_union = flood.union_all() if not flood.empty else None
    blocked = {b: blocked_edges(edges_meta, (None if flood_union is None else (flood_union if b == 0 else flood_union.buffer(b))), min_block)
               for b in buffers}

    snapped, rows_meta = {}, {}
    for idx, r in sett.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        sid = osm_id(r, f"settlement_{idx}")
        pt = g if g.geom_type == "Point" else g.centroid
        snapped[sid] = snap_point_to_graph(pt, coords, ids, snap_tolerance_m)
        rows_meta[sid] = (_name(r, f"Settlement {sid}"), pt)

    cls = classify_settlements(G0, blocked, snapped, targets, buffers)
    edge_by_id = {e["edge_id"]: e for e in edges_meta}

    results: list[SettlementAccessResult] = []
    feats, route_feats = [], []
    per_buf = {b: {"cut_off": 0, "uncertain": 0, "connected": 0} for b in buffers}
    tot = {"cut_off": 0, "uncertain": 0, "connected": 0}
    to4326 = lambda geom: gpd.GeoSeries([geom], crs=metric_crs).to_crs("EPSG:4326").iloc[0]  # noqa: E731
    seen_blocked: set[int] = set()

    for sid, rec in cls.items():
        name, pt = rows_meta[sid]
        st: AccessStatus = rec["status"]
        key = {AccessStatus.CONNECTED: "connected", AccessStatus.POTENTIALLY_CUT_OFF: "cut_off",
               AccessStatus.UNCERTAIN: "uncertain"}[st]
        tot[key] += 1
        for b, bs in rec["by_buffer"].items():
            per_buf[b][{AccessStatus.CONNECTED: "connected", AccessStatus.POTENTIALLY_CUT_OFF: "cut_off",
                        AccessStatus.UNCERTAIN: "uncertain"}[bs]] += 1

        evidence: list[str] = []
        route = rec["route0"]
        if st == AccessStatus.POTENTIALLY_CUT_OFF and len(route) > 1:
            pts = []
            for u, v in zip(route[:-1], route[1:]):
                d = G0.get_edge_data(u, v) or {}
                if d.get("edge_id") in blocked[buffers[0]] or d.get("edge_id") in blocked[buffers[-1]]:
                    if d.get("way") and d["way"] not in evidence:
                        evidence.append(d["way"])
                    eid = d.get("edge_id")
                    if eid not in seen_blocked and eid in edge_by_id:
                        seen_blocked.add(eid)
                        route_feats.append({"type": "Feature", "properties": {"kind": "blocked_edge", "way": d.get("way")},
                                            "geometry": mapping(to4326(edge_by_id[eid]["geom"]))})
                pts.append((G0.nodes[u]["x"], G0.nodes[u]["y"]))
            pts.append((G0.nodes[route[-1]]["x"], G0.nodes[route[-1]]["y"]))
            if len(pts) > 1:
                route_feats.append({"type": "Feature",
                                    "properties": {"kind": "pre_event_route", "settlement_id": sid, "settlement_name": name},
                                    "geometry": mapping(to4326(LineString(pts)))})

        res = SettlementAccessResult(
            settlement_id=sid, settlement_name=name,
            pre_event_reachable_to_hospital=None if rec["reason"] in ("snapping_failed", "no_targets") else rec["d0_km"] is not None,
            hazard_affected_reachable_to_hospital=None if rec["d0_km"] is None else rec["d1_km"] is not None,
            pre_event_distance_km=None if rec["d0_km"] is None else round(rec["d0_km"], 2),
            hazard_affected_distance_km=None if rec["d1_km"] is None else round(rec["d1_km"], 2),
            status=st, confidence=rec["confidence"], hazard_buffer_m=buffers[0],
            evidence_blocked_edge_ids=evidence,
            status_by_buffer={str(int(b)): s.value for b, s in rec["by_buffer"].items()},
            snap_distance_m=None if rec["snap"] == float("inf") else round(rec["snap"], 1),
            reason=rec["reason"],
        )
        results.append(res)
        feats.append({"type": "Feature", "properties": {
            "settlement_id": sid, "name": name, "status": st.value, "confidence": res.confidence.value,
            "pre_distance_km": res.pre_event_distance_km, "post_distance_km": res.hazard_affected_distance_km,
            "status_by_buffer": res.status_by_buffer, "reason": res.reason, "evidence_ways": evidence},
            "geometry": mapping(to4326(pt))})

    cut_counts = [per_buf[b]["cut_off"] for b in buffers]
    summary = {"assessed": len(results), "cut_off": tot["cut_off"], "uncertain": tot["uncertain"],
               "connected": tot["connected"], "cut_off_min": min(cut_counts), "cut_off_max": max(cut_counts),
               "n_targets": len(targets),
               "buffer_rows": [BufferSensitivity(buffer_m=b, **per_buf[b]) for b in buffers]}
    access_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2), encoding="utf-8")
    (run_dir / "route_evidence.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": route_feats}, indent=2), encoding="utf-8")
    logger.info("Access summary: %s", {k: v for k, v in summary.items() if k != "buffer_rows"})
    return results, summary, access_path
