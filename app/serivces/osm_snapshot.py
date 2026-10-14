"""
FloodLens Nepal — OSM pre-event snapshot via the ohsome API
===========================================================
Every request carries an explicit snapshot timestamp STRICTLY BEFORE the event date
(`_validate_snapshot_date`).  This module never talks to Overpass and never uses OSMnx,
which would return TODAY's map (post-event edits => disqualification).

Changes vs the first version
  * `bpolys` (the real AOI polygon) instead of its bounding box.
  * Failed requests are reported (`failed_layers`) instead of silently becoming "0 features".
  * Bridge filter covers bridge=yes|viaduct|... (not only 'yes').
  * Hospitals/clinics only (pharmacies / any healthcare=* are not hospitals).
  * Towns/cities fetched separately as route destinations.
  * Feature ids come from ohsome's `@osmId`.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

import requests
from shapely.geometry import mapping, shape

from app.config import get_config

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5


def _validate_snapshot_date(snapshot_date: date, event_date: date) -> None:
    if snapshot_date >= event_date:
        raise ValueError(f"OSM snapshot date {snapshot_date} must be STRICTLY BEFORE event date "
                         f"{event_date}. Post-event OSM edits are not permitted.")


def _bpolys(aoi_geojson: dict[str, Any]) -> str:
    return json.dumps({"type": "FeatureCollection",
                       "features": [{"type": "Feature", "properties": {"id": "aoi"},
                                     "geometry": mapping(shape(aoi_geojson))}]})


HEADERS = {"User-Agent": "FloodLens-Nepal/0.4 (educational hackathon prototype; python-requests)"}


class OhsomeBlocked(RuntimeError):
    """HTTP 403 from the ohsome front end (often a request-pattern filter, not a missing permission)."""


def _ohsome_post(data: dict) -> Optional[dict]:
    url = f"{get_config()['osm']['ohsome_url']}/elements/geometry"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, data=data, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("ohsome %d (attempt %d/%d): %s", resp.status_code, attempt, MAX_RETRIES, resp.text[:600])
            if resp.status_code == 403:             # retrying the same request will not help; caller tries a bbox
                raise OhsomeBlocked(resp.text[:300])
            if resp.status_code in (400, 404):      # bad filter / date: retrying will not help
                break
        except requests.exceptions.RequestException as exc:
            logger.warning("ohsome error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


def _via_overpass(aoi: dict, snapshot: date, event: date, name: str) -> Optional[dict]:
    from app.services import osm_overpass
    logger.warning("ohsome refuses geometry requests: using the Overpass attic snapshot of %s for %s", snapshot, name)
    res = osm_overpass.fetch_layer(name, aoi, snapshot, event)
    if res is not None:
        _STATE["sources"][name] = f"overpass_attic@{snapshot.isoformat()}"
    return res


def _clip_to_aoi(fc: Optional[dict], aoi: dict) -> Optional[dict]:
    """Keep only features that intersect the exact AOI polygon (used after a bbox request)."""
    if not fc or "features" not in fc:
        return fc
    poly = shape(aoi)
    kept = []
    for ft in fc["features"]:
        try:
            if ft.get("geometry") and shape(ft["geometry"]).intersects(poly):
                kept.append(ft)
        except Exception:  # noqa: BLE001  (malformed geometry: drop it)
            continue
    return {**fc, "features": kept}


_STATE = {"ohsome_geometry_blocked": False, "sources": {}}


def _fetch(aoi: dict, snapshot: date, event: date, osm_filter: str, name: str) -> Optional[dict]:
    _validate_snapshot_date(snapshot, event)
    logger.info("OSM %s @ %s", name, snapshot.isoformat())
    base = {"filter": osm_filter, "time": snapshot.isoformat(), "properties": "tags,metadata"}
    if _STATE["ohsome_geometry_blocked"]:
        return _via_overpass(aoi, snapshot, event, name)
    try:
        res = _ohsome_post({**base, "bpolys": _bpolys(aoi)})
        if res is not None:
            _STATE["sources"][name] = "ohsome"
        return res
    except OhsomeBlocked:
        # Some front ends reject JSON polygons in the request body. A bounding box is a superset of the AOI,
        # so we request that and clip the features back to the exact AOI ourselves.
        logger.warning("ohsome returned 403 for bpolys (%s); retrying with a bounding box and clipping locally", name)
        minx, miny, maxx, maxy = shape(aoi).bounds
        try:
            res = _ohsome_post({**base, "bboxes": f"{minx},{miny},{maxx},{maxy}"})
        except OhsomeBlocked:
            _STATE["ohsome_geometry_blocked"] = True          # do not hammer the same blocked endpoint for every layer
            return _via_overpass(aoi, snapshot, event, name)
        if res is not None:
            _STATE["sources"][name] = "ohsome"
        return _clip_to_aoi(res, aoi)


def fetch_buildings(aoi, snapshot, event):
    return _fetch(aoi, snapshot, event, "building=* and geometry:polygon", "buildings")


def fetch_roads(aoi, snapshot, event, motorable_only: bool = True):
    tags = ",".join(get_config()["osm"]["motorable_highway_tags"])
    flt = f"highway in ({tags}) and geometry:line" if motorable_only else "highway=* and geometry:line"
    return _fetch(aoi, snapshot, event, flt, "roads")


def fetch_bridges(aoi, snapshot, event):
    return _fetch(aoi, snapshot, event, "bridge=* and bridge!=no and type:way and geometry:line", "bridges")


def fetch_settlements(aoi, snapshot, event):
    return _fetch(aoi, snapshot, event, "place in (village,hamlet,suburb,neighbourhood,locality) and geometry:point", "villages")


def fetch_towns(aoi, snapshot, event):
    return _fetch(aoi, snapshot, event, "place in (town,city) and geometry:point", "towns")


def fetch_health_facilities(aoi, snapshot, event):
    types = ",".join(get_config()["accessibility"]["target_health_types"])
    return _fetch(aoi, snapshot, event,
                  f"(amenity in ({types}) or healthcare in (hospital,clinic)) and (geometry:point or geometry:polygon)",
                  "health")


def _expand_aoi(aoi_geojson: dict[str, Any], margin_km: float) -> dict[str, Any]:
    """AOI grown by ~margin_km (degrees approximation) for road/target context only."""
    if not margin_km or margin_km <= 0:
        return aoi_geojson
    return mapping(shape(aoi_geojson).buffer(margin_km / 111.0))


def fetch_all_osm_layers(aoi_geojson: dict[str, Any], event_date: date,
                         snapshot_date: Optional[date] = None,
                         context_margin_km: float = 0.0) -> dict[str, Any]:
    """Buildings, bridges and settlements come from the AOI itself. Roads, towns and health
    facilities come from the AOI grown by `context_margin_km`: otherwise the graph is truncated
    at the AOI edge and settlements whose real escape route leaves the AOI look 'cut off'.
    Flood polygons exist only inside the AOI, so blocking is still evaluated only there."""
    cfg = get_config()["osm"]
    snapshot_date = snapshot_date or event_date - timedelta(days=cfg["snapshot_days_before_event"])
    _validate_snapshot_date(snapshot_date, event_date)

    ctx = _expand_aoi(aoi_geojson, context_margin_km)
    _STATE["ohsome_geometry_blocked"], _STATE["sources"] = False, {}
    layers = {
        "buildings": fetch_buildings(aoi_geojson, snapshot_date, event_date),
        "roads": fetch_roads(ctx, snapshot_date, event_date),
        "bridges": fetch_bridges(aoi_geojson, snapshot_date, event_date),
        "settlements": fetch_settlements(aoi_geojson, snapshot_date, event_date),
        "towns": fetch_towns(ctx, snapshot_date, event_date),
        "health_facilities": fetch_health_facilities(ctx, snapshot_date, event_date),
    }
    failed = [k for k, v in layers.items() if v is None]
    counts = {k: (len(v.get("features", [])) if v else 0) for k, v in layers.items()}
    logger.info("OSM counts: %s | failed: %s", counts, failed)
    from app.services import osm_overpass
    reasons = {k: osm_overpass.LAST_ERROR.get(k2, "unknown") for k in failed
               for k2 in [{"settlements": "villages", "health_facilities": "health"}.get(k, k)]}
    return {"snapshot_date": snapshot_date.isoformat(), **layers, "counts": counts, "failed_layers": failed, "failure_reasons": reasons,
            "sources": dict(_STATE["sources"])}
