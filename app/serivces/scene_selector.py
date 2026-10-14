"""
FloodLens Nepal — Sentinel-1 / Sentinel-2 scene selector (CDSE STAC)
====================================================================
Rules (from the execution plan):
  * S1 before/after MUST share relative orbit AND direction (same track).
  * Never silently fall back to a mismatched pair.  If no same-track pair exists
    the result has s1_before = s1_after = None, same_track = False, an exact
    reason, and the candidate scenes so the UI can show them.
  * Extra same-track pre-event scenes are returned for a median reference
    (cuts speckle / false positives).
  * OSM snapshot date is strictly before the event AND not later than the newest
    ohsome extract date.

NOTE: STAC collection ids and property names are configurable and MUST be
verified on Day 1 (`python run.py search ...` logs what was found).
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests
from shapely.geometry import shape

from app.api.schemas import (
    OrbitDirection, S1SceneInfo, S2SceneInfo, SceneSelectionResult,
)
from app.config import get_config

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 30
MAX_RESULTS = 100
METADATA_RETRIES = 3
METADATA_RETRY_DELAY_S = 2.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _aoi_bbox(aoi_geojson: dict[str, Any]) -> list[float]:
    return list(shape(aoi_geojson).bounds)


def _to_iso(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_orbit_direction(props: dict) -> Optional[OrbitDirection]:
    raw = str(props.get("sat:orbit_state") or props.get("orbitDirection") or "").upper()
    if "DESC" in raw:
        return OrbitDirection.DESCENDING
    if "ASC" in raw:
        return OrbitDirection.ASCENDING
    return None  # unknown -> scene is rejected (we never guess a direction)


def _search_stac(collection: str, bbox: list[float], start: date, end: date,
                 limit: int = MAX_RESULTS) -> list[dict]:
    cfg = get_config()["scene_selection"]
    payload = {
        "collections": [collection], "bbox": bbox, "limit": limit,
        "datetime": f"{_to_iso(start)}/{_to_iso(end + timedelta(days=1))}",   # `end` day is INCLUSIVE
    }
    try:
        r = requests.post(f"{cfg['stac_url']}/search", json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json().get("features", [])
    except requests.exceptions.RequestException as exc:
        logger.error("STAC search failed (%s): %s", collection, exc)
        return []


def _parse_s1_item(item: dict, cfg: dict) -> Optional[S1SceneInfo]:
    props = item.get("properties", {})
    try:
        mode = props.get("sar:instrument_mode")
        if mode and str(mode).upper() != cfg["instrument_mode"]:
            return None
        ptype = str(props.get("sar:product_type") or props.get("product:type") or props.get("productType") or "GRD")
        if "GRD" not in ptype.upper():
            return None
        pols = props.get("sar:polarizations")
        if pols:
            if not set(cfg["required_polarizations"]).issubset({str(p).upper() for p in pols}):
                return None
        else:
            pols = list(cfg["required_polarizations"])  # property absent: cannot verify
        direction = _parse_orbit_direction(props)
        rel = props.get("sat:relative_orbit", props.get("relativeOrbitNumber"))
        if direction is None or rel is None:
            logger.warning("S1 item %s lacks orbit info; rejected.", item.get("id"))
            return None
        dt = _parse_dt(props.get("datetime") or props.get("start_datetime"))
        return S1SceneInfo(
            scene_id=item.get("id", "unknown"), datetime=dt, relative_orbit=int(rel),
            orbit_direction=direction, polarizations=[str(p) for p in pols],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse S1 item %s: %s", item.get("id"), exc)
        return None


def _parse_s2_item(item: dict) -> Optional[S2SceneInfo]:
    props = item.get("properties", {})
    try:
        cc = props.get("eo:cloud_cover", props.get("cloudCover"))
        # BUGFIX: the old `x or 100.0` turned a genuine 0 % into 100 %.
        cloud = 100.0 if cc is None else float(cc)
        return S2SceneInfo(
            scene_id=item.get("id", "unknown"),
            datetime=_parse_dt(props.get("datetime") or props.get("start_datetime")),
            cloud_cover_pct=cloud,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse S2 item %s: %s", item.get("id"), exc)
        return None


def _dedupe(scenes: list[S1SceneInfo]) -> list[S1SceneInfo]:
    seen, out = set(), []
    for s in sorted(scenes, key=lambda x: x.datetime):
        if s.scene_id not in seen:
            seen.add(s.scene_id)
            out.append(s)
    return out


# ── pair selection ────────────────────────────────────────────────────────────

def find_best_s1_pair(
    before: list[S1SceneInfo], after: list[S1SceneInfo], event_date: date,
    max_extra: int = 3,
) -> tuple[Optional[S1SceneInfo], Optional[S1SceneInfo], list[S1SceneInfo], list[str]]:
    """Closest same-track (orbit + direction) pair straddling the event.
    Returns (before, after, extra_same_track_before_scenes, warnings).
    NEVER returns a mismatched pair."""
    warnings: list[str] = []
    if not before:
        return None, None, [], ["No Sentinel-1 GRD IW VV+VH scenes found in the pre-event window."]
    if not after:
        return None, None, [], ["No Sentinel-1 GRD IW VV+VH scenes found in the post-event window."]

    key = lambda s: (s.relative_orbit, s.orbit_direction.value)  # noqa: E731
    best, best_gap = None, 10**9
    for b in before:
        if b.datetime.date() >= event_date:
            continue
        for a in after:
            if a.datetime.date() < event_date or key(a) != key(b):
                continue
            gap = (event_date - b.datetime.date()).days + (a.datetime.date() - event_date).days
            if gap < best_gap:
                best, best_gap = (b, a), gap
    if best is None:
        tracks_b = sorted({key(s) for s in before})
        tracks_a = sorted({key(s) for s in after})
        warnings.append(
            "NO same-track Sentinel-1 pair straddles the event. Pre-event tracks "
            f"(orbit, direction): {tracks_b}; post-event tracks: {tracks_a}. "
            "Pixel-wise change mapping is disabled; widen the window or choose another date."
        )
        return None, None, [], warnings
    b, a = best
    if a.datetime.date() == event_date:
        warnings.append(f"Selected post-event scene {a.scene_id} was acquired on the event date ({event_date}); "
                        "check its UTC time against the event time: it may pre-date the flood.")
    extras = sorted(
        [s for s in before if key(s) == key(b) and s.scene_id != b.scene_id and s.datetime < b.datetime],
        key=lambda s: s.datetime, reverse=True)[:max_extra]
    return b, a, extras, warnings


def latest_ohsome_date() -> Optional[date]:
    """Newest timestamp ohsome can serve (the case study says 27 Jul 2026). Retries transient 5xx/network errors."""
    url = get_config()["osm"]["ohsome_url"] + "/metadata"
    for attempt in range(1, METADATA_RETRIES + 1):
        try:
            r = requests.get(url, timeout=DEFAULT_TIMEOUT)
            if r.status_code >= 500:
                raise requests.exceptions.RequestException(f"{r.status_code} {r.reason}")
            r.raise_for_status()
            ts = r.json()["extractRegion"]["temporalExtent"]["toTimestamp"]
            return _parse_dt(ts).date()
        except requests.exceptions.RequestException as exc:
            logger.warning("ohsome metadata attempt %d/%d failed: %s", attempt, METADATA_RETRIES, exc)
            if attempt < METADATA_RETRIES:
                time.sleep(METADATA_RETRY_DELAY_S * attempt)
        except Exception as exc:  # noqa: BLE001  (unexpected JSON shape)
            logger.warning("Could not parse ohsome metadata: %s", exc)
            return None
    return None


def choose_osm_snapshot_date(event_date: date) -> tuple[date, list[str]]:
    cfg = get_config()["osm"]
    snap = event_date - timedelta(days=cfg["snapshot_days_before_event"])
    warns: list[str] = []
    latest = latest_ohsome_date()
    if latest is None and cfg.get("fallback_latest_date"):
        latest = date.fromisoformat(cfg["fallback_latest_date"])
        warns.append(f"ohsome /metadata unreachable: assuming the latest OSM data is {latest} "
                     "(value from the problem statement, not verified live).")
    if latest and snap > latest:
        warns.append(f"Requested OSM snapshot {snap} is newer than the latest ohsome data "
                     f"({latest}); using {latest}.")
        snap = latest
    assert snap < event_date, "OSM snapshot must be strictly before the event date"
    return snap, warns


# ── public API ────────────────────────────────────────────────────────────────

def select_scenes(
    aoi_geojson: dict[str, Any], event_date: date,
    before_window_days: int = 18, after_window_days: int = 18,
) -> SceneSelectionResult:
    cfg = get_config()["scene_selection"]
    bbox = _aoi_bbox(aoi_geojson)

    raw_b = _search_stac(cfg["s1_collection"], bbox,
                         event_date - timedelta(days=before_window_days),
                         event_date - timedelta(days=1))
    raw_a = _search_stac(cfg["s1_collection"], bbox, event_date,
                         event_date + timedelta(days=after_window_days))
    before = _dedupe([s for i in raw_b if (s := _parse_s1_item(i, cfg))])
    after = _dedupe([s for i in raw_a if (s := _parse_s1_item(i, cfg))])
    logger.info("S1 candidates: %d before, %d after.", len(before), len(after))

    b, a, extras, warnings = find_best_s1_pair(
        before, after, event_date, cfg["max_pre_event_reference_scenes"])

    # S2: closest-in-time among low-cloud scenes (cloud is re-computed in the AOI later)
    best_s2 = None
    raw_s2 = _search_stac(cfg["s2_collection"], bbox, event_date,
                          event_date + timedelta(days=after_window_days))
    s2 = [s for i in raw_s2 if (s := _parse_s2_item(i))]
    ok = [s for s in s2 if s.cloud_cover_pct <= cfg["s2_max_cloud_pct_scene"]]
    if ok:
        best_s2 = min(ok, key=lambda s: (abs((s.datetime.date() - event_date).days), s.cloud_cover_pct))
    elif s2:
        best_s2 = min(s2, key=lambda s: s.cloud_cover_pct)
        warnings.append(f"Best S2 scene has {best_s2.cloud_cover_pct:.0f}% scene cloud; "
                        "optical confirmation will be limited.")
    else:
        warnings.append("No Sentinel-2 L2A scene in the post-event window; SAR-only result.")

    snap, snap_warn = choose_osm_snapshot_date(event_date)
    warnings += snap_warn

    result = SceneSelectionResult(
        s1_before=b, s1_after=a, s1_before_extra=extras, s2_after=best_s2,
        osm_snapshot_date=snap, warnings=warnings,
        candidates_before=before if b is None else [],
        candidates_after=after if a is None else [],
    )
    if b is None:
        result.same_track = False
    return result


def print_scene_summary(result: SceneSelectionResult) -> None:
    print("\n" + "=" * 60 + "\nSCENE SELECTION RESULT\n" + "=" * 60)
    for lbl, s in (("S1 BEFORE", result.s1_before), ("S1 AFTER ", result.s1_after)):
        if s:
            print(f"  {lbl}: {s.scene_id}\n             {s.datetime.date()} | orbit "
                  f"{s.relative_orbit} | {s.orbit_direction.value}")
        else:
            print(f"  {lbl}: NOT FOUND")
    print(f"  EXTRA PRE-EVENT REFERENCE SCENES: {len(result.s1_before_extra)}")
    print(f"  SAME TRACK: {'YES' if result.same_track else 'NO — pixel mapping disabled'}")
    if result.s2_after:
        print(f"  S2 AFTER  : {result.s2_after.scene_id} | scene cloud {result.s2_after.cloud_cover_pct:.1f}%")
    print(f"  OSM DATE  : {result.osm_snapshot_date}")
    for w in result.warnings:
        print(f"    ! {w}")
    print("=" * 60 + "\n")
