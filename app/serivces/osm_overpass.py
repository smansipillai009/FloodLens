"""
Pre-event OSM snapshot through the Overpass *attic* date feature: FALLBACK when the ohsome API refuses geometry requests (HTTP 403).

Compliance (data rules: "OpenStreetMap as it was before the event"):
  * EVERY query starts with  [date:"<snapshot>T00:00:00Z"]  and the snapshot must be STRICTLY BEFORE the event date. Without that header Overpass returns
    TODAY's map (post-event edits => disqualification), so build_query() refuses to produce a query without it and a test enforces it.
  * This is the only module allowed to talk to Overpass (tests/test_compliance.py).
Output has the same shape as ohsome results: a GeoJSON FeatureCollection whose properties are the OSM tags plus '@osmId' ("way/123").

v2 (queue-aware + cached client) - what changed and why
  * HTTP 504 / 429 from the public server means "too busy", not "bad query".  Before each request we read the server's /status page and wait for a
    free slot; failures are retried with exponential backoff + jitter (honouring Retry-After).  Requests are also paced (MIN_INTERVAL_S).
  * Heavy layers are split into small tiles on a fixed global lattice (default 0.1 deg ~ 11 km).  Small queries are far more likely to be scheduled,
    and a tile that answers "out of memory / timed out" is split again (up to MAX_SPLIT_DEPTH).
  * Every successful tile response is cached on disk, keyed by the exact query text (which contains the attic date + bounds).  A dated historical query
    never changes, so the cache never goes stale.  A re-run only fetches the tiles that are still missing, so a flaky server can no longer lose work.
  * A layer is returned only if ALL its tiles succeeded.  A partial road network would create false "cut-off" results, so an incomplete layer is
    reported as failed (None + LAST_ERROR) and the good tiles stay cached for the next try.
  * Empty answers for roads/buildings are never cached (an overloaded server can answer 200 with nothing); a layer with 0 roads logs a warning.
Limits: relations (e.g. multipolygon buildings) are skipped and counted; coverage of the attic database should be checked on the first live run.
If you add extra URLs to osm.overpass_urls, each MUST support attic ([date:...]) queries - a server that ignores the header returns today's map.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape

from app.config import get_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "FloodLens-Nepal/0.4 (educational hackathon prototype; python-requests)"}
QUERY_TIMEOUT_S = 90          # server-side limit written into every query (small queries are scheduled more readily)
HTTP_TIMEOUT_S = 120
MIN_INTERVAL_S = 2.0          # minimum gap between two network requests (the public servers rate-limit bursts)
STATUS_TIMEOUT_S = 10
MAX_SLOT_WAIT_S = 120         # longest total wait for a free slot before we just try anyway
MAX_SPLIT_DEPTH = 2
DEFAULT_TILE_DEG = 0.1
LAST_ERROR: dict[str, str] = {}
LAST_INFO: dict[str, dict] = {}
LAST_WARNINGS: dict[str, list[str]] = {}
STATS = {"cache_hits": 0, "network_requests": 0, "retries": 0}

# layer name used by osm_snapshot._fetch  ->  kind
POINT_KINDS = {"settlements", "towns", "health"}
KIND_BY_NAME = {"buildings": "buildings", "roads": "roads", "bridges": "bridges",
                "villages": "settlements", "towns": "towns", "health": "health"}
SINGLE_QUERY_KINDS = {"settlements", "towns"}                 # node-only queries are cheap: one request, no tiles
EMPTY_IS_NORMAL = {"bridges", "settlements", "towns", "health"}  # an empty answer for these may be cached

# hooks the tests replace
_SESSION: Any = None
_sleep = time.sleep
_now = time.monotonic
_last_request_at = [0.0]


def _http():
    return _SESSION if _SESSION is not None else requests


def _opt(key: str, default: Any) -> Any:
    try:
        v = get_config()["osm"].get(key, default)
        return default if v is None else v
    except Exception:  # noqa: BLE001
        return default


# ── query building (unchanged contract) ──────────────────────────────────────

def _rx(values) -> str:
    return "^(" + "|".join(values) + ")$"


def _body(kind: str, bb: str) -> str:
    cfg = get_config()
    if kind == "buildings":
        return f'way["building"]({bb});'
    if kind == "roads":
        return f'way["highway"~"{_rx(cfg["osm"]["motorable_highway_tags"])}"]({bb});'
    if kind == "bridges":
        return f'way["bridge"]["bridge"!="no"]({bb});'
    if kind == "settlements":
        return f'node["place"~"{_rx(["village", "hamlet", "suburb", "neighbourhood", "locality"])}"]({bb});'
    if kind == "towns":
        return f'node["place"~"{_rx(["town", "city"])}"]({bb});'
    if kind == "health":
        types = _rx(cfg["accessibility"]["target_health_types"])
        return (f'node["amenity"~"{types}"]({bb}); way["amenity"~"{types}"]({bb});'
                f'node["healthcare"~"{_rx(["hospital", "clinic"])}"]({bb}); way["healthcare"~"{_rx(["hospital", "clinic"])}"]({bb});')
    raise ValueError(f"unknown OSM layer kind {kind!r}")


def build_query(kind: str, bounds: tuple[float, float, float, float], snapshot: date, event: date) -> str:
    """Overpass QL with an explicit attic date. Refuses snapshot >= event."""
    if snapshot >= event:
        raise ValueError(f"OSM snapshot {snapshot} must be STRICTLY BEFORE event date {event}. Post-event OSM edits are not permitted.")
    minx, miny, maxx, maxy = bounds
    bb = f"{miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f}"        # Overpass order: south,west,north,east
    q = f'[out:json][timeout:{QUERY_TIMEOUT_S}][date:"{snapshot.isoformat()}T00:00:00Z"];({_body(kind, bb)});out geom;'
    assert '[date:"' in q                                       # belt and braces: never send an undated query
    return q


def _to_feature(el: dict, kind: str) -> Optional[dict]:
    tags = dict(el.get("tags") or {})
    props = {**tags, "@osmId": f'{el["type"]}/{el["id"]}'}
    if el["type"] == "node" and kind not in POINT_KINDS:
        return None
    if el["type"] == "way" and kind in ("settlements", "towns"):
        return None
    if el["type"] == "node":
        if "lat" not in el:
            return None
        geom = mapping(Point(el["lon"], el["lat"]))
    elif el["type"] == "way":
        pts = [(g["lon"], g["lat"]) for g in el.get("geometry", []) if g]
        if len(pts) < 2:
            return None
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        if kind in ("buildings", "health") and closed:
            geom = mapping(Polygon(pts))
        elif kind == "buildings":
            return None                                           # an unclosed building way is not a polygon
        else:
            geom = mapping(LineString(pts))
    else:
        return None
    return {"type": "Feature", "properties": props, "geometry": geom}


def to_geojson(overpass_json: dict, kind: str, aoi: dict) -> dict:
    poly = shape(aoi)
    feats, skipped_rel = [], 0
    for el in overpass_json.get("elements", []):
        if el.get("type") == "relation":
            skipped_rel += 1
            continue
        ft = _to_feature(el, kind)
        if ft and shape(ft["geometry"]).intersects(poly):
            feats.append(ft)
    if skipped_rel:
        logger.info("Overpass %s: %d relations skipped (multipolygons not assembled)", kind, skipped_rel)
    return {"type": "FeatureCollection", "features": feats}


# ── disk cache (keyed by the exact dated query) ──────────────────────────────

def _cache_dir() -> Path:
    p = Path(str(_opt("overpass_cache_dir", "data/cache/overpass")))
    return p if p.is_absolute() else ROOT / p


def _cache_file(query: str) -> Path:
    return _cache_dir() / (hashlib.sha256(query.encode("utf-8")).hexdigest() + ".json")


def _cache_get(query: str) -> Optional[list]:
    assert '[date:"' in query                                   # the cache can never serve an undated query
    f = _cache_file(query)
    if not f.exists():
        return None
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("v") == 1 and rec.get("query") == query and isinstance(rec.get("elements"), list):
            return rec["elements"]
    except (OSError, ValueError):
        pass
    return None


def _cache_put(query: str, elements: list) -> None:
    assert '[date:"' in query
    try:
        d = _cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        f = _cache_file(query)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"v": 1, "query": query, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "n_elements": len(elements), "elements": elements}), encoding="utf-8")
        os.replace(tmp, f)                                      # atomic: an interrupted run never leaves a half-written entry
    except OSError as exc:
        logger.warning("Overpass cache not written (%s)", exc)


# ── queue awareness ──────────────────────────────────────────────────────────

def parse_status(text: str) -> tuple[bool, int]:
    """Overpass /api/status text -> (slot_free_now, seconds_to_wait). Unknown format => do not block."""
    low = text.lower()
    if re.search(r"rate limit:\s*0\b", low):
        return True, 0                                          # no rate limit on this server
    m = re.search(r"(\d+)\s+slots?\s+available\s+now", low)
    if m and int(m.group(1)) > 0:
        return True, 0
    waits = [int(x) for x in re.findall(r"in\s+(\d+)\s+seconds", low)]
    if waits:
        return False, min(waits)
    return True, 0


def wait_for_slot(url: str, max_wait_s: float = MAX_SLOT_WAIT_S) -> float:
    """Poll /status until a query slot is free. Returns seconds waited. Never raises (status being down must not stop us)."""
    status_url = url.rsplit("/", 1)[0] + "/status"
    waited = 0.0
    while waited < max_wait_s:
        try:
            r = _http().get(status_url, headers=HEADERS, timeout=STATUS_TIMEOUT_S)
            if getattr(r, "status_code", 200) != 200:
                return waited
            free, wait = parse_status(r.text)
        except (requests.exceptions.RequestException, ValueError, AttributeError):
            return waited
        if free:
            return waited
        pause = min(max(wait, 1) + 1, 30)
        logger.info("Overpass busy: all slots in use, waiting %ds", pause)
        _sleep(pause)
        waited += pause
    return waited


def _pace() -> None:
    gap = MIN_INTERVAL_S - (_now() - _last_request_at[0])
    if gap > 0:
        _sleep(gap)
    _last_request_at[0] = _now()


def _backoff(attempt: int, retry_after: Any = None) -> None:
    base, cap = float(_opt("overpass_backoff_base_s", 15)), float(_opt("overpass_backoff_cap_s", 120))
    delay = min(cap, base * (2 ** (attempt - 1))) * (0.75 + 0.5 * random.random())
    try:
        delay = max(delay, min(float(retry_after), cap))
    except (TypeError, ValueError):
        pass
    STATS["retries"] += 1
    logger.info("Overpass retry in %.0fs", delay)
    _sleep(delay)


def _post_with_policy(query: str, ctx: dict) -> tuple[str, Any]:
    """('ok', json) | ('heavy', reason) | ('fail', reason)."""
    urls = list(_opt("overpass_urls", [DEFAULT_URL])) or [DEFAULT_URL]
    max_attempts = int(_opt("overpass_max_attempts", 6))
    last = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        if _now() > ctx["deadline"]:
            return "fail", f"layer time budget used up (last: {last})"
        url = urls[(attempt - 1) % len(urls)]
        wait_for_slot(url)
        _pace()
        STATS["network_requests"] += 1
        try:
            r = _http().post(url, data={"data": query}, headers=HEADERS, timeout=HTTP_TIMEOUT_S)
        except requests.exceptions.RequestException as exc:
            last = type(exc).__name__
            logger.warning("Overpass request failed (attempt %d/%d): %s", attempt, max_attempts, last)
            if attempt < max_attempts:
                _backoff(attempt)
            continue
        code = r.status_code
        if code == 200:
            try:
                js = r.json()
            except ValueError:
                last = "HTTP 200 but not JSON"
                if attempt < max_attempts:
                    _backoff(attempt)
                continue
            remark = str(js.get("remark", "")).lower()
            if "runtime error" in remark:
                if "out of memory" in remark or "timed out" in remark:
                    return "heavy", remark[:120]
                last = remark[:120]
                if attempt < max_attempts:
                    _backoff(attempt)
                continue
            return "ok", js
        if code in (400, 403, 404):
            return "fail", f"HTTP {code} (not retried)"
        last = f"HTTP {code}"
        logger.warning("Overpass HTTP %d (attempt %d/%d, %s) - server busy, backing off", code, attempt, max_attempts, url)
        if attempt < max_attempts:
            _backoff(attempt, getattr(r, "headers", {}).get("Retry-After") if hasattr(r, "headers") else None)
    return "fail", f"{last} after {max_attempts} attempts"


# ── tiles ────────────────────────────────────────────────────────────────────

def plan_tiles(aoi: dict, deg: float) -> list[tuple[float, float, float, float]]:
    """Tiles on a fixed global lattice (so overlapping AOIs share cached tiles) that intersect the AOI polygon."""
    poly = shape(aoi)
    minx, miny, maxx, maxy = poly.bounds
    i0, i1 = math.floor(minx / deg + 1e-9), math.floor(maxx / deg - 1e-9)
    j0, j1 = math.floor(miny / deg + 1e-9), math.floor(maxy / deg - 1e-9)
    tiles = []
    for j in range(j0, max(j0, j1) + 1):
        for i in range(i0, max(i0, i1) + 1):
            b = (round(i * deg, 6), round(j * deg, 6), round((i + 1) * deg, 6), round((j + 1) * deg, 6))
            if poly.intersects(box(*b)):
                tiles.append(b)
    return tiles or [tuple(poly.bounds)]


def _split4(b: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = b
    mx, my = round((x0 + x1) / 2, 6), round((y0 + y1) / 2, 6)
    return [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]


def _fetch_tile(kind: str, bounds, snapshot: date, event: date, ctx: dict, depth: int = 0) -> tuple[Optional[list], str]:
    q = build_query(kind, bounds, snapshot, event)              # always validates the pre-event date, even for cache hits
    cached = _cache_get(q)
    if cached is not None:
        STATS["cache_hits"] += 1
        return cached, ""
    status, payload = _post_with_policy(q, ctx)
    if status == "ok":
        els = payload.get("elements", [])
        if els or kind in EMPTY_IS_NORMAL:
            _cache_put(q, els)
        return els, ""
    if status == "heavy" and depth < MAX_SPLIT_DEPTH:
        logger.info("Overpass %s tile too heavy (%s): splitting into 4", kind, payload)
        merged: dict[tuple, dict] = {}
        for sub in _split4(bounds):
            els, why = _fetch_tile(kind, sub, snapshot, event, ctx, depth + 1)
            if els is None:
                return None, why
            for e in els:
                merged[(e.get("type"), e.get("id"))] = e
        return list(merged.values()), ""
    return None, str(payload)


# ── public entry point ───────────────────────────────────────────────────────

def fetch_layer(name: str, aoi: dict[str, Any], snapshot: date, event: date) -> Optional[dict]:
    """One OSM layer as GeoJSON, or None if it could not be completely retrieved (reason in LAST_ERROR[name])."""
    kind = KIND_BY_NAME[name]
    LAST_ERROR.pop(name, None)
    LAST_WARNINGS.pop(name, None)
    poly = shape(aoi)
    deg = float(_opt("overpass_tile_deg", DEFAULT_TILE_DEG))
    tiles = [tuple(poly.bounds)] if kind in SINGLE_QUERY_KINDS else plan_tiles(aoi, deg)
    ctx = {"deadline": _now() + float(_opt("overpass_layer_deadline_s", 900))}
    hits0, net0 = STATS["cache_hits"], STATS["network_requests"]
    merged: dict[tuple, dict] = {}
    failed: list[str] = []
    for tb in tiles:
        els, why = _fetch_tile(kind, tb, snapshot, event, ctx)
        if els is None:
            failed.append(why)
            continue
        for e in els:
            merged[(e.get("type"), e.get("id"))] = e
    info = {"tiles": len(tiles), "cache_hits": STATS["cache_hits"] - hits0,
            "network_requests": STATS["network_requests"] - net0, "elements": len(merged)}
    LAST_INFO[name] = info
    if failed:
        LAST_ERROR[name] = f"{len(failed)} of {len(tiles)} tiles failed: {failed[0]}"
        logger.error("Overpass %s INCOMPLETE (%s). Good tiles are cached; re-run to fetch the rest.", name, LAST_ERROR[name])
        return None
    logger.info("Overpass %s: %d tiles (%d cached, %d requests) -> %d elements", name, len(tiles),
                info["cache_hits"], info["network_requests"], len(merged))
    if kind == "roads" and not merged:
        msg = "0 roads in every tile: unusual for a populated valley - check attic coverage / the highway tag list before trusting 'no roads'."
        LAST_WARNINGS[name] = [msg]
        logger.warning("Overpass roads: %s", msg)
    return to_geojson({"elements": list(merged.values())}, kind, aoi)
