# Patch: queue-aware, cached Overpass client

Built and tested against FloodLens v0.4.15 (153 tests pass). Your folder is v0.4_17 - see "Apply" for the safety check.

## What it fixes
The public Overpass server answers HTTP 504 / 429 when it is busy. The old client retried 3x and gave up, so heavy layers (roads, health) failed and
the cut-off analysis was "not assessed". Now:
- **Queue-aware**: reads the server's /status page and waits for a free slot before each request; backs off exponentially (with jitter, honours Retry-After); paces requests.
- **Tiled**: big areas are split into ~11 km tiles on a fixed global lattice. A tile that answers "out of memory / timed out" is split again.
- **Cached**: every successful tile is stored in `data/cache/overpass/` keyed by the exact dated query. Dated historical queries never change, so the cache never goes stale. Re-runs fetch only what is missing.
- **All-or-nothing layers**: a layer is returned only if every tile succeeded (a partial road network would create false "cut-off" results). Otherwise it is reported as failed, never as 0 features, and the good tiles stay cached.
- **Suspicious zero**: empty answers for roads/buildings are not cached; 0 roads logs a warning.
- Unchanged compliance rules: every query carries `[date:"<snapshot>T00:00:00Z"]`, snapshot strictly before the event, cache can never serve an undated query. No config change is required (optional keys below).

## Files (only these change)
- `app/services/osm_overpass.py`  - replaced (same public names: build_query, to_geojson, fetch_layer, LAST_ERROR, HEADERS, ...)
- `tests/test_overpass_client.py` - new, 12 tests with a fake server (no network)
- `scripts/warm_osm_cache.py`     - new, fills the cache for an AOI

## Apply
1. Back up: `copy app\services\osm_overpass.py app\services\osm_overpass.py.bak`
2. Extract this zip over your `FloodLens` folder (same folder that contains `app`, `configs`, `run.py`).
3. Check: `python -m pytest tests/test_osm_overpass.py tests/test_compliance.py tests/test_overpass_client.py -q`
   If anything fails or imports break, restore the .bak file and send me `app/services/osm_overpass.py` and `app/services/osm_snapshot.py` from your folder.

## Use
```
python scripts/warm_osm_cache.py --aoi <your aoi file> --date 2026-08-26
```
Run it now, and again a few minutes later if it says INCOMPLETE. Afterwards `run.py analyse` reads OSM from the cache (fast, no 504s) for that AOI.
Tiles are on a global lattice, so a slightly different AOI over the same area reuses most tiles. A brand-new area still needs live requests (the queue-aware retry helps there).

## Optional config keys (under `osm:` - defaults are built in, do not add them after you freeze)
overpass_tile_deg (0.1), overpass_cache_dir ("data/cache/overpass"), overpass_max_attempts (6), overpass_layer_deadline_s (900),
overpass_backoff_base_s (15), overpass_backoff_cap_s (120)

## Caveats
- Do not add other Overpass mirrors unless they support attic ([date:...]) queries; a server that ignores the header returns today's map (post-event edits).
- If the server is down for a long time, a layer can still fail after its time budget; then re-run later. If roads stay empty in every tile, that may be real (check attic coverage).
- This does not make ohsome work; it only makes the Overpass fallback reliable.
