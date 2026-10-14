"""
Pre-fetch the pre-event OSM layers for an AOI and fill the local Overpass cache, so later runs (and the live demo) do not depend on the public server's mood.

  python scripts/warm_osm_cache.py --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26

Safe to re-run: tiles already cached are not requested again, so an interrupted or partly failed run simply resumes. Exit code 1 if any layer is still incomplete.
Same compliance rules as the pipeline: the snapshot is always strictly before the event date and every Overpass query carries the attic date.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_config                      # noqa: E402
from app.services import osm_overpass                  # noqa: E402
from app.services.osm_snapshot import fetch_all_osm_layers   # noqa: E402


def load_aoi(path: str) -> dict:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]
    return gj["geometry"] if gj.get("type") == "Feature" else gj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", required=True, help="event date YYYY-MM-DD (the snapshot is taken before it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    aoi, event = load_aoi(args.aoi), date.fromisoformat(args.date)
    margin = float(get_config()["osm"].get("access_context_margin_km", 0) or 0)
    kwargs = {"context_margin_km": margin} if "context_margin_km" in inspect.signature(fetch_all_osm_layers).parameters else {}
    res = fetch_all_osm_layers(aoi, event, **kwargs)
    print("\nsnapshot date:", res["snapshot_date"], "| context margin km:", margin if kwargs else "n/a")
    for k, v in res["counts"].items():
        info = osm_overpass.LAST_INFO.get({"settlements": "villages", "health_facilities": "health"}.get(k, k), {})
        print(f"  {k:<18} {v:>6} features   {info}")
    for name, warns in osm_overpass.LAST_WARNINGS.items():
        for w in warns:
            print(f"  WARNING {name}: {w}")
    if res["failed_layers"]:
        print("\nINCOMPLETE layers:", res["failed_layers"], res.get("failure_reasons"))
        print("Good tiles are cached. Run this script again (ideally a few minutes later) to fetch the rest.")
        return 1
    print("\nAll layers complete and cached under", osm_overpass._cache_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
