"""
FloodLens Nepal — CLI Entry Point
===================================
Run the full analysis pipeline from the command line.

Usage examples
--------------
# Quick scene-search only (Day 1 proof-of-concept):
    python run.py search --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26

# Full OSM fetch:
    python run.py osm --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26

# Day-1 live-service smoke test:
    python run.py fetch-test --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-20

# Full pipeline (end-to-end):
    python run.py analyse --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("floodlens")

ROOT = Path(__file__).parent
for _stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252; OSM names contain e.g. "\u1e45"
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
import app.config  # noqa: E402,F401  (side effect: loads <repo>/.env before any os.getenv call)
OUTPUTS_ROOT = ROOT / "outputs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_aoi(aoi_path: str) -> dict:
    """Load AOI from a GeoJSON file and return the geometry dict."""
    path = Path(aoi_path)
    if not path.exists():
        logger.error("AOI file not found: %s", path)
        sys.exit(1)
    with path.open() as f:
        fc = json.load(f)
    # Accept a FeatureCollection, a single Feature or a bare geometry. geojson.io files often start with a marker/line,
    # so pick the first POLYGON instead of blindly taking features[0].
    kinds = ("Polygon", "MultiPolygon")
    if fc.get("type") == "FeatureCollection":
        geoms = [ft.get("geometry") or {} for ft in fc.get("features", [])]
    elif fc.get("type") == "Feature":
        geoms = [fc.get("geometry") or {}]
    else:
        geoms = [fc]
    geom = next((g for g in geoms if g.get("type") in kinds), None)
    if geom is None:
        found = [g.get("type") for g in geoms]
        logger.error("No Polygon found in %s (geometry types present: %s). Draw a POLYGON, not a marker or line.", path, found)
        sys.exit(1)
    return geom


def parse_date(date_str: str) -> date:
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        logger.error("Invalid date format: %s (expected YYYY-MM-DD)", date_str)
        sys.exit(1)


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_search(args: argparse.Namespace) -> None:
    """Query CDSE STAC for valid Sentinel-1 scene pairs."""
    from app.services.scene_selector import select_scenes, print_scene_summary

    aoi = load_aoi(args.aoi)
    event_date = parse_date(args.date)

    logger.info("Querying CDSE STAC for S1/S2 scenes …")
    result = select_scenes(
        aoi_geojson=aoi,
        event_date=event_date,
        before_window_days=args.before_window,
        after_window_days=args.after_window,
    )
    print_scene_summary(result)

    # Save to outputs/scene_selection.json
    OUTPUTS_ROOT.mkdir(exist_ok=True)
    out = OUTPUTS_ROOT / "scene_selection.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved → %s", out)


def cmd_osm(args: argparse.Namespace) -> None:
    """Fetch OSM pre-event snapshot via ohsome API."""
    from app.services.osm_snapshot import fetch_all_osm_layers

    aoi = load_aoi(args.aoi)
    event_date = parse_date(args.date)

    logger.info("Fetching pre-event OSM data for %s …", event_date.isoformat())
    data = fetch_all_osm_layers(aoi_geojson=aoi, event_date=event_date)
    if data["failed_layers"]:
        print(f"!! layers that FAILED to download: {data['failed_layers']}")

    print("\n" + "=" * 60)
    print("OSM SNAPSHOT RESULT")
    print("=" * 60)
    print(f"  Snapshot date : {data['snapshot_date']}")
    for k, v in data["counts"].items():
        print(f"  {k:<20}: {v:>5} features")
    print("=" * 60 + "\n")

    # Save raw GeoJSONs
    OUTPUTS_ROOT.mkdir(exist_ok=True)
    for layer in ("buildings", "roads", "bridges", "settlements", "towns", "health_facilities"):
        if data[layer]:
            out = OUTPUTS_ROOT / f"osm_{layer}.geojson"
            out.write_text(json.dumps(data[layer], indent=2), encoding="utf-8")
            logger.info("Saved → %s", out)


def cmd_analyse(args: argparse.Namespace) -> None:
    """Run full end-to-end pipeline (primary entry point and demo fallback)."""
    from app.api.schemas import ModelMode, RunRequest
    from app.services.orchestrator import run_pipeline

    aoi = load_aoi(args.aoi)
    request = RunRequest(
        aoi_geojson=aoi, event_date=parse_date(args.date), event_name=args.event_name,
        before_window_days=args.before_window, after_window_days=args.after_window,
        model_mode=ModelMode(args.model_mode), flood_threshold=args.threshold,
        use_optical_confirmation=not args.no_optical,
        buffers_m=[float(b) for b in args.buffers.split(",")],
    )
    before = [Path(p) for p in args.before_tif] if args.before_tif else None
    summary = run_pipeline(
        request=request,
        s1_before_tif=before[0] if before else None,
        s1_before_extra_tifs=before[1:] if before else None,
        s1_after_tif=Path(args.after_tif) if args.after_tif else None,
        s2_after_tif=Path(args.s2_tif) if args.s2_tif else None,
    )
    print("\n" + "=" * 60)
    print(f"RUN {summary.run_id}: {summary.status.value.upper()}  ({summary.runtime_s:.0f} s)")
    print("=" * 60)
    m = summary.flood_metrics
    print(f"  Flood/debris area : {m.flood_debris_area_km2:.2f} km2 (water {m.water_area_km2:.2f}, debris {m.debris_area_km2:.2f})")
    print(f"  Confidence        : {summary.quality.flood_confidence.value.upper()}")
    print(f"  Settlements       : {summary.settlements_assessed} assessed, {summary.settlements_potentially_cut_off} potentially cut off "
          f"(range {summary.cut_off_min_over_buffers}-{summary.cut_off_max_over_buffers}), {summary.settlements_uncertain} uncertain")
    oc = Path("outputs") / "runs" / summary.run_id / "osm_counts.json"
    if oc.exists():
        print(f"  OSM layers        : {json.loads(oc.read_text(encoding='utf-8'))}   (targets used for cut-off: {summary.n_osm_targets})")
    for w in summary.quality.warnings:
        print(f"  ! {w}")
    print(f"  Report            : outputs/runs/{summary.run_id}/situation_report_en.md")
    print("=" * 60 + "\n")


def cmd_fetch_test(args: argparse.Namespace) -> None:
    """Day-1 smoke test of every live service the pipeline depends on. Prints PASS/FAIL per check."""
    import os
    import requests
    from app.config import get_config
    cfg = get_config()
    ok_all = True

    def check(name, fn):
        nonlocal ok_all
        try:
            print(f"PASS  {name}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            print(f"FAIL  {name}: {exc}")

    def stac_collections():
        # /collections is PAGINATED (first page only), so ask for each configured collection directly.
        base = cfg["scene_selection"]["stac_url"]
        missing = []
        for need in (cfg["scene_selection"]["s1_collection"], cfg["scene_selection"]["s2_collection"]):
            r = requests.get(f"{base}/collections/{need}", timeout=30)
            if r.status_code != 200:
                missing.append(f"{need} (HTTP {r.status_code})")
        if missing:
            ids = [c["id"] for c in requests.get(f"{base}/collections", timeout=30).json().get("collections", [])]
            raise RuntimeError(f"not found at {base}: {missing}. First page of ids offered: {ids[:20]}")
        return "configured S1/S2 collection ids exist"

    def stac_props():
        from app.services.scene_selector import _search_stac
        aoi = load_aoi(args.aoi)
        from shapely.geometry import shape
        d = parse_date(args.date)
        items = _search_stac(cfg["scene_selection"]["s1_collection"], list(shape(aoi).bounds),
                             d - timedelta(days=7), d + timedelta(days=7), limit=3)
        if not items:
            raise RuntimeError("no S1 items within +-7 days of that date (or the STAC request failed: see the log line above); "
                               "check stac_url / s1_collection, or try another --date")
        keys = sorted(items[0]["properties"].keys())
        need = ["sat:relative_orbit", "sat:orbit_state", "sar:instrument_mode", "sar:polarizations"]
        missing = [k for k in need if k not in keys]
        return f"{len(items)} item(s); missing expected props: {missing or 'none'}; all props: {keys}"

    def ohsome_meta():
        from app.services.scene_selector import latest_ohsome_date
        d = latest_ohsome_date()
        if d is None:
            raise RuntimeError("could not read /metadata temporal extent")
        return f"latest ohsome date {d}"

    def token():
        from app.services.cdse_fetch import get_token
        if not (os.getenv("CDSE_CLIENT_ID") and os.getenv("CDSE_CLIENT_SECRET")):
            env_file = ROOT / ".env"
            hint = ("no .env file in " + str(ROOT) if not env_file.exists() else ".env exists but the two values are empty")
            extra = " (found .env.txt: Notepad added .txt; rename it to .env)" if (ROOT / ".env.txt").exists() else ""
            raise RuntimeError(f"CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set: {hint}{extra}")
        return "OAuth token obtained" if get_token() else "no token"

    def dem_tile():
        from app.services.terrain import COP_DEM_S3_BASE, _dem_tile_name
        # _dem_tile_name already returns "<folder>/<file>.tif"; terrain.py builds the URL as BASE/<that>
        url = f"{COP_DEM_S3_BASE}/{_dem_tile_name(27, 85)}"
        r = requests.head(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        return "Copernicus DEM tile reachable"

    check("CDSE STAC collections", stac_collections)
    check("CDSE STAC S1 item properties", stac_props)
    check("ohsome metadata", ohsome_meta)
    check("CDSE OAuth token", token)
    check("Copernicus DEM tile", dem_tile)
    print("\nAll checks passed." if ok_all else "\nFix the FAIL lines (configs/config.yaml, .env) before relying on live runs.")
    sys.exit(0 if ok_all else 1)


def cmd_diagnose_ohsome(args: argparse.Namespace) -> None:
    """Find out WHICH kind of ohsome request gets a 403 from your network (runs a ladder of small requests)."""
    import requests
    from app.config import get_config
    cfg = get_config()
    base = cfg["osm"]["ohsome_url"]
    t = cfg["osm"].get("fallback_latest_date", "2026-07-27")
    bb = "85.30,28.00,85.31,28.01"
    poly = json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"id": "a"},
          "geometry": {"type": "Polygon", "coordinates": [[[85.30, 28.0], [85.31, 28.0], [85.31, 28.01], [85.30, 28.01], [85.30, 28.0]]]}}]})
    ua = {"User-Agent": "FloodLens-Nepal/0.4 (educational hackathon prototype; python-requests)"}
    tests = [
        ("count, simple filter, bboxes (curl-like)", "/elements/count", {"bboxes": bb, "filter": "building=*", "time": t}, None),
        ("count, 'and geometry:polygon' filter", "/elements/count", {"bboxes": bb, "filter": "building=* and geometry:polygon", "time": t}, None),
        ("geometry, simple filter, bboxes", "/elements/geometry", {"bboxes": bb, "filter": "building=*", "time": t}, None),
        ("geometry + properties=tags,metadata", "/elements/geometry", {"bboxes": bb, "filter": "building=*", "time": t, "properties": "tags,metadata"}, None),
        ("geometry + properties=tags", "/elements/geometry", {"bboxes": bb, "filter": "building=*", "time": t, "properties": "tags"}, None),
        ("geometry, bpolys polygon", "/elements/geometry", {"bpolys": poly, "filter": "building=*", "time": t}, None),
        ("geometry, roads filter 'in (...)'", "/elements/geometry", {"bboxes": bb, "filter": "highway in (primary,secondary,track) and geometry:line", "time": t}, None),
        ("geometry, same as first but default UA", "/elements/geometry", {"bboxes": bb, "filter": "building=*", "time": t}, {}),
    ]
    for name, ep, data, hdr in tests:
        try:
            r = requests.post(base + ep, data=data, headers=ua if hdr is None else hdr, timeout=60)
            ok = r.status_code == 200
            snippet = "" if ok else " | " + " ".join(r.text.split())[:140]
            print(f"{'PASS' if ok else 'FAIL'}  HTTP {r.status_code}  {name}{snippet}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name}: {exc}")
    print("\nSend me this list: the first FAIL row shows which feature of the request triggers the block.")


def cmd_osm_report(args: argparse.Namespace) -> None:
    """What does pre-event OSM contain for this valley? (counts, place/health tags, names)"""
    from collections import Counter
    from app.config import get_config
    from app.services.osm_snapshot import fetch_all_osm_layers
    from app.services.scene_selector import choose_osm_snapshot_date
    cfg = get_config()
    aoi, ev = load_aoi(args.aoi), parse_date(args.date)
    snap, warns = choose_osm_snapshot_date(ev)
    for w in warns:
        print("!", w)
    osm = fetch_all_osm_layers(aoi, ev, snap, context_margin_km=cfg["osm"].get("access_context_margin_km", 0))
    print(f"\nOSM snapshot {osm['snapshot_date']} | sources {osm.get('sources')} | AOI + {cfg['osm'].get('access_context_margin_km', 0)} km margin for roads/towns/health")
    print("counts:", osm["counts"])
    if osm["failed_layers"]:
        print("FAILED layers:", osm["failed_layers"], "reasons:", osm.get("failure_reasons"))

    def props(layer):
        return [f.get("properties", {}) for f in (osm.get(layer) or {}).get("features", [])]
    print("\nsettlement place= values:", dict(Counter(p.get("place") for p in props("settlements"))))
    names = [p.get("name") or p.get("name:en") or "(unnamed)" for p in props("settlements")]
    print(f"settlement names ({len(names)}): {names[:25]}{' ...' if len(names) > 25 else ''}")
    print("towns/cities:", [(p.get("name") or "(unnamed)", p.get("place")) for p in props("towns")])
    print("health facilities:", [(p.get("name") or "(unnamed)", p.get("amenity") or p.get("healthcare")) for p in props("health_facilities")])
    print("road highway= values:", dict(Counter(p.get("highway") for p in props("roads")).most_common(8)))
    out = Path("outputs") / "osm_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"snapshot": osm["snapshot_date"], "counts": osm["counts"], "failed": osm["failed_layers"],
                               "reasons": osm.get("failure_reasons")}, indent=2), encoding="utf-8")
    print("\nSaved", out)


def cmd_report(args: argparse.Namespace) -> None:
    """Re-render the report from an existing run's summary.json."""
    from app.services.report_generator import generate_report_from_run
    print(generate_report_from_run(Path(args.run_dir)))


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodlens",
        description="FloodLens Nepal — Satellite flood impact mapping CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args
    aoi_arg = argparse.ArgumentParser(add_help=False)
    aoi_arg.add_argument(
        "--aoi", required=True,
        help="Path to AOI GeoJSON file (Polygon or FeatureCollection)."
    )
    aoi_arg.add_argument(
        "--date", required=True,
        help="Flood event date (YYYY-MM-DD)."
    )

    # search
    p_search = sub.add_parser("search", parents=[aoi_arg],
                               help="Query STAC and print available S1/S2 scenes.")
    p_search.add_argument("--before-window", type=int, default=18)
    p_search.add_argument("--after-window", type=int, default=18)
    p_search.set_defaults(func=cmd_search)

    # osm
    p_osm = sub.add_parser("osm", parents=[aoi_arg],
                            help="Fetch pre-event OSM layers via ohsome API.")
    p_osm.set_defaults(func=cmd_osm)

    # analyse
    p_an = sub.add_parser("analyse", parents=[aoi_arg], help="Run full end-to-end pipeline.")
    p_an.add_argument("--before-window", type=int, default=18)
    p_an.add_argument("--after-window", type=int, default=18)
    p_an.add_argument("--before-tif", nargs="+", default=None,
                      help="Local terrain-corrected pre-event S1 GeoTIFF(s) [VV,VH]; extras = same-track reference scenes.")
    p_an.add_argument("--after-tif", default=None, help="Local post-event S1 GeoTIFF [VV,VH].")
    p_an.add_argument("--s2-tif", default=None, help="Local post-event S2 GeoTIFF [B03,B04,B08,B11,SCL,mask].")
    p_an.add_argument("--no-optical", action="store_true")
    p_an.add_argument("--model-mode", default="baseline", choices=["baseline", "ai", "ensemble"])
    p_an.add_argument("--threshold", type=float, default=None, help="Override water log-ratio threshold (dB).")
    p_an.add_argument("--buffers", default="0,30,60", help="Comma-separated hazard buffers in metres.")
    p_an.add_argument("--event-name", default="Trishuli Flood Aug 2026")
    p_an.set_defaults(func=cmd_analyse)

    p_ft = sub.add_parser("fetch-test", parents=[aoi_arg], help="Day-1 smoke test of STAC, ohsome, OAuth and DEM access.")
    p_ft.set_defaults(func=cmd_fetch_test)

    p_rep = sub.add_parser("report", help="Re-render the report for a finished run.")
    p_rep.add_argument("--run-dir", required=True)
    p_rep.set_defaults(func=cmd_report)

    p_or = sub.add_parser("osm-report", parents=[aoi_arg], help="Show what pre-event OSM contains for an AOI (settlements, towns, hospitals, roads).")
    p_or.set_defaults(func=cmd_osm_report)

    p_do = sub.add_parser("diagnose-ohsome", help="Test which ohsome request shapes get HTTP 403 on your network.")
    p_do.set_defaults(func=cmd_diagnose_ohsome)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
