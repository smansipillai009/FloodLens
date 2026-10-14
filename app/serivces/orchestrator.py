"""
FloodLens Nepal — pipeline orchestrator
=======================================
scene selection -> (fetch) -> terrain -> S1 preprocessing -> detection (baseline / U-Net water + baseline debris)
-> S2 confirmation -> OSM pre-event snapshot -> impact -> access -> summary.json -> report.

Behaviour fixed vs the first version
  * No NameError when S1 inputs are missing (flood_result used to be undefined).
  * No same-track pair => pixel mapping is skipped, run is PARTIAL with the exact reason + candidates.
  * AOI size limit enforced (auto-shrink + warning); metric CRS chosen per AOI (not hard-coded UTM 45N).
  * `osm_snapshot_verified_pre_event` is computed, not assumed.
  * AI/ensemble modes really exist; they fall back to baseline with a warning if no checkpoint.
  * Timings, config hash and config-freeze status are recorded.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from app.api.schemas import (
    REQUIRED_ATTRIBUTIONS, AnalysisSummary, FloodConfidence, FloodMetrics, InfrastructureImpact,
    ModelMode, QualityFlags, RunRequest, RunStatus, SceneSelectionResult,
)
from app.config import config_hash, get_config, locked_hash, utm_crs_for_geom
from app.services.provenance import FLOODLENS_VERSION, create_run_directory, write_manifest
from app.services.report_generator import write_report

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_ROOT, DATA_ROOT = ROOT / "outputs", ROOT / "data"

STANDARD_ATTRIBUTIONS = REQUIRED_ATTRIBUTIONS  # single source of truth: app/api/schemas.py

STANDARD_LIMITATIONS = [
    "Sentinel satellites revisit every few days: this system cannot warn of a sudden glacier collapse; it maps impact after the fact.",
    "C-band SAR gives false positives on steep terrain, radar shadow/layover and wet soil, and cannot see water under dense vegetation.",
    "Optical data is cloud-limited; reconstructed/synthetic pixels are never used.",
    "A flood/debris mask cannot confirm structural damage to buildings or bridges.",
    "OSM may omit roads, bridges, buildings, hospitals or settlements. A road crossing the mask is only potentially blocked.",
    "A graph path does not mean a hospital is operational or a road passable; 'potentially cut off' is based on mapped-road connectivity only.",
    "Water labels in the training data are flood water, not debris; the model is trained mostly outside the Himalaya.",
    "Estimates support prioritisation and are not an official assessment; field verification is required.",
]


def _area_km2(aoi: dict, crs: str) -> float:
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    return transform(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, shape(aoi)).area / 1e6


def shrink_aoi(aoi: dict, crs: str, max_km2: float) -> tuple[dict, float]:
    """Scale the AOI about its centroid so area == max_km2 (plan: auto-shrink + warn)."""
    from shapely.affinity import scale
    from shapely.geometry import mapping, shape
    area = _area_km2(aoi, crs)
    if area <= max_km2:
        return aoi, area
    f = (max_km2 / area) ** 0.5
    g = scale(shape(aoi), xfact=f, yfact=f, origin="centroid")
    return mapping(g), max_km2


def validate_aoi_size(area_km2: float, cfg: dict) -> None:
    """A zero-size AOI (point, line, sliver) used to reach the Process API as width=0/height=0 and fail with a 400."""
    lo = cfg["aoi"].get("min_area_km2", 1.0)
    if area_km2 < lo:
        raise ValueError(f"AOI is only {area_km2:.4f} km2 (minimum {lo} km2). Draw a POLYGON (not a marker or line) "
                         "of roughly 50-200 km2 around the valley.")


def _confidence(q: QualityFlags, res_m: float = 10.0) -> FloodConfidence:
    if not (q.sar_pair_matched and q.dem_available):
        return FloodConfidence.LOW
    aligned = q.sar_alignment_rmse_m is None or q.sar_alignment_rmse_m <= 1.5 * res_m
    # HIGH requires MEASURED optical agreement; without it the best we can say is MEDIUM
    # (no accuracy evidence exists for a SAR-only run).
    optical_ok = q.optical_agreement_pct is not None and q.optical_agreement_pct >= 60
    if q.osm_snapshot_verified_pre_event and aligned and optical_ok:
        return FloodConfidence.HIGH
    return FloodConfidence.MEDIUM


def run_pipeline(
    request: RunRequest,
    outputs_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    s1_before_tif: Optional[Path] = None,
    s1_after_tif: Optional[Path] = None,
    s1_before_extra_tifs: Optional[list[Path]] = None,
    s2_after_tif: Optional[Path] = None,
    scene_result: Optional[SceneSelectionResult] = None,
    progress: Optional[Callable[[str, int], None]] = None,
) -> AnalysisSummary:
    t_start = time.time()
    cfg = get_config()
    outputs_root, data_root = outputs_root or OUTPUTS_ROOT, data_root or DATA_ROOT
    tick = progress or (lambda stage, pct: None)

    run_id, run_dir = create_run_directory(outputs_root)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    # ── 0. AOI validation ────────────────────────────────────────────────────
    aoi = request.aoi_geojson
    metric_crs = utm_crs_for_geom(aoi)
    validate_aoi_size(_area_km2(aoi, metric_crs), cfg)
    aoi, area = shrink_aoi(aoi, metric_crs, cfg["aoi"]["max_area_km2"])
    if aoi is not request.aoi_geojson:
        warnings.append(f"AOI exceeded {cfg['aoi']['max_area_km2']} km2 and was shrunk about its centre.")
    if area > cfg["aoi"]["live_target_km2"]:
        warnings.append(f"AOI is {area:.0f} km2 (> live target {cfg['aoi']['live_target_km2']} km2): expect a slow run.")

    summary = AnalysisSummary(
        run_id=run_id, status=RunStatus.RUNNING, event_date=request.event_date, event_name=request.event_name,
        aoi_area_km2=round(area, 1), model_mode=request.model_mode,
        attributions=STANDARD_ATTRIBUTIONS, limitations=STANDARD_LIMITATIONS,
        config_hash=config_hash(), config_frozen=(locked_hash() == config_hash()),
        floodlens_version=FLOODLENS_VERSION,
    )
    quality = QualityFlags()

    def stage(name, pct):
        logger.info("[%3d%%] %s", pct, name)
        tick(name, pct)

    # ── 1. scenes ────────────────────────────────────────────────────────────
    t = time.time(); stage("Scene selection", 5)
    if scene_result is None:
        from app.services.scene_selector import select_scenes
        scene_result = select_scenes(aoi, request.event_date, request.before_window_days, request.after_window_days)
    summary.scene_selection = scene_result
    summary.osm_snapshot_date = scene_result.osm_snapshot_date
    quality.sar_pair_matched = scene_result.same_track
    warnings += scene_result.warnings
    timings["scene_selection"] = time.time() - t

    have_local = bool(s1_before_tif and s1_after_tif and Path(s1_before_tif).exists() and Path(s1_after_tif).exists())
    flood_res: Optional[dict] = None
    terrain: dict = {}
    pixel_mapping_possible = have_local or scene_result.same_track
    if have_local and not scene_result.same_track:
        warnings.append("Local Sentinel-1 GeoTIFFs were supplied without a verified same-track scene pair: "
                        "YOU must ensure identical relative orbit and direction (different tracks cannot be compared).")

    if not pixel_mapping_possible:
        warnings.append("Pixel-wise flood mapping SKIPPED: no same-track Sentinel-1 before/after pair. "
                        "Candidate scenes are listed in summary.json.")

    # ── 2. acquisition ───────────────────────────────────────────────────────
    if pixel_mapping_possible and not have_local:
        t = time.time(); stage("Downloading terrain-corrected Sentinel-1", 12)
        try:
            from app.services import cdse_fetch
            tok = cdse_fetch.get_token()
            raw = run_dir / "raw"
            s1_before_tif, w1 = cdse_fetch.fetch_s1_scene(scene_result.s1_before, aoi, raw / "s1_before.tif", tok)
            s1_after_tif, w2 = cdse_fetch.fetch_s1_scene(scene_result.s1_after, aoi, raw / "s1_after.tif", tok)
            s1_before_extra_tifs = [cdse_fetch.fetch_s1_scene(s, aoi, raw / f"s1_before_ref{i}.tif", tok)[0]
                                    for i, s in enumerate(scene_result.s1_before_extra)]
            warnings += w1 + w2
            if request.use_optical_confirmation and scene_result.s2_after and not s2_after_tif:
                s2_after_tif = cdse_fetch.fetch_s2_scene(scene_result.s2_after, aoi, raw / "s2_after.tif", tok)
            have_local = True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not retrieve Sentinel data from CDSE ({exc}). Provide local GeoTIFFs or set CDSE_CLIENT_ID/SECRET.")
            logger.exception("CDSE retrieval failed")
        timings["download"] = time.time() - t

    # ── 3. terrain ───────────────────────────────────────────────────────────
    if have_local:
        t = time.time(); stage("Terrain (Copernicus DEM)", 30)
        from app.services.terrain import prepare_terrain_layers
        direction = scene_result.s1_before.orbit_direction.value if scene_result.s1_before else "ASCENDING"
        if not scene_result.s1_before:
            warnings.append("Orbit direction unknown (no scene metadata): ASCENDING assumed for the layover/shadow mask.")
        terrain = prepare_terrain_layers(
            aoi, run_dir, data_root / "reference", metric_crs, cfg["dem"]["max_slope_deg"],
            cfg["sentinel1"]["incidence_angle_deg"], 80.0 if direction == "ASCENDING" else 280.0,
            cfg["flood_detection"]["use_layover_shadow_mask"])
        quality.dem_available = terrain.get("dem") is not None
        if not quality.dem_available:
            warnings.append("DEM unavailable: terrain/HAND/layover filtering NOT applied; expect extra false positives.")
        elif terrain.get("hand_method") != "pysheds_d8_hand":
            warnings.append("HAND is only a local-relief proxy (pysheds not installed) and is NOT used as a filter; slope and radar-shadow masks apply. Expect more false positives on slopes.")
        timings["terrain"] = time.time() - t

        # ── 4. S1 preprocessing ──────────────────────────────────────────────
        t = time.time(); stage("Sentinel-1 preprocessing", 45)
        from app.services.sentinel1 import preprocess_s1_pair
        # The U-Net was trained on 10 m Lee-filtered patches. Feeding it a 30 m multilooked grid is a different input distribution, so AI mode keeps 10 m.
        ai_requested = request.model_mode in (ModelMode.AI, ModelMode.ENSEMBLE)
        if ai_requested and int(cfg["sentinel1"].get("multilook_factor", 1)) > 1:
            warnings.append("AI mode: Sentinel-1 kept at the native 10 m grid (multilook off) to match the U-Net training data; the baseline debris rule therefore runs at 10 m too.")
        s1 = preprocess_s1_pair([Path(s1_before_tif)] + [Path(p) for p in (s1_before_extra_tifs or [])],
                                Path(s1_after_tif), run_dir, aoi, metric_crs, multilook_factor=1 if ai_requested else None)
        quality.sar_alignment_rmse_m = s1.get("alignment_rmse_m")
        timings["s1_preprocessing"] = time.time() - t

        # ── 5. detection ─────────────────────────────────────────────────────
        t = time.time(); stage("Flood / debris detection", 60)
        from app.services.flood_detection import detect_flood
        water_prob = None
        if request.model_mode in (ModelMode.AI, ModelMode.ENSEMBLE):
            from app.ml import infer
            if request.model_mode == ModelMode.ENSEMBLE:
                warnings.append("Mask-level ensemble is not implemented; using U-Net water + baseline debris.")
            water_prob = infer.predict_water_probability(
                s1["s1_before_vv"], s1["s1_before_vh"], s1["s1_after_vv"], s1["s1_after_vh"], run_dir / "unet_water_probability.tif")
            if water_prob is None:
                warnings.append("AI mode requested but no trained checkpoint found: fell back to the SAR baseline.")
        flood_res = detect_flood(s1["s1_before_vv"], s1["s1_before_vh"], s1["s1_after_vv"], s1["s1_after_vh"],
                                 run_dir, terrain=terrain, water_prob_path=water_prob,
                                 thresholds=({"water_change_db_vv": -abs(request.flood_threshold)} if request.flood_threshold else None))
        warnings += flood_res["warnings"]
        summary.flood_detection_method = flood_res["method"]
        quality.detection_method_used = flood_res["method"]
        summary.flood_metrics = FloodMetrics(
            flood_debris_area_km2=flood_res["flood_area_km2"], water_area_km2=flood_res["water_area_km2"],
            debris_area_km2=flood_res["debris_area_km2"], water_pixel_count=flood_res["water_pixel_count"],
            debris_pixel_count=flood_res["debris_pixel_count"])
        timings["detection"] = time.time() - t

        # ── 6. optical confirmation ──────────────────────────────────────────
        if request.use_optical_confirmation and s2_after_tif and Path(s2_after_tif).exists():
            t = time.time(); stage("Sentinel-2 confirmation (clear pixels only)", 70)
            try:
                from app.services.s2_confirmation import run_s2_confirmation
                o = run_s2_confirmation(Path(s2_after_tif), flood_res["flood_class_path"], run_dir)
                quality.optical_cloud_pct_in_aoi = o["optical_cloud_pct_in_aoi"]
                quality.optical_agreement_pct = o["optical_agreement_pct"]
                summary.flood_metrics.optical_confirmed_km2 = o["optical_confirmed_km2"]
                if o["insufficient_optical"]:
                    warnings.append("Too little clear Sentinel-2 coverage over the candidates: no optical confirmation (SAR-only).")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Optical confirmation failed ({exc}); result is SAR-only.")
                logger.exception("S2 confirmation failed")
            timings["optical"] = time.time() - t
        elif request.use_optical_confirmation:
            warnings.append("No Sentinel-2 data used: result is SAR-only (no optical confirmation).")

    # ── 7. OSM ───────────────────────────────────────────────────────────────
    t = time.time(); stage("OSM pre-event snapshot (ohsome)", 78)
    from app.services.osm_snapshot import fetch_all_osm_layers
    osm = fetch_all_osm_layers(aoi, request.event_date, scene_result.osm_snapshot_date,
                               context_margin_km=cfg["osm"].get("access_context_margin_km", 0))
    quality.osm_snapshot_verified_pre_event = date.fromisoformat(osm["snapshot_date"]) < request.event_date
    for k in ("buildings", "roads", "bridges", "settlements", "towns", "health_facilities"):
        if osm.get(k):
            (run_dir / f"osm_{k}.geojson").write_text(json.dumps(osm[k]), encoding="utf-8")
    (run_dir / "osm_counts.json").write_text(json.dumps({"snapshot_date": osm["snapshot_date"], **osm["counts"]}, indent=2), encoding="utf-8")
    if any(str(v).startswith("overpass") for v in osm.get("sources", {}).values()):
        warnings.append(f"OSM layers were taken from an Overpass attic snapshot dated {osm['snapshot_date']} because the ohsome API "
                        "refused geometry requests (same pre-event date; coverage of the attic database is not verified).")
    if osm["failed_layers"]:
        warnings.append(f"OSM layers failed to download: {osm['failed_layers']} (reasons: {osm.get('failure_reasons', {})}). "
                        "Impact/access for them is NOT assessed.")
    summary.settlements_assessed = osm["counts"]["settlements"]
    timings["osm"] = time.time() - t

    # ── 8. impact + access ───────────────────────────────────────────────────
    if flood_res is not None:
        t = time.time(); stage("Infrastructure impact & cut-off analysis", 88)
        from app.services.accessibility import analyze_accessibility
        from app.services.impact_analysis import analyze_infrastructure_impact
        poly = flood_res["flood_polygons_path"]
        impact, det = analyze_infrastructure_impact(poly, osm.get("buildings"), osm.get("roads"), osm.get("bridges"),
                                                    run_dir, metric_crs=metric_crs, buffer_m_list=request.buffers_m)
        summary.infrastructure = impact
        (run_dir / "impact_details.json").write_text(json.dumps(det, indent=2), encoding="utf-8")
        res, acc, _ = analyze_accessibility(osm.get("roads"), osm.get("settlements"), osm.get("health_facilities"),
                                            poly, run_dir, metric_crs=metric_crs, towns_data=osm.get("towns"),
                                            buffer_m_list=request.buffers_m)
        summary.settlement_results = res
        summary.settlements_assessed = acc["assessed"]
        summary.settlements_potentially_cut_off, summary.settlements_uncertain = acc["cut_off"], acc["uncertain"]
        summary.settlements_connected = acc["connected"]
        summary.cut_off_min_over_buffers, summary.cut_off_max_over_buffers = acc["cut_off_min"], acc["cut_off_max"]
        summary.n_osm_targets = acc["n_targets"]
        by_b = {r["buffer_m"]: r for r in det["buffers"]}
        for br in acc["buffer_rows"]:
            r = by_b.get(br.buffer_m, {})
            br.buildings, br.road_km, br.bridges = r.get("buildings", 0), r.get("road_km", 0.0), r.get("bridges", 0)
        summary.buffer_sensitivity = acc["buffer_rows"]
        if acc["n_targets"] == 0:
            warnings.append("No hospital/clinic/town mapped in OSM inside the AOI: every settlement is 'uncertain'. Facilities are never inferred.")
        timings["impact_access"] = time.time() - t

    # ── 9. wrap-up ───────────────────────────────────────────────────────────
    stage("Writing summary and report", 96)
    quality.warnings = list(dict.fromkeys(warnings))
    quality.flood_confidence = _confidence(quality, cfg["sentinel1"]["resolution_m"])
    summary.quality = quality
    summary.status = RunStatus.DONE if flood_res is not None else RunStatus.PARTIAL
    summary.runtime_s = round(time.time() - t_start, 1)

    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    write_manifest(run_dir, run_id, request.model_dump(mode="json"), scene_result.model_dump(mode="json"),
                   extra={"timings_s": {k: round(v, 1) for k, v in timings.items()}, "runtime_s": summary.runtime_s,
                          "metric_crs": metric_crs, "osm_snapshot_date": osm["snapshot_date"],
                          "inputs": {"s1_before": str(s1_before_tif), "s1_after": str(s1_after_tif),
                                     "s1_before_extra": [str(p) for p in (s1_before_extra_tifs or [])],
                                     "s2_after": str(s2_after_tif) if s2_after_tif else None}})
    write_report(summary, run_dir / "situation_report_en.md")
    stage("Done", 100)
    logger.info("Run %s %s in %.0f s", run_id, summary.status.value, summary.runtime_s)
    return summary
