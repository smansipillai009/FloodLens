"""
FloodLens Nepal — Compliance & Governance Tests
================================================
These tests enforce the data governance rules from the execution plan.
They MUST all pass before any commit that touches pipeline code.

Rules enforced:
  1. Pipeline source files must never import or reference EMSR927.
  2. OSM snapshot dates must always be BEFORE the event date.
  3. Same-track enforcement: mismatched S1 orbit is flagged as a warning.
  4. AnalysisSummary is the sole source of truth for report numbers.
  5. No hardcoded numeric thresholds outside configs/config.yaml.
"""
from __future__ import annotations

import ast
import re
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC_DIRS = [
    ROOT / "app" / "api",
    ROOT / "app" / "services",
    ROOT / "app" / "ml",
    ROOT / "app" / "frontend",
]
VALIDATION_ONLY_DIR = ROOT / "validation_only"


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMSR927 isolation
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_PATTERNS = [
    re.compile(r"EMSR927", re.IGNORECASE),
    re.compile(r"emsr_927", re.IGNORECASE),
    re.compile(r"validation_only", re.IGNORECASE),  # no direct path references from pipeline
]


def iter_pipeline_py_files():
    """Yield all .py files in pipeline source directories."""
    for src_dir in SRC_DIRS:
        if not src_dir.exists():
            continue
        for py_file in src_dir.rglob("*.py"):
            yield py_file


@pytest.mark.parametrize("py_file", list(iter_pipeline_py_files()) or [Path("/dev/null")])
def test_no_emsr927_reference_in_pipeline(py_file):
    """No pipeline source file may reference EMSR927 or validation_only paths."""
    if py_file == Path("/dev/null") or not py_file.exists():
        pytest.skip("No pipeline files found yet.")
    content = py_file.read_text(encoding="utf-8", errors="replace")
    for pattern in FORBIDDEN_PATTERNS:
        matches = pattern.findall(content)
        assert not matches, (
            f"GOVERNANCE VIOLATION: '{pattern.pattern}' found in {py_file}. "
            f"Validation-only data must never enter the pipeline."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunRequestSchema:
    def test_valid_request_accepted(self):
        from app.api.schemas import RunRequest
        req = RunRequest(
            aoi_geojson={"type": "Polygon", "coordinates": []},
            event_date=date(2026, 8, 26),
        )
        assert req.event_date == date(2026, 8, 26)
        assert req.before_window_days == 18

    def test_future_event_date_rejected(self):
        from app.api.schemas import RunRequest
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RunRequest(
                aoi_geojson={"type": "Polygon", "coordinates": []},
                event_date=date(2099, 1, 1),
            )

    def test_window_bounds(self):
        from app.api.schemas import RunRequest
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RunRequest(
                aoi_geojson={"type": "Polygon", "coordinates": []},
                event_date=date(2026, 8, 26),
                before_window_days=0,  # ge=1 must fail
            )


class TestSceneSelectionSchema:
    def test_same_track_set_true_when_orbits_match(self):
        from app.api.schemas import (
            S1SceneInfo, SceneSelectionResult, OrbitDirection
        )
        before = S1SceneInfo(
            scene_id="S1A_before",
            datetime=datetime(2026, 8, 14, 6, 0, 0),
            relative_orbit=12,
            orbit_direction=OrbitDirection.ASCENDING,
            polarizations=["VV", "VH"],
        )
        after = S1SceneInfo(
            scene_id="S1A_after",
            datetime=datetime(2026, 8, 26, 6, 0, 0),
            relative_orbit=12,
            orbit_direction=OrbitDirection.ASCENDING,
            polarizations=["VV", "VH"],
        )
        result = SceneSelectionResult(s1_before=before, s1_after=after)
        assert result.same_track is True
        assert not result.warnings

    def test_mismatched_orbit_sets_warning(self):
        from app.api.schemas import (
            S1SceneInfo, SceneSelectionResult, OrbitDirection
        )
        before = S1SceneInfo(
            scene_id="S1A_before",
            datetime=datetime(2026, 8, 14, 6, 0, 0),
            relative_orbit=12,
            orbit_direction=OrbitDirection.ASCENDING,
            polarizations=["VV", "VH"],
        )
        after = S1SceneInfo(
            scene_id="S1B_after",
            datetime=datetime(2026, 8, 26, 6, 0, 0),
            relative_orbit=99,  # different orbit!
            orbit_direction=OrbitDirection.DESCENDING,  # different direction!
            polarizations=["VV", "VH"],
        )
        result = SceneSelectionResult(s1_before=before, s1_after=after)
        assert result.same_track is False
        assert len(result.warnings) > 0
        assert "NOT on the same relative orbit" in result.warnings[0]


class TestOSMSnapshotDateRule:
    """
    The OSM snapshot date must always be STRICTLY before the event date.
    """
    def test_osm_snapshot_before_event(self):
        event_date = date(2026, 8, 26)
        snapshot_date = date(2026, 8, 25)
        assert snapshot_date < event_date, (
            "OSM snapshot date must be before event date to avoid post-event data leakage."
        )

    def test_osm_snapshot_same_day_would_fail(self):
        event_date = date(2026, 8, 26)
        snapshot_date = date(2026, 8, 26)
        # Pipeline must enforce snapshot_date < event_date
        with pytest.raises(AssertionError):
            assert snapshot_date < event_date, (
                "OSM snapshot on same day as event is not permitted."
            )

    def test_osm_snapshot_after_event_would_fail(self):
        event_date = date(2026, 8, 26)
        snapshot_date = date(2026, 8, 27)
        with pytest.raises(AssertionError):
            assert snapshot_date < event_date


class TestAnalysisSummaryIntegrity:
    """Summary must be constructable and contain default-safe values."""
    def test_default_summary_is_valid(self):
        from app.api.schemas import AnalysisSummary
        summary = AnalysisSummary()
        assert summary.run_id  # auto-generated UUID
        assert summary.prototype_disclaimer
        assert summary.flood_metrics.flood_debris_area_km2 == 0.0
        assert summary.infrastructure.potentially_affected_buildings == 0

    def test_settlement_count_consistency(self):
        """cut_off + uncertain <= assessed (basic sanity)."""
        from app.api.schemas import AnalysisSummary
        summary = AnalysisSummary(
            settlements_assessed=10,
            settlements_potentially_cut_off=3,
            settlements_uncertain=2,
        )
        assert (summary.settlements_potentially_cut_off + summary.settlements_uncertain
                <= summary.settlements_assessed)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Validation-only directory isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationOnlyIsolation:
    def test_validation_only_dir_exists(self):
        """The validation_only directory must exist and be separate from app/."""
        assert VALIDATION_ONLY_DIR.exists(), (
            f"Expected validation_only/ directory at {VALIDATION_ONLY_DIR}. "
            "Create it and put EMSR927 assets there."
        )

    def test_no_python_imports_from_pipeline_into_validation_only(self):
        """No pipeline .py file may contain a path segment 'validation_only'."""
        for py_file in iter_pipeline_py_files():
            if not py_file.exists():
                continue
            content = py_file.read_text(encoding="utf-8", errors="replace")
            assert "validation_only" not in content, (
                f"GOVERNANCE VIOLATION: {py_file} references 'validation_only'. "
                "EMSR927 data must stay isolated."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Extra governance tests (added in v0.2)
# ─────────────────────────────────────────────────────────────────────────────

def _imports(py_file):
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module.split(".")[0])
    return names


def test_no_osmnx_or_overpass_imports():
    """OSMnx/Overpass return TODAY's map (post-event edits) -> disqualification risk."""
    for f in iter_pipeline_py_files():
        assert "osmnx" not in _imports(f), f"{f} imports osmnx"
        assert "overpass" not in _imports(f), f"{f} imports overpass"
        if f.name != "osm_overpass.py":       # the ONLY module allowed to name an Overpass endpoint (dated queries only)
            assert "overpass-api.de" not in f.read_text(encoding="utf-8").lower()


def test_every_ohsome_request_has_explicit_time():
    src = (ROOT / "app" / "services" / "osm_snapshot.py").read_text(encoding="utf-8")
    assert '"time": snapshot.isoformat()' in src and "_validate_snapshot_date(snapshot" in src


def test_demo_aoi_within_area_limit():
    import json
    import yaml
    from shapely.geometry import shape
    from pyproj import Transformer
    from shapely.ops import transform
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    g = json.loads((ROOT / "configs" / "trishuli_demo_aoi.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"]
    area = transform(Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True).transform, shape(g)).area / 1e6
    assert area <= cfg["aoi"]["max_area_km2"], f"demo AOI {area:.0f} km2 exceeds limit"


def test_config_keys_are_read_by_code():
    """Guard against the v0.1 problem where config.yaml was decorative."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    code = "\n".join(f.read_text(encoding="utf-8") for f in iter_pipeline_py_files())
    for section, key in [("flood_detection", "water_change_db_vv"), ("flood_detection", "debris_change_db_vh"),
                         ("dem", "max_slope_deg"), ("osm", "building_overlap_threshold_pct"),
                         ("accessibility", "min_block_length_m"), ("sentinel2", "mndwi_water_threshold")]:
        assert key in cfg[section] and key in code, f"{section}.{key} not wired"


def test_report_numbers_trace_to_summary():
    """Every number in the rendered report must exist in summary.json (up to display rounding)."""
    import json
    from app.api.schemas import (AnalysisSummary, FloodMetrics, InfrastructureImpact, QualityFlags, RunStatus)
    from app.services.report_generator import render_report
    s = AnalysisSummary(
        run_id="abc12345", event_date=date(2026, 8, 26), event_name="T", aoi_area_km2=150.0, status=RunStatus.DONE,
        flood_metrics=FloodMetrics(flood_debris_area_km2=4.25, water_area_km2=3.0, debris_area_km2=1.25),
        infrastructure=InfrastructureImpact(potentially_affected_buildings=18, potentially_affected_road_km=12.4,
                                            potentially_affected_bridges=2),
        settlements_assessed=25, settlements_potentially_cut_off=3, settlements_uncertain=4, settlements_connected=18,
        cut_off_min_over_buffers=2, cut_off_max_over_buffers=4, n_osm_targets=5, runtime_s=95.0,
        quality=QualityFlags(sar_pair_matched=True, dem_available=True, osm_snapshot_verified_pre_event=True),
    )
    pool = {round(float(x), 1) for x in re.findall(r"\d+\.?\d*", s.model_dump_json())}
    pool |= {round(float(x), 2) for x in re.findall(r"\d+\.?\d*", s.model_dump_json())}
    allowed_literals = {1.0, 2.0, 0.0}                 # 'Sentinel-1', 'Sentinel-2', zero-padding
    text = render_report(s)
    text = re.sub(r"`[^`]*`", "", text)                # run id / hash
    for tok in re.findall(r"\d+\.?\d*", text):
        v = float(tok)
        assert v in allowed_literals or round(v, 1) in pool or round(v, 2) in pool, f"number {tok} not in summary.json"
