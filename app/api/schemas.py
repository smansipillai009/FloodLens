"""
FloodLens Nepal — Central Data Schemas
=======================================
All Pydantic models that flow through the pipeline live here.
Every API input, pipeline output, and result field is defined in this file.
Nothing in the pipeline may invent a number that is not in these models.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class OrbitDirection(str, Enum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


class ModelMode(str, Enum):
    BASELINE = "baseline"
    AI = "ai"
    ENSEMBLE = "ensemble"


class Language(str, Enum):
    EN = "en"
    NE = "ne"


class AccessStatus(str, Enum):
    CONNECTED = "connected"
    POTENTIALLY_CUT_OFF = "potentially_cut_off"
    UNCERTAIN = "uncertain"


class FloodConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"   # ran, but a mandatory input was missing (e.g. no same-track S1 pair)
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Input request to start a FloodLens analysis run."""
    aoi_geojson: dict[str, Any] = Field(
        ...,
        description="GeoJSON Polygon or MultiPolygon defining the Area of Interest."
    )
    event_date: date = Field(
        ...,
        description="Date of the flood event (YYYY-MM-DD)."
    )
    event_name: Optional[str] = Field(
        None,
        description="Human-readable name for the event (e.g. 'Trishuli Flood Aug 2026')."
    )
    before_window_days: int = Field(
        default=18,
        ge=1,
        le=60,
        description="Days before event_date to search for a pre-event S1 scene."
    )
    after_window_days: int = Field(
        default=18,
        ge=1,
        le=60,
        description="Days after event_date to search for a post-event S1 scene."
    )
    model_mode: ModelMode = Field(
        default=ModelMode.BASELINE,
        description="Which flood detection mode to use."
    )
    use_optical_confirmation: bool = Field(
        default=True,
        description="Use clear-pixel Sentinel-2 to confirm/contradict SAR candidates. "
                    "No cloud reconstruction is ever used."
    )
    buffers_m: list[float] = Field(
        default_factory=lambda: [0.0, 30.0, 60.0],
        description="Hazard buffer sensitivity scenarios (metres)."
    )
    language: Language = Field(
        default=Language.EN,
        description="Language for the situation report."
    )
    flood_threshold: Optional[float] = Field(
        default=None,
        description="Manual log-ratio threshold override; None = automatic."
    )

    @model_validator(mode="after")
    def event_date_not_future(self) -> RunRequest:
        if self.event_date > date.today():
            raise ValueError("event_date cannot be in the future.")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Scene selection
# ─────────────────────────────────────────────────────────────────────────────

class S1SceneInfo(BaseModel):
    scene_id: str
    datetime: datetime
    relative_orbit: int
    orbit_direction: OrbitDirection
    polarizations: list[str]
    cloud_cover_pct: float = 0.0   # SAR is always 0 but kept for schema consistency
    platform: str = "Sentinel-1"
    download_url: Optional[str] = None


class S2SceneInfo(BaseModel):
    scene_id: str
    datetime: datetime
    cloud_cover_pct: float
    platform: str = "Sentinel-2"
    download_url: Optional[str] = None


class SceneSelectionResult(BaseModel):
    """Output of scene_selector: the chosen S1 before/after pair and optional S2."""
    s1_before: Optional[S1SceneInfo] = None
    s1_after: Optional[S1SceneInfo] = None
    same_track: bool = False
    s1_before_extra: list[S1SceneInfo] = Field(default_factory=list)  # extra same-track pre-event scenes (median reference)
    s2_after: Optional[S2SceneInfo] = None
    osm_snapshot_date: Optional[date] = None   # always < event_date
    candidates_before: list[S1SceneInfo] = Field(default_factory=list)  # shown when no pair exists
    candidates_after: list[S1SceneInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_same_track(self) -> SceneSelectionResult:
        """Enforce that if both S1 scenes are present, they share orbit + direction."""
        if self.s1_before and self.s1_after:
            self.s1_before_extra = [
                e for e in self.s1_before_extra
                if e.relative_orbit == self.s1_before.relative_orbit
                and e.orbit_direction == self.s1_before.orbit_direction
            ]
            if (self.s1_before.relative_orbit != self.s1_after.relative_orbit or
                    self.s1_before.orbit_direction != self.s1_after.orbit_direction):
                self.same_track = False
                msg = ("S1 before/after scenes are NOT on the same relative orbit. "
                       "Change detection reliability is severely degraded.")
                if msg not in self.warnings:
                    self.warnings.append(msg)
            else:
                self.same_track = True
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Quality flags
# ─────────────────────────────────────────────────────────────────────────────

class QualityFlags(BaseModel):
    sar_pair_matched: bool = False
    sar_alignment_rmse_m: Optional[float] = None   # MEASURED (phase correlation), never assumed
    optical_cloud_pct_in_aoi: Optional[float] = None
    optical_agreement_pct: Optional[float] = None  # % of SAR candidate px (clear S2) that S2 confirms
    detection_method_used: str = "none"
    flood_confidence: FloodConfidence = FloodConfidence.LOW
    dem_available: bool = False
    osm_snapshot_verified_pre_event: bool = False
    warnings: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Impact metrics
# ─────────────────────────────────────────────────────────────────────────────

class InfrastructureImpact(BaseModel):
    """Counts of potentially affected OSM features."""
    potentially_affected_buildings: int = 0
    potentially_affected_road_km: float = 0.0
    potentially_affected_bridges: int = 0
    # File references (relative to run output dir)
    impacted_buildings_file: Optional[str] = None
    impacted_roads_file: Optional[str] = None
    impacted_bridges_file: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Accessibility
# ─────────────────────────────────────────────────────────────────────────────

class SettlementAccessResult(BaseModel):
    """Per-settlement road-connectivity result."""
    settlement_id: str
    settlement_name: Optional[str] = None
    pre_event_reachable_to_hospital: Optional[bool] = None
    hazard_affected_reachable_to_hospital: Optional[bool] = None
    pre_event_distance_km: Optional[float] = None
    hazard_affected_distance_km: Optional[float] = None
    status: AccessStatus = AccessStatus.UNCERTAIN
    evidence_blocked_edge_ids: list[str] = Field(default_factory=list)
    confidence: FloodConfidence = FloodConfidence.LOW
    hazard_buffer_m: float = 0.0  # buffer used for the headline distances (0 = strict mask)
    status_by_buffer: dict[str, str] = Field(default_factory=dict)  # e.g. {"0": "connected", "30": "potentially_cut_off"}
    snap_distance_m: Optional[float] = None
    reason: Optional[str] = None  # why 'uncertain' (snapping_failed / no_targets / baseline_unreachable / unstable_across_buffers)


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis summary — the ONLY source for report generation
# ─────────────────────────────────────────────────────────────────────────────

class FloodMetrics(BaseModel):
    """All fields are computed AFTER terrain + size filtering (consistent with the mask)."""
    flood_debris_area_km2: float = 0.0      # union of water + debris
    water_area_km2: float = 0.0
    debris_area_km2: float = 0.0
    water_pixel_count: int = 0
    debris_pixel_count: int = 0
    optical_confirmed_km2: float = 0.0


class BufferSensitivity(BaseModel):
    buffer_m: float
    buildings: int = 0
    road_km: float = 0.0
    bridges: int = 0
    cut_off: int = 0
    uncertain: int = 0
    connected: int = 0


REQUIRED_ATTRIBUTIONS = [
    "Contains modified Copernicus Sentinel data 2026.",
    ("Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus Defence and Space GmbH "
     "2014\u20132018 provided under COPERNICUS by the European Union and ESA; all rights reserved."),
    "\u00a9 OpenStreetMap contributors.",
    "Training data: Kuro Siwo \u2014 Bountos et al., 2024 (NeurIPS Datasets & Benchmarks). "
    "Sen1Floods11 (if used) \u2014 Bonafilia et al., CVPR Workshops 2020.",
    "European Union, Copernicus Emergency Management Service data (used for comparison only, never as an input).",
]


class AnalysisSummary(BaseModel):
    """
    Machine-readable summary of a completed FloodLens run.
    This is the SOLE input to report_generator.py.
    No number may appear in the report that is not a field here.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: RunStatus = RunStatus.QUEUED

    # ── Request echo ───────────────────────────────────────────────
    aoi_area_km2: float = 0.0
    event_date: Optional[date] = None
    event_name: Optional[str] = None

    # ── Data provenance ────────────────────────────────────────────
    scene_selection: Optional[SceneSelectionResult] = None
    osm_snapshot_date: Optional[date] = None
    dem_source: str = "Copernicus DEM GLO-30"

    # ── Quality ────────────────────────────────────────────────────
    quality: QualityFlags = Field(default_factory=QualityFlags)

    # ── Results ────────────────────────────────────────────────────
    flood_metrics: FloodMetrics = Field(default_factory=FloodMetrics)
    infrastructure: InfrastructureImpact = Field(default_factory=InfrastructureImpact)
    settlements_assessed: int = 0
    settlements_potentially_cut_off: int = 0
    settlements_uncertain: int = 0
    settlements_connected: int = 0
    cut_off_min_over_buffers: int = 0
    cut_off_max_over_buffers: int = 0
    buffer_sensitivity: list[BufferSensitivity] = Field(default_factory=list)
    settlement_results: list[SettlementAccessResult] = Field(default_factory=list)
    n_osm_targets: int = 0   # hospitals/clinics/towns used as destinations

    # ── Reproducibility ────────────────────────────────────────────
    runtime_s: float = 0.0
    config_hash: Optional[str] = None
    config_frozen: bool = False   # True if config hash == configs/config.lock
    floodlens_version: Optional[str] = None

    # ── Methods ────────────────────────────────────────────────────
    flood_detection_method: str = "none"
    model_mode: ModelMode = ModelMode.BASELINE

    # ── Attributions ───────────────────────────────────────────────
    attributions: list[str] = Field(default_factory=lambda: list(REQUIRED_ATTRIBUTIONS))
    limitations: list[str] = Field(default_factory=list)
    prototype_disclaimer: str = (
        "Educational prototype — not operational emergency guidance. "
        "All results require field verification."
    )


# ─────────────────────────────────────────────────────────────────────────────
# API response models
# ─────────────────────────────────────────────────────────────────────────────

class RunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus
    warnings: list[str] = Field(default_factory=list)


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    stage: Optional[str] = None
    progress_pct: int = 0
    scene_selection: Optional[SceneSelectionResult] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
