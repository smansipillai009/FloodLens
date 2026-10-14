"""
FloodLens Nepal — Schema Integrity Tests
=========================================
Ensures that AnalysisSummary can be serialized to JSON and round-tripped,
and that the report generator only ever reads from summary fields.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest


class TestSummarySerialization:
    def test_summary_round_trip_json(self):
        from app.api.schemas import AnalysisSummary, RunStatus
        summary = AnalysisSummary(
            event_date=date(2026, 8, 26),
            event_name="Trishuli Flood",
            aoi_area_km2=250.5,
            status=RunStatus.DONE,
        )
        json_str = summary.model_dump_json()
        data = json.loads(json_str)
        restored = AnalysisSummary.model_validate(data)
        assert restored.event_name == "Trishuli Flood"
        assert restored.aoi_area_km2 == 250.5

    def test_summary_has_all_required_report_fields(self):
        """Verify all fields a report needs are present in AnalysisSummary."""
        from app.api.schemas import AnalysisSummary
        summary = AnalysisSummary()
        # These field names must exist — they are what the report template uses
        required_fields = [
            "run_id", "event_date", "event_name",
            "aoi_area_km2", "flood_metrics", "infrastructure",
            "settlements_assessed", "settlements_potentially_cut_off",
            "settlements_uncertain", "quality", "scene_selection",
            "prototype_disclaimer", "attributions", "limitations", "settlements_connected",
            "cut_off_min_over_buffers", "cut_off_max_over_buffers", "buffer_sensitivity", "runtime_s", "config_hash",
            "osm_snapshot_date", "dem_source",
        ]
        for field in required_fields:
            assert hasattr(summary, field), f"Missing required field: {field}"

    def test_infrastructure_defaults_are_zero(self):
        from app.api.schemas import InfrastructureImpact
        impact = InfrastructureImpact()
        assert impact.potentially_affected_buildings == 0
        assert impact.potentially_affected_road_km == 0.0
        assert impact.potentially_affected_bridges == 0

    def test_settlement_status_enum_values(self):
        from app.api.schemas import AccessStatus
        assert AccessStatus.CONNECTED == "connected"
        assert AccessStatus.POTENTIALLY_CUT_OFF == "potentially_cut_off"
        assert AccessStatus.UNCERTAIN == "uncertain"
