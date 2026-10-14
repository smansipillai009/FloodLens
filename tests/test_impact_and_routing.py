"""
FloodLens Nepal — Unit Tests for Impact Overlay & Settlement Cut-Off Routing
=============================================================================
Tests deterministic geometry overlays and NetworkX G0/G1 differential routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point, Polygon, mapping

from app.api.schemas import AccessStatus
from app.services.accessibility import (
    analyze_accessibility,
    build_road_graph,
    snap_point_to_graph,
)
from app.services.impact_analysis import analyze_infrastructure_impact

CRS_METRIC = {"type": "name", "properties": {"name": "EPSG:32645"}}


class TestInfrastructureImpact:
    def test_building_overlap_threshold(self, tmp_path: Path):
        # 100m x 100m building at origin (10,000 m^2) in EPSG:32645
        bld_poly = Polygon([(340000, 3100000), (340100, 3100000), (340100, 3100100), (340000, 3100100)])
        bld_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{
                "type": "Feature",
                "id": "way/123",
                "properties": {"building": "yes"},
                "geometry": mapping(bld_poly),
            }],
        }
        # Flood overlaps 20m x 100m = 2,000 m^2 (20% overlap)
        flood_poly = Polygon([(340000, 3100000), (340020, 3100000), (340020, 3100100), (340000, 3100100)])
        flood_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{
                "type": "Feature",
                "properties": {"label": "flood"},
                "geometry": mapping(flood_poly),
            }],
        }

        # Test threshold = 10% (should be flagged)
        impact, details = analyze_infrastructure_impact(
            flood_polygons=flood_fc,
            osm_buildings=bld_fc,
            osm_roads=None,
            osm_bridges=None,
            run_dir=tmp_path,
            building_overlap_pct=10.0,
        )
        assert impact.potentially_affected_buildings == 1

        # Test threshold = 30% (should NOT be flagged)
        impact30, _ = analyze_infrastructure_impact(
            flood_polygons=flood_fc,
            osm_buildings=bld_fc,
            osm_roads=None,
            osm_bridges=None,
            run_dir=tmp_path,
            building_overlap_pct=30.0,
        )
        assert impact30.potentially_affected_buildings == 0

    def test_road_intersection_length(self, tmp_path: Path):
        # Road from (340000, 3100050) to (340500, 3100050) (500m length)
        road_line = LineString([(340000, 3100050), (340500, 3100050)])
        road_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{
                "type": "Feature",
                "id": "way/road1",
                "properties": {"highway": "primary"},
                "geometry": mapping(road_line),
            }],
        }
        # Flood from x=340100 to x=340200 (100m cut)
        flood_poly = Polygon([(340100, 3100000), (340200, 3100000), (340200, 3100100), (340100, 3100100)])
        flood_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{
                "type": "Feature",
                "properties": {"label": "flood"},
                "geometry": mapping(flood_poly),
            }],
        }

        impact, details = analyze_infrastructure_impact(
            flood_polygons=flood_fc,
            osm_buildings=None,
            osm_roads=road_fc,
            osm_bridges=None,
            run_dir=tmp_path,
        )
        # 100m = 0.1 km
        assert impact.potentially_affected_road_km == 0.1
        assert details["buffers"][0]["buffer_m"] == 0.0
        assert details["buffers"][0]["road_km"] == 0.1


class TestSettlementRouting:
    def test_settlement_cut_off_when_road_severed(self, tmp_path: Path):
        """
        Scenario:
        Town/Hospital at (340000, 3100000).
        Road 1: (340000, 3100000) -> (341000, 3100000)
        Road 2: (341000, 3100000) -> (342000, 3100000)
        Settlement at (342000, 3100000).
        Flood severs Road 1 at x=340500.
        Result: Settlement must be 'potentially_cut_off'.
        """
        road1 = LineString([(340000, 3100000), (341000, 3100000)])
        road2 = LineString([(341000, 3100000), (342000, 3100000)])
        roads_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [
                {"type": "Feature", "id": "r1", "properties": {"highway": "primary"}, "geometry": mapping(road1)},
                {"type": "Feature", "id": "r2", "properties": {"highway": "primary"}, "geometry": mapping(road2)},
            ],
        }
        hospital_pt = Point(340000, 3100000)
        health_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{"type": "Feature", "id": "hosp1", "properties": {"amenity": "hospital"}, "geometry": mapping(hospital_pt)}],
        }
        settlement_pt = Point(342000, 3100000)
        settlements_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{"type": "Feature", "id": "s1", "properties": {"name": "Isolated Village"}, "geometry": mapping(settlement_pt)}],
        }
        # Flood across x=340400 to x=340600 (blocks r1)
        flood_poly = Polygon([(340400, 3099900), (340600, 3099900), (340600, 3100100), (340400, 3100100)])
        flood_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{"type": "Feature", "properties": {"label": "flood"}, "geometry": mapping(flood_poly)}],
        }

        results, summary, geojson_path = analyze_accessibility(
            roads_data=roads_fc,
            settlements_data=settlements_fc,
            health_data=health_fc,
            flood_polygons_data=flood_fc,
            run_dir=tmp_path,
            buffer_m_list=[0.0],
        )

        assert summary["assessed"] == 1
        assert summary["cut_off"] == 1
        assert results[0].status == AccessStatus.POTENTIALLY_CUT_OFF
        assert results[0].pre_event_reachable_to_hospital is True
        assert results[0].hazard_affected_reachable_to_hospital is False
        assert geojson_path.exists()

    def test_baseline_disconnected_is_marked_uncertain(self, tmp_path: Path):
        """
        If a settlement had no road to hospital at baseline (G0),
        it must be classified as 'uncertain', NOT 'potentially_cut_off'.
        """
        # Road 1: (340000, 3100000) -> (340500, 3100000) (hospital is on Road 1)
        road1 = LineString([(340000, 3100000), (340500, 3100000)])
        # Road 2: (345000, 3100000) -> (346000, 3100000) (isolated island of road)
        road2 = LineString([(345000, 3100000), (346000, 3100000)])
        roads_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [
                {"type": "Feature", "id": "r1", "properties": {"highway": "primary"}, "geometry": mapping(road1)},
                {"type": "Feature", "id": "r2", "properties": {"highway": "primary"}, "geometry": mapping(road2)},
            ],
        }
        health_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{"type": "Feature", "id": "h1", "properties": {"amenity": "hospital"}, "geometry": mapping(Point(340000, 3100000))}],
        }
        settlements_fc = {
            "type": "FeatureCollection",
            "crs": CRS_METRIC,
            "features": [{"type": "Feature", "id": "s1", "properties": {"name": "Remote Hamlet"}, "geometry": mapping(Point(346000, 3100000))}],
        }
        flood_fc = {"type": "FeatureCollection", "crs": CRS_METRIC, "features": []}

        results, summary, _ = analyze_accessibility(
            roads_data=roads_fc,
            settlements_data=settlements_fc,
            health_data=health_fc,
            flood_polygons_data=flood_fc,
            run_dir=tmp_path,
        )

        assert summary["assessed"] == 1
        assert summary["uncertain"] == 1
        assert summary["cut_off"] == 0
        assert results[0].status == AccessStatus.UNCERTAIN

