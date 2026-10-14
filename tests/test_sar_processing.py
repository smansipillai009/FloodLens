"""
FloodLens Nepal — Unit Tests for SAR Processing and Terrain Operations
========================================================================
Tests algorithmic logic on synthetic data to ensure numerical correctness.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.api.schemas import (
    AnalysisSummary,
    FloodMetrics,
    InfrastructureImpact,
    QualityFlags,
    RunStatus,
)
from app.services.flood_detection import (
    compute_flood_probability,
    compute_log_ratio,
    threshold_change_map,
    remove_small_patches,
    apply_terrain_mask,
)
from app.services.report_generator import render_report
from app.services.sentinel1 import (
    boxcar_filter,
    calibrate_to_sigma0,
    lee_filter,
    to_db,
)
from app.services.terrain import compute_slope


class TestSARLogRatio:
    def test_decrease_indicates_water(self):
        """Pre = 1.0, Post = 0.1 -> 10 * (log10(0.1) - log10(1.0)) = -10 dB."""
        pre = np.array([[1.0]], dtype=np.float32)
        post = np.array([[0.1]], dtype=np.float32)
        diff = compute_log_ratio(pre, post)
        assert not np.isnan(diff[0, 0])
        assert math.isclose(diff[0, 0], -10.0, rel_tol=1e-3)

    def test_increase_indicates_debris(self):
        """Pre = 0.1, Post = 1.0 -> +10 dB."""
        pre = np.array([[0.1]], dtype=np.float32)
        post = np.array([[1.0]], dtype=np.float32)
        diff = compute_log_ratio(pre, post)
        assert math.isclose(diff[0, 0], 10.0, rel_tol=1e-3)

    def test_nodata_zeros_produce_nan(self):
        pre = np.array([[0.0, 1.0]], dtype=np.float32)
        post = np.array([[1.0, 0.0]], dtype=np.float32)
        diff = compute_log_ratio(pre, post)
        assert np.isnan(diff[0, 0])
        assert np.isnan(diff[0, 1])


class TestSARThresholding:
    def test_threshold_decrease(self):
        change = np.array([[-3.0, -1.0, 0.0, 2.0]], dtype=np.float32)
        mask = threshold_change_map(change, threshold=2.0, direction="decrease")
        # -3.0 < -2.0 -> 1, others 0
        assert np.array_equal(mask, [[1, 0, 0, 0]])

    def test_threshold_increase(self):
        change = np.array([[-3.0, 1.0, 2.5, 4.0]], dtype=np.float32)
        mask = threshold_change_map(change, threshold=2.0, direction="increase")
        # 2.5 > 2.0 and 4.0 > 2.0 -> 1
        assert np.array_equal(mask, [[0, 0, 1, 1]])

    def test_apply_terrain_mask(self):
        candidates = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        terrain_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)  # 0 = excluded
        result = apply_terrain_mask(candidates, terrain_mask)
        assert np.array_equal(result, [[1, 0], [0, 1]])

    def test_remove_small_patches(self):
        # 1 isolated pixel vs a 2x2 patch
        mask = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        # Pixel area = 100 m^2 (10m x 10m). If min_area = 300 m^2, 1 pixel (100 m^2) gets filtered
        cleaned = remove_small_patches(mask, pixel_area_m2=100.0, min_area_m2=300.0)
        assert cleaned[0, 0] == 0
        assert np.all(cleaned[1:3, 2:4] == 1)


class TestTerrainSlope:
    def test_flat_terrain_has_zero_slope(self, tmp_path: Path):
        dem_path = tmp_path / "flat_dem.tif"
        slope_path = tmp_path / "slope.tif"
        data = np.full((10, 10), 1000.0, dtype=np.float32)
        transform = from_origin(100000, 100000, 30.0, 30.0)
        with rasterio.open(
            dem_path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float32", crs="EPSG:32645", transform=transform, nodata=-9999.0
        ) as dst:
            dst.write(data, 1)

        compute_slope(dem_path, slope_path)
        with rasterio.open(slope_path) as src:
            slope = src.read(1)
            # Internal pixels (ignoring edge padding) should have 0 slope
            assert np.allclose(slope[1:-1, 1:-1], 0.0, atol=1e-2)


class TestReportRendering:
    def test_render_report_with_valid_summary(self):
        summary = AnalysisSummary(
            run_id="test-run-1234",
            event_name="Trishuli Flood",
            event_date=date(2026, 8, 26),
            aoi_area_km2=150.0,
            status=RunStatus.DONE,
            flood_metrics=FloodMetrics(
                flood_debris_area_km2=4.25,
                water_area_km2=3.0,
                debris_area_km2=1.25,
                water_pixel_count=3500,
                debris_pixel_count=750,
            ),
            infrastructure=InfrastructureImpact(
                potentially_affected_buildings=18,
                potentially_affected_road_km=12.4,
                potentially_affected_bridges=2,
            ),
            settlements_assessed=25,
            settlements_potentially_cut_off=3,
            settlements_uncertain=2,
            quality=QualityFlags(
                sar_pair_matched=True,
                dem_available=True,
                osm_snapshot_verified_pre_event=True,
            ),
        )
        rendered = render_report(summary)
        assert "SITUATION REPORT" in rendered
        assert "test-run-1234" in rendered
        assert "4.25 km²" in rendered
        assert "18" in rendered
        assert "12.4 km" in rendered
        assert "3" in rendered  # cut off
        assert "Educational prototype" in rendered
        assert "Contains modified Copernicus Sentinel data" in rendered


class TestCalibrationAndSpeckle:
    def test_explicit_units_db(self):
        out = calibrate_to_sigma0(np.array([[-20.0, 0.0]], dtype=np.float32), "db")
        assert math.isclose(out[0, 0], 0.01, rel_tol=1e-4) and out[0, 1] == 0

    def test_explicit_units_linear_passthrough(self):
        out = calibrate_to_sigma0(np.array([[0.05, -1.0]], dtype=np.float32), "linear_sigma0")
        assert math.isclose(out[0, 0], 0.05, rel_tol=1e-5) and out[0, 1] == 0

    def test_unknown_units_raise(self):
        with pytest.raises(ValueError):
            calibrate_to_sigma0(np.ones((2, 2)), "nonsense")

    def test_lee_reduces_variance_keeps_nodata_and_mean(self):
        rng = np.random.default_rng(0)
        img = rng.gamma(4.4, 0.1 / 4.4, (80, 80)).astype(np.float32)
        img[:5] = 0
        f = lee_filter(img, 5)
        assert (f[:5] == 0).all()
        assert f[10:].std() < img[10:].std()
        assert abs(f[10:].mean() - img[10:].mean()) < 0.01

    def test_boxcar_keeps_nodata(self):
        img = np.ones((10, 10), dtype=np.float32); img[0] = 0
        assert (boxcar_filter(img, 3)[0] == 0).all()


class TestTerrainGeometry:
    def _dem(self, tmp_path, fn):
        y, x = np.mgrid[0:9, 0:9]
        data = fn(x, y).astype(np.float32)
        p = tmp_path / "dem.tif"
        with rasterio.open(p, "w", driver="GTiff", height=9, width=9, count=1, dtype="float32",
                           crs="EPSG:32645", transform=from_origin(100000, 100000, 30.0, 30.0), nodata=-9999.0) as d:
            d.write(data, 1)
        return p

    def test_slope_and_aspect_of_east_rising_plane(self, tmp_path):
        from app.services.terrain import compute_slope
        dem = self._dem(tmp_path, lambda x, y: 0.5 * x * 30.0)          # rises to the east, faces WEST
        compute_slope(dem, tmp_path / "s.tif", tmp_path / "a.tif")
        with rasterio.open(tmp_path / "s.tif") as s, rasterio.open(tmp_path / "a.tif") as a:
            assert math.isclose(float(s.read(1)[4, 4]), math.degrees(math.atan(0.5)), abs_tol=0.05)
            assert math.isclose(float(a.read(1)[4, 4]), 270.0, abs_tol=0.5)

    def test_aspect_north_rising_faces_south(self, tmp_path):
        from app.services.terrain import compute_slope
        dem = self._dem(tmp_path, lambda x, y: (8 - y) * 0.5 * 30.0)    # rows run south, so z grows northwards
        compute_slope(dem, tmp_path / "s.tif", tmp_path / "a.tif")
        with rasterio.open(tmp_path / "a.tif") as a:
            assert math.isclose(float(a.read(1)[4, 4]), 180.0, abs_tol=0.5)

    def test_layover_when_steep_slope_faces_sensor(self, tmp_path):
        from app.services.terrain import compute_layover_shadow_mask, compute_slope
        dem = self._dem(tmp_path, lambda x, y: 1.0 * x * 30.0)          # 45 deg, faces west
        compute_slope(dem, tmp_path / "s.tif", tmp_path / "a.tif")
        # ascending, right-looking: sensor looks east, so west-facing slopes face the sensor
        compute_layover_shadow_mask(tmp_path / "s.tif", tmp_path / "a.tif", tmp_path / "ls.tif", 39.0, 80.0)
        with rasterio.open(tmp_path / "ls.tif") as m:
            assert m.read(1)[4, 4] == 0
        # descending (looks west): the same slope faces away and is NOT layover
        compute_layover_shadow_mask(tmp_path / "s.tif", tmp_path / "a.tif", tmp_path / "ls2.tif", 39.0, 280.0)
        with rasterio.open(tmp_path / "ls2.tif") as m:
            assert m.read(1)[4, 4] == 1

    def test_dem_tile_name_has_00_suffixes(self):
        from app.services.terrain import _dem_tile_name
        assert "N27_00_E085_00_DEM" in _dem_tile_name(27, 85)
