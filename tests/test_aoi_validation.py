import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

POLY = {"type": "Polygon", "coordinates": [[[85.3, 28.0], [85.35, 28.0], [85.35, 28.05], [85.3, 28.05], [85.3, 28.0]]]}
POINT = {"type": "Point", "coordinates": [85.3, 28.0]}


def _write(tmp_path, features):
    p = tmp_path / "a.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection",
                             "features": [{"type": "Feature", "properties": {}, "geometry": g} for g in features]}), encoding="utf-8")
    return str(p)


def test_load_aoi_skips_marker_and_takes_first_polygon(tmp_path):
    import run
    assert run.load_aoi(_write(tmp_path, [POINT, POLY]))["type"] == "Polygon"


def test_load_aoi_with_only_a_point_exits_cleanly(tmp_path):
    import run
    with pytest.raises(SystemExit):
        run.load_aoi(_write(tmp_path, [POINT]))


def test_degenerate_aoi_is_rejected_before_calling_the_api():
    from app.services.cdse_fetch import _size_for
    tiny = {"type": "Polygon", "coordinates": [[[85.3, 28.0], [85.30001, 28.0], [85.30001, 28.00001], [85.3, 28.00001], [85.3, 28.0]]]}
    with pytest.raises(ValueError, match="smaller than one"):
        _size_for(tiny, 10, 2500)
    assert _size_for(POLY, 10, 2500)[0] > 100


def test_min_area_validation_message():
    from app.config import get_config
    from app.services.orchestrator import validate_aoi_size
    with pytest.raises(ValueError, match="POLYGON"):
        validate_aoi_size(0.0, get_config())
    validate_aoi_size(60.0, get_config())     # normal AOI passes silently


def test_diagnose_ohsome_command_is_registered():
    import run
    assert callable(run.cmd_diagnose_ohsome)


def test_detection_funnel_is_written_even_when_nothing_is_found(tmp_path):
    import numpy as np, rasterio, json
    from rasterio.transform import from_origin
    from app.services.flood_detection import detect_flood
    rng = np.random.default_rng(0)
    base = rng.gamma(4, 0.05, (60, 60)).astype("float32")
    paths = {}
    for n in ("bvv", "bvh", "avv", "avh"):
        pth = tmp_path / f"{n}.tif"
        with rasterio.open(pth, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32", crs="EPSG:32645",
                           transform=from_origin(0, 600, 10, 10), nodata=0.0) as d:
            d.write(base, 1)
        paths[n] = pth
    res = detect_flood(paths["bvv"], paths["bvh"], paths["avv"], paths["avh"], tmp_path)   # identical images -> no change
    f = json.loads((tmp_path / "detection_funnel.json").read_text(encoding="utf-8"))
    assert res["water_pixel_count"] == 0 and f["valid_pixels"] == 3600
    assert f["water"]["1_vv_drop_enough"] == 0 and "log_ratio_vv_db" in f["stats_db"]


def test_proxy_hand_uses_its_own_looser_limits(tmp_path):
    """Regression for run 75683f5b: real-HAND limits applied to a local-relief proxy removed 99.5% of candidates."""
    import json
    import numpy as np, rasterio
    from rasterio.transform import from_origin
    from app.services.flood_detection import detect_flood
    n = 80
    rng = np.random.default_rng(1)
    base = rng.gamma(20, 0.02, (n, n)).astype("float32")           # ~ -4 dB, low speckle
    post_vv = base.copy(); post_vv[20:50, 20:50] *= 0.01          # -20 dB drop -> dark water
    def w(name, a, dtype="float32", nodata=0.0):
        p = tmp_path / name
        with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype, crs="EPSG:32645",
                           transform=from_origin(0, n * 10, 10, 10), nodata=nodata) as d:
            d.write(a.astype(dtype), 1)
        return p
    hand = np.full((n, n), 45.0, "float32")                         # 45 m: fails true-HAND (15 m) but passes the proxy limit (60 m)
    terrain = {"hand": w("hand.tif", hand, nodata=-9999.0), "terrain_mask": None, "hand_method": "local_relief_proxy"}
    res = detect_flood(w("a.tif", base), w("b.tif", base), w("c.tif", post_vv), w("d.tif", base), tmp_path, terrain=terrain)
    f = json.loads((tmp_path / "detection_funnel.json").read_text(encoding="utf-8"))
    assert f["hand"]["limit_water_m"] == 60.0 and f["hand_at_candidates_m"]["water"]["p50"] == 45.0
    assert res["water_pixel_count"] > 0
    terrain["hand_method"] = "pysheds_d8_hand"                      # real HAND: strict limit applies again
    res2 = detect_flood(w("a.tif", base), w("b.tif", base), w("c.tif", post_vv), w("d.tif", base), tmp_path, terrain=terrain)
    assert res2["water_pixel_count"] == 0
