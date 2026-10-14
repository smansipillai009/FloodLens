from datetime import date, datetime, timezone

from app.api.schemas import OrbitDirection, S1SceneInfo
from app.services.scene_selector import find_best_s1_pair

EV = date(2026, 8, 26)


def sc(i, day, orbit, d=OrbitDirection.ASCENDING):
    return S1SceneInfo(scene_id=i, datetime=datetime(2026, 8, day, 0, 0, tzinfo=timezone.utc),
                       relative_orbit=orbit, orbit_direction=d, polarizations=["VV", "VH"])


def test_picks_same_track_pair_and_extra_references():
    before = [sc("b1", 2, 12), sc("b2", 14, 12), sc("bx", 20, 99)]
    after = [sc("a1", 27, 77), sc("a2", 27, 12)]
    b, a, extra, w = find_best_s1_pair(before, after, EV)
    assert (b.scene_id, a.scene_id) == ("b2", "a2")
    assert [e.scene_id for e in extra] == ["b1"] and not w


def test_never_returns_mismatched_pair():
    b, a, extra, w = find_best_s1_pair([sc("b", 14, 12)], [sc("a", 27, 99)], EV)
    assert b is None and a is None and extra == []
    assert "NO same-track" in w[0]


def test_direction_must_match_too():
    b, a, _, _ = find_best_s1_pair([sc("b", 14, 12)], [sc("a", 27, 12, OrbitDirection.DESCENDING)], EV)
    assert b is None


def test_prefers_smallest_gap():
    before = [sc("far", 2, 12), sc("near", 20, 12)]
    after = [sc("late", 27, 12)]
    assert find_best_s1_pair(before, after, EV)[0].scene_id == "near"


def test_warns_when_after_scene_is_on_event_day():
    b, a, _, w = find_best_s1_pair([sc("b", 14, 12)], [sc("a", 26, 12)], EV)
    assert a.scene_id == "a" and any("event date" in x for x in w)


def test_confidence_not_high_without_measured_optical_agreement():
    from app.api.schemas import FloodConfidence, QualityFlags
    from app.services.orchestrator import _confidence
    q = QualityFlags(sar_pair_matched=True, dem_available=True, osm_snapshot_verified_pre_event=True)
    assert _confidence(q) == FloodConfidence.MEDIUM
    q.optical_agreement_pct = 75.0
    assert _confidence(q) == FloodConfidence.HIGH


def test_osm_context_margin_grows_aoi():
    from shapely.geometry import shape
    from app.services.osm_snapshot import _expand_aoi
    aoi = {"type": "Polygon", "coordinates": [[[85, 28], [85.1, 28], [85.1, 28.1], [85, 28.1], [85, 28]]]}
    assert shape(_expand_aoi(aoi, 10)).area > shape(aoi).area * 4
    assert _expand_aoi(aoi, 0) == aoi


def test_fetch_test_dem_url_is_not_doubled():
    from app.services.terrain import COP_DEM_S3_BASE, _dem_tile_name
    url = f"{COP_DEM_S3_BASE}/{_dem_tile_name(27, 85)}"
    assert url == ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E085_00_DEM/"
                   "Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif")
    import inspect, run
    assert "{name}/{name}.tif" not in inspect.getsource(run)


def test_stac_datetime_end_day_is_inclusive(monkeypatch):
    from app.services import scene_selector as ss
    sent = {}
    class R:
        def raise_for_status(self): pass
        def json(self): return {"features": []}
    monkeypatch.setattr(ss.requests, "post", lambda url, json=None, timeout=None: (sent.update(json), R())[1])
    ss._search_stac("sentinel-1-grd", [85, 28, 85.1, 28.1], date(2026, 8, 1), date(2026, 8, 3))
    assert sent["datetime"] == "2026-08-01T00:00:00Z/2026-08-04T00:00:00Z"


def test_config_uses_current_cdse_stac():
    from app.config import get_config
    assert get_config()["scene_selection"]["stac_url"].startswith("https://stac.dataspace.copernicus.eu")


def test_ohsome_metadata_retries_then_falls_back(monkeypatch):
    from app.services import scene_selector as ss
    calls = {"n": 0}
    class R:
        status_code = 503; reason = "Service Unavailable"
        def raise_for_status(self): raise ss.requests.exceptions.HTTPError("503")
    def fake_get(url, timeout=None):
        calls["n"] += 1; return R()
    monkeypatch.setattr(ss.requests, "get", fake_get)
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    snap, warns = ss.choose_osm_snapshot_date(date(2026, 8, 26))
    assert calls["n"] == ss.METADATA_RETRIES
    # event after the assumed latest data (27 Jul): snapshot is capped at 27 Jul, still strictly pre-event
    assert snap == date(2026, 7, 27) and any("unreachable" in w for w in warns)
    snap2, _ = ss.choose_osm_snapshot_date(date(2026, 7, 20))       # event earlier than the cap: day before the event
    assert snap2 == date(2026, 7, 19)


def test_parse_s1_item_reads_new_cdse_product_type():
    from app.config import get_config
    from app.services.scene_selector import _parse_s1_item
    cfgs = get_config()["scene_selection"]
    base = {"id": "S1A_X", "properties": {"datetime": "2026-08-20T00:12:00Z", "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"], "sat:relative_orbit": 12, "sat:orbit_state": "ascending"}}
    slc = {**base, "properties": {**base["properties"], "product:type": "IW_SLC__1S"}}
    grd = {**base, "properties": {**base["properties"], "product:type": "IW_GRDH_1S"}}
    assert _parse_s1_item(slc, cfgs) is None
    assert _parse_s1_item(grd, cfgs) is not None
