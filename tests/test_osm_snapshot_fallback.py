import json
from datetime import date

from shapely.geometry import shape

from app.services import osm_snapshot as osm

AOI = {"type": "Polygon", "coordinates": [[[85.0, 28.0], [85.2, 28.0], [85.2, 28.2], [85.0, 28.2], [85.0, 28.0]]]}


class FakeResp:
    def __init__(self, code, js=None, text=""):
        self.status_code, self._js, self.text = code, js, text
    def json(self): return self._js


def _pt(x, y): return {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [x, y]}}


def test_403_on_bpolys_falls_back_to_bbox_and_clips(monkeypatch):
    seen = []
    def fake_post(url, data=None, headers=None, timeout=None):
        seen.append((dict(data), headers))
        if "bpolys" in data:
            return FakeResp(403, text="<html>403 Forbidden</html>")
        return FakeResp(200, {"type": "FeatureCollection", "features": [_pt(85.1, 28.1), _pt(86.0, 29.0)]})
    monkeypatch.setattr(osm.requests, "post", fake_post)
    res = osm._fetch(AOI, date(2026, 7, 27), date(2026, 8, 26), "place=village", "villages")
    assert len(res["features"]) == 1                          # the point outside the AOI was clipped away
    assert "bboxes" in seen[1][0] and seen[1][0]["time"] == "2026-07-27"   # still an explicit pre-event time
    assert seen[0][1]["User-Agent"].startswith("FloodLens")
    assert len(seen) == 2                                     # 403 is not retried 3x


def test_403_on_both_returns_none_not_crash(monkeypatch):
    monkeypatch.setattr(osm.requests, "post", lambda *a, **k: FakeResp(403, text="forbidden"))
    assert osm._fetch(AOI, date(2026, 7, 27), date(2026, 8, 26), "place=village", "villages") is None


def test_process_api_error_text_is_surfaced(monkeypatch, tmp_path):
    from app.services import cdse_fetch
    class R:
        ok = False; status_code = 400
        text = '{"error":{"message":"Invalid backCoeff for this DEM"}}'
        content = b""
    monkeypatch.setattr(cdse_fetch.requests, "post", lambda *a, **k: R())
    try:
        cdse_fetch._post({}, tmp_path / "x.tif", "tok")
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "Invalid backCoeff" in str(e) and "400" in str(e)
