from datetime import date

import pytest

from app.services import osm_overpass as op

AOI = {"type": "Polygon", "coordinates": [[[85.0, 28.0], [85.5, 28.0], [85.5, 28.4], [85.0, 28.4], [85.0, 28.0]]]}
EV, SNAP = date(2026, 8, 26), date(2026, 7, 27)


def test_tiles_cover_the_bbox_without_gaps_and_respect_the_max_size():
    ts = op.tiles((85.0, 28.0, 85.5, 28.4), 0.15)
    assert len(ts) == 4 * 3
    assert min(t[0] for t in ts) == 85.0 and max(t[2] for t in ts) == 85.5 and min(t[1] for t in ts) == 28.0 and max(t[3] for t in ts) == 28.4
    assert all(t[2] - t[0] <= 0.15 + 1e-9 and t[3] - t[1] <= 0.15 + 1e-9 for t in ts)
    assert len(op.tiles((85.0, 28.0, 85.1, 28.1), 0.15)) == 1


class R:
    def __init__(self, js, code=200): self.status_code, self._js, self.text = code, js, "x"
    def json(self): return self._js


def _way(i):
    return {"type": "way", "id": i, "tags": {"highway": "track"}, "geometry": [{"lat": 28.1, "lon": 85.1}, {"lat": 28.1, "lon": 85.2}]}


def test_tiles_are_merged_and_features_crossing_borders_are_deduplicated(monkeypatch):
    monkeypatch.setattr(op.time, "sleep", lambda s: None)
    calls = []
    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(data["data"])
        return R({"elements": [_way(1), _way(len(calls) + 100)]})          # way 1 appears in EVERY tile; the others are unique
    monkeypatch.setattr(op.requests, "post", fake_post)
    res = op.fetch_layer("roads", AOI, SNAP, EV)
    ids = sorted(f["properties"]["@osmId"] for f in res["features"])
    assert len(calls) == 12 and ids.count("way/1") == 1 and len(ids) == 13
    assert all('[date:"2026-07-27T00:00:00Z"]' in q for q in calls)      # every tile query is dated


def test_one_failed_tile_fails_the_whole_layer_and_says_which(monkeypatch):
    monkeypatch.setattr(op.time, "sleep", lambda s: None)
    n = {"i": 0}
    def fake_post(url, data=None, headers=None, timeout=None):
        n["i"] += 1
        if n["i"] >= 3:
            raise op.requests.exceptions.SSLError("boom")
        return R({"elements": []})
    monkeypatch.setattr(op.requests, "post", fake_post)
    assert op.fetch_layer("roads", AOI, SNAP, EV) is None
    assert op.LAST_ERROR["roads"].startswith("tile 3/12") and "SSLError" in op.LAST_ERROR["roads"]
