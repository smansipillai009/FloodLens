from datetime import date

import pytest

from app.services import osm_overpass as op

AOI = {"type": "Polygon", "coordinates": [[[85.0, 28.0], [85.2, 28.0], [85.2, 28.2], [85.0, 28.2], [85.0, 28.0]]]}
EV, SNAP = date(2026, 8, 26), date(2026, 7, 27)


@pytest.mark.parametrize("kind", ["buildings", "roads", "bridges", "settlements", "towns", "health"])
def test_every_overpass_query_carries_the_attic_date(kind):
    q = op.build_query(kind, (85.0, 28.0, 85.2, 28.2), SNAP, EV)
    assert q.startswith('[out:json]') and '[date:"2026-07-27T00:00:00Z"]' in q and q.endswith("out geom;")


def test_snapshot_on_or_after_event_is_refused():
    for s in (EV, date(2026, 9, 1)):
        with pytest.raises(ValueError, match="STRICTLY BEFORE"):
            op.build_query("roads", (85, 28, 85.2, 28.2), s, EV)


def test_conversion_to_ohsome_like_geojson():
    js = {"elements": [
        {"type": "way", "id": 1, "tags": {"building": "yes"}, "geometry": [{"lat": 28.1, "lon": 85.1}, {"lat": 28.1, "lon": 85.11},
                                                                           {"lat": 28.11, "lon": 85.11}, {"lat": 28.1, "lon": 85.1}]},
        {"type": "way", "id": 2, "tags": {"building": "yes"}, "geometry": [{"lat": 28.1, "lon": 85.1}, {"lat": 28.1, "lon": 85.11}]},   # unclosed
        {"type": "way", "id": 3, "tags": {"highway": "track"}, "geometry": [{"lat": 28.05, "lon": 85.05}, {"lat": 28.06, "lon": 85.06}]},
        {"type": "node", "id": 4, "tags": {"place": "village", "name": "X"}, "lat": 28.1, "lon": 85.1},
        {"type": "node", "id": 5, "tags": {"place": "village"}, "lat": 29.5, "lon": 86.5},      # outside AOI
        {"type": "relation", "id": 6, "tags": {"building": "yes"}, "members": []}]}
    b = op.to_geojson(js, "buildings", AOI)
    assert [f["properties"]["@osmId"] for f in b["features"]] == ["way/1"] and b["features"][0]["geometry"]["type"] == "Polygon"
    r = op.to_geojson(js, "roads", AOI)
    assert any(f["geometry"]["type"] == "LineString" and f["properties"]["@osmId"] == "way/3" for f in r["features"])
    s = op.to_geojson(js, "settlements", AOI)
    assert [f["properties"]["@osmId"] for f in s["features"]] == ["node/4"]


def test_ohsome_403_falls_back_to_overpass_and_is_recorded(monkeypatch):
    from app.services import osm_snapshot as snap
    class R:
        status_code = 403; text = "forbidden"
    monkeypatch.setattr(snap.requests, "post", lambda *a, **k: R())
    seen = {}
    def fake_fetch_layer(name, aoi, snapshot, event):
        seen.setdefault("calls", []).append((name, snapshot, event))
        return {"type": "FeatureCollection", "features": []}
    monkeypatch.setattr(op, "fetch_layer", fake_fetch_layer)
    snap._STATE["ohsome_geometry_blocked"], snap._STATE["sources"] = False, {}
    snap._fetch(AOI, SNAP, EV, "building=*", "buildings")
    snap._fetch(AOI, SNAP, EV, "place=village", "villages")      # second layer must skip ohsome entirely
    assert [c[0] for c in seen["calls"]] == ["buildings", "villages"]
    assert snap._STATE["sources"]["villages"].startswith("overpass_attic@2026-07-27")
