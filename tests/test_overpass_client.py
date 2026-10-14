"""Queue-aware, cached, tiled Overpass client (app/services/osm_overpass.py). No network: a fake server is injected."""
import re
from datetime import date

import pytest

from app.services import osm_overpass as op

EV, SNAP = date(2026, 8, 26), date(2026, 8, 25)
AOI2 = {"type": "Polygon", "coordinates": [[[85.05, 28.01], [85.15, 28.01], [85.15, 28.05], [85.05, 28.05], [85.05, 28.01]]]}   # 2 lattice tiles
AOI1 = {"type": "Polygon", "coordinates": [[[85.02, 28.01], [85.08, 28.01], [85.08, 28.04], [85.02, 28.04], [85.02, 28.01]]]}   # 1 lattice tile
DATE_HDR = '[date:"2026-08-25T00:00:00Z"]'
BBOX = re.compile(r"\((\d+\.\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)\)")


class Resp:
    def __init__(self, code=200, js=None, text="", headers=None):
        self.status_code, self._js, self.text, self.headers = code, js, text, headers or {}

    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js


class FakeServer:
    def __init__(self, handler, status_script=None):
        self.handler, self.queries, self.status_script, self.gets = handler, [], list(status_script or []), 0

    def get(self, url, headers=None, timeout=None):
        self.gets += 1
        text = self.status_script.pop(0) if self.status_script else "Rate limit: 2\n2 slots available now.\n"
        return Resp(200, text=text)

    def post(self, url, data, headers=None, timeout=None):
        self.queries.append(data["data"])
        return self.handler(data["data"])


def way(i, lon, lat):
    return {"type": "way", "id": i, "tags": {"highway": "track"}, "geometry": [{"lat": lat, "lon": lon}, {"lat": lat + 0.01, "lon": lon + 0.01}]}


def west_of(q):
    s, w, n, e = (float(x) for x in BBOX.search(q).groups())
    return s, w, n, e


def tile_handler(q):
    s, w, n, e = west_of(q)
    return Resp(200, {"elements": [way(int(w * 100), 85.06 if w < 85.1 else 85.12, 28.02)]})   # always inside AOI2


@pytest.fixture
def env(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(op, "_sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(op, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(op, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(op, "_SESSION", None)
    for d in (op.LAST_ERROR, op.LAST_INFO, op.LAST_WARNINGS):
        d.clear()

    def install(server):
        monkeypatch.setattr(op, "_SESSION", server)
        return server
    return install, sleeps, tmp_path


def test_parse_status_variants():
    assert op.parse_status("Rate limit: 2\n2 slots available now.\n") == (True, 0)
    assert op.parse_status("Rate limit: 2\nSlot available after: x, in 9 seconds.\nSlot available after: y, in 5 seconds.\n") == (False, 5)
    assert op.parse_status("Rate limit: 0\n") == (True, 0)
    assert op.parse_status("something unexpected") == (True, 0)          # unknown format must never block us


def test_wait_for_slot_waits_for_the_announced_time(env):
    install, sleeps, _ = env
    srv = install(FakeServer(tile_handler, ["Rate limit: 2\nSlot available after: t, in 5 seconds.\n", "Rate limit: 2\n2 slots available now.\n"]))
    waited = op.wait_for_slot("https://example.invalid/api/interpreter")
    assert sleeps == [6] and waited == 6 and srv.gets == 2


def test_busy_server_is_retried_with_backoff_then_succeeds(env):
    install, sleeps, _ = env
    n = {"c": 0}

    def handler(q):
        n["c"] += 1
        return Resp(504, text="gateway timeout") if n["c"] <= 2 else tile_handler(q)
    srv = install(FakeServer(handler))
    out = op.fetch_layer("roads", AOI1, SNAP, EV)
    assert out is not None and len(out["features"]) == 1
    assert len(srv.queries) == 3 and len(sleeps) == 2 and all(s > 0 for s in sleeps)
    assert all(DATE_HDR in q for q in srv.queries)


def test_second_run_is_served_from_cache(env):
    install, _, _ = env
    srv = install(FakeServer(tile_handler))
    first = op.fetch_layer("roads", AOI2, SNAP, EV)
    n = len(srv.queries)
    assert n == 2 and len(first["features"]) == 2
    second = op.fetch_layer("roads", AOI2, SNAP, EV)
    assert len(srv.queries) == n and len(second["features"]) == 2
    assert op.LAST_INFO["roads"]["cache_hits"] == 2


def test_partial_failure_returns_none_keeps_good_tiles_and_resumes(env):
    install, _, _ = env
    state = {"fail_west": 85.1}

    def handler(q):
        s, w, n, e = west_of(q)
        return Resp(504, text="busy") if w == state["fail_west"] else tile_handler(q)
    srv = install(FakeServer(handler))
    assert op.fetch_layer("roads", AOI2, SNAP, EV) is None             # an incomplete road network must NOT be returned
    assert "1 of 2 tiles failed" in op.LAST_ERROR["roads"]
    state["fail_west"] = None                                          # server recovers
    before = len(srv.queries)
    out = op.fetch_layer("roads", AOI2, SNAP, EV)
    assert out is not None and len(out["features"]) == 2
    assert len(srv.queries) - before == 1                              # only the missing tile was requested again
    assert "roads" not in op.LAST_ERROR


def test_out_of_memory_tile_is_split_and_duplicates_merged(env):
    install, _, _ = env

    def handler(q):
        s, w, n, e = west_of(q)
        if e - w > 0.06:
            return Resp(200, {"remark": "runtime error: Query run out of memory using about 2048 MB of RAM.", "elements": []})
        return Resp(200, {"elements": [way(7, 85.03, 28.02)]})          # the same way appears in several sub-tiles
    srv = install(FakeServer(handler))
    out = op.fetch_layer("roads", AOI1, SNAP, EV)
    assert len(srv.queries) == 5 and len(out["features"]) == 1


def test_permanent_error_is_not_retried(env):
    install, sleeps, _ = env
    srv = install(FakeServer(lambda q: Resp(400, text="syntax error")))
    assert op.fetch_layer("roads", AOI1, SNAP, EV) is None
    assert len(srv.queries) == 1 and sleeps == []
    assert "HTTP 400" in op.LAST_ERROR["roads"]


def test_post_event_snapshot_is_refused_before_any_network_or_cache_use(env):
    install, _, _ = env
    srv = install(FakeServer(tile_handler))
    for snap in (EV, date(2026, 9, 1)):
        with pytest.raises(ValueError, match="STRICTLY BEFORE"):
            op.fetch_layer("roads", AOI1, snap, EV)
    assert srv.queries == []


def test_every_query_is_dated(env):
    install, _, _ = env
    srv = install(FakeServer(lambda q: Resp(200, {"elements": []})))
    for name in ("buildings", "roads", "bridges", "villages", "towns", "health"):
        op.fetch_layer(name, AOI2, SNAP, EV)
    assert srv.queries and all(q.startswith("[out:json]") and DATE_HDR in q for q in srv.queries)


def test_empty_roads_are_not_cached_but_empty_bridges_are(env):
    install, _, tmp = env
    srv = install(FakeServer(lambda q: Resp(200, {"elements": []})))
    out = op.fetch_layer("roads", AOI1, SNAP, EV)
    assert out["features"] == [] and "roads" in op.LAST_WARNINGS
    n = len(srv.queries)
    op.fetch_layer("roads", AOI1, SNAP, EV)
    assert len(srv.queries) == n + 1                                    # re-asked: an empty road answer is suspicious
    op.fetch_layer("bridges", AOI1, SNAP, EV)
    m = len(srv.queries)
    op.fetch_layer("bridges", AOI1, SNAP, EV)
    assert len(srv.queries) == m                                        # empty bridges are normal and cached


def test_point_layers_use_one_request_even_for_a_big_area(env):
    install, _, _ = env
    big = {"type": "Polygon", "coordinates": [[[84.8, 27.8], [85.4, 27.8], [85.4, 28.3], [84.8, 28.3], [84.8, 27.8]]]}
    srv = install(FakeServer(lambda q: Resp(200, {"elements": []})))
    op.fetch_layer("villages", big, SNAP, EV)
    assert len(srv.queries) == 1


def test_tile_lattice_is_global_and_stable():
    a = op.plan_tiles(AOI2, 0.1)
    shifted = {"type": "Polygon", "coordinates": [[[85.07, 28.02], [85.17, 28.02], [85.17, 28.06], [85.07, 28.06], [85.07, 28.02]]]}
    b = op.plan_tiles(shifted, 0.1)
    assert (85.1, 28.0, 85.2, 28.1) in b and (85.1, 28.0, 85.2, 28.1) in op.plan_tiles(
        {"type": "Polygon", "coordinates": [[[85.12, 28.01], [85.18, 28.01], [85.18, 28.05], [85.12, 28.05], [85.12, 28.01]]]}, 0.1)
    assert a == [(85.0, 28.0, 85.1, 28.1), (85.1, 28.0, 85.2, 28.1)]
