import pytest


@pytest.fixture(autouse=True)
def _reset_osm_state():
    """osm_snapshot remembers 'ohsome geometry blocked' within one run; never let that leak between tests."""
    from app.services import osm_snapshot
    osm_snapshot._STATE["ohsome_geometry_blocked"], osm_snapshot._STATE["sources"] = False, {}
    yield
    osm_snapshot._STATE["ohsome_geometry_blocked"], osm_snapshot._STATE["sources"] = False, {}
