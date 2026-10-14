"""Pure graph logic of the cut-off analysis (no geometry)."""
import networkx as nx

from app.api.schemas import AccessStatus
from app.services.accessibility import classify_settlements, distance_tree, walk_route


def _graph():
    G = nx.Graph()
    for i, (u, v) in enumerate([(0, 1), (1, 2), (2, 3), (2, 4), (8, 9)]):
        G.add_edge(u, v, weight=100.0, edge_id=i, way=f"way/{i}")
    return G


def test_distance_tree_and_route():
    d, p = distance_tree(_graph(), {0})
    assert d[3] == 300.0 and walk_route(p, 3) == [3, 2, 1, 0]


def test_cut_off_connected_uncertain_classes():
    snapped = {"A": (3, 10.0), "C": (9, 10.0), "D": (None, float("inf")), "E": (1, 50.0)}
    r = classify_settlements(_graph(), {0.0: {1}}, snapped, {0}, [0.0])
    assert r["A"]["status"] == AccessStatus.POTENTIALLY_CUT_OFF and r["A"]["d1_km"] is None
    assert r["E"]["status"] == AccessStatus.CONNECTED
    assert r["C"]["status"] == AccessStatus.UNCERTAIN and r["C"]["reason"] == "baseline_unreachable"
    assert r["D"]["reason"] == "snapping_failed"


def test_status_flip_across_buffers_is_uncertain():
    # edge 3 = (2,4) only blocked at the 60 m buffer
    r = classify_settlements(_graph(), {0.0: set(), 30.0: set(), 60.0: {3}}, {"B": (4, 5.0)}, {0}, [0.0, 30.0, 60.0])
    assert r["B"]["status"] == AccessStatus.UNCERTAIN and r["B"]["reason"] == "unstable_across_buffers"


def test_no_targets_never_cut_off():
    r = classify_settlements(_graph(), {0.0: {1}}, {"A": (3, 5.0)}, set(), [0.0])
    assert r["A"]["status"] == AccessStatus.UNCERTAIN and r["A"]["reason"] == "no_targets"
