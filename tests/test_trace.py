"""
trace.py — parsing a recorded GPX and map-matching it onto the network.

Map-matching's two jobs, and the two ways it can lie:
  - it must FOLLOW the streets you actually walked (snap + shortest-path join);
  - it must NOT invent streets across a GPS dropout (the connector cap).
Both are pinned below, plus parsing and the named-vs-unnamed coverage split.
"""
import networkx as nx

from amble import trace
from amble import progress as prog
from tests.helpers import coord_grid


GPX_11 = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><name>Sunday Morning Walk</name><trkseg>
    <trkpt lat="37.10" lon="-122.10"></trkpt>
    <trkpt lat="37.20" lon="-122.20"><ele>5.0</ele></trkpt>
    <trkpt lat="37.30" lon="-122.30"></trkpt>
  </trkseg></trk>
</gpx>
"""

# no xmlns at all — a vendor file the matcher must still read
GPX_BARE = """<gpx><trk><trkseg>
  <trkpt lat="1.0" lon="2.0"/><trkpt lat="3.0" lon="4.0"/>
</trkseg></trk></gpx>"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_parse_gpx_reads_points_in_order(tmp_path):
    pts = trace.parse_gpx(_write(tmp_path, "w.gpx", GPX_11))
    assert pts == [(37.10, -122.10), (37.20, -122.20), (37.30, -122.30)]


def test_parse_gpx_is_namespace_agnostic(tmp_path):
    assert trace.parse_gpx(_write(tmp_path, "b.gpx", GPX_BARE)) == [(1.0, 2.0), (3.0, 4.0)]


def test_track_name_from_trk(tmp_path):
    assert trace.track_name(_write(tmp_path, "w.gpx", GPX_11)) == "Sunday Morning Walk"
    assert trace.track_name(_write(tmp_path, "b.gpx", GPX_BARE)) is None


def test_match_follows_the_walked_street():
    # a noisy trace down row0 of the grid must match exactly its three edges
    G = coord_grid(4)
    trk = [(0.0, 0.0002), (0.0, 0.0009), (0.0, 0.0021), (0.0, 0.0030)]  # lat, lon
    m = trace.match_trace(G, trk)
    expected = {prog.edge_id(G, (0, c), (0, c + 1), 0) for c in range(3)}
    assert m["edge_ids"] == expected
    assert m["n_skipped"] == 0
    assert m["named_m"] == 3.0          # three 1 m "row0" segments, all named


def test_match_skips_gps_dropout():
    # A and B are ~11 m apart in space but only joined by a 1200 m detour through
    # C. The matcher must NOT claim that detour — it's a dropout, not a walk.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0001, y=0.0)        # ~11 m east of A
    G.add_node("C", x=0.0, y=0.01)          # far north
    G.add_edge("A", "C", length=600.0, name="Long St")
    G.add_edge("C", "B", length=600.0, name="Long St")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.0001)])
    assert m["edge_ids"] == set()
    assert m["n_skipped"] == 1
    # but a hop whose graph path IS within the cap matches normally
    m2 = trace.match_trace(G, [(0.0, 0.0), (0.01, 0.0)])   # A -> C, 600 m path
    assert m2["edge_ids"] == {prog.edge_id(G, "A", "C", 0)}
    assert m2["n_skipped"] == 0


def test_match_dedupes_repeated_fixes_no_self_loops():
    # many fixes parked on one corner must not manufacture edges
    G = coord_grid(4)
    m = trace.match_trace(G, [(0.0, 0.0)] * 25)
    assert m["edge_ids"] == set()
    assert m["n_snapped"] == 1


def test_match_splits_named_and_unnamed_length():
    # walk a named block then an unnamed connector: only the named one counts
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_node("D", x=0.002, y=0.0)
    G.add_edge("A", "B", length=50.0, name="Main St")
    G.add_edge("B", "D", length=50.0)                       # unnamed connector
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001), (0.0, 0.002)])
    assert m["edge_ids"] == {prog.edge_id(G, "A", "B", 0), prog.edge_id(G, "B", "D", 0)}
    assert m["matched_m"] == 100.0
    assert m["named_m"] == 50.0


def test_match_uses_shortest_parallel_edge_key():
    # a road and a longer parallel stairway between the same corners: the match
    # picks the shortest parallel edge's key, and edge_id reflects it
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=120.0, name="road")         # key 0
    G.add_edge("A", "B", length=300.0, name="stairs")       # key 1
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001)])
    assert m["edge_ids"] == {prog.edge_id(G, "A", "B", 0)}
    assert m["matched_m"] == 120.0


def test_import_into_store_is_idempotent():
    # matching a track and marking it walked updates the store; re-importing the
    # same track adds nothing (edge identity is stable, so no double counting)
    G = coord_grid(4)
    trk = [(0.0, 0.0), (0.0, 0.001), (0.0, 0.002), (0.0, 0.003)]
    store = {"walked": {}}
    ids = trace.match_trace(G, trk)["edge_ids"]
    assert prog.mark_edges_walked(store, ids, when="2026-06-07", note="x") == 3
    assert prog.mark_edges_walked(store, ids, when="2026-06-07", note="x") == 0
    assert prog.stats(G, store)["walked_edges"] == 3


def test_match_empty_trace_is_empty():
    G = coord_grid(3)
    m = trace.match_trace(G, [])
    assert m["edge_ids"] == set() and m["n_points"] == 0 and m["n_snapped"] == 0
