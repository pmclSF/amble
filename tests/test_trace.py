"""
trace.py — parsing a recorded GPX and map-matching it onto the network.

Map-matching's two jobs, and the two ways it can lie:
  - it must FOLLOW the streets you actually walked (snap + shortest-path join);
  - it must NOT invent streets across a GPS dropout (the connector cap).
Both are pinned below, plus parsing and the named-vs-unnamed coverage split.
"""
import random

import networkx as nx

from amble import trace
from amble import progress as prog
from tests.helpers import coord_grid


def _sample_walk(G, node_seq, step_m=5.0, noise_m=3.0, seed=0):
    """A dense, slightly noisy GPS track walking a node sequence (lat, lon).
    Synthetic ground truth: we know exactly which edges were walked."""
    rnd = random.Random(seed)
    pts = []
    la2 = lo2 = 0.0
    for u, v in zip(node_seq, node_seq[1:]):
        la1, lo1 = G.nodes[u]["y"], G.nodes[u]["x"]
        la2, lo2 = G.nodes[v]["y"], G.nodes[v]["x"]
        seg = trace._geo_m(la1, lo1, la2, lo2)
        n = max(1, int(seg / step_m))
        for s in range(n):
            t = s / n
            pts.append((la1 + (la2 - la1) * t + rnd.gauss(0, noise_m) / 110540.0,
                        lo1 + (lo2 - lo1) * t + rnd.gauss(0, noise_m) / 111320.0))
    pts.append((la2, lo2))
    return pts


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
    # Interval distance reflects the actually observed portions of the two end
    # blocks plus the proven connector block AND the plausibility-gated
    # entry/exit ranges of the crossing transition (the walker demonstrably
    # traversed block 1's last 10% and block 3's first 10% to reach block 2).
    assert abs(m["named_m"] - 2.8) < 1e-9


def test_match_skips_gps_dropout():
    # A and B are ~100 m apart in space but only joined by a 1200 m detour through
    # C. The matcher must NOT claim that detour — it's a dropout, not a walk. (The
    # gap is >GPS-noise and >cap, so there's no honest stationary reading either:
    # the HMM has to break the run and report a skip, never fabricate the detour.)
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0009, y=0.0)        # ~100 m east of A
    G.add_node("C", x=0.0, y=0.01)          # far north
    G.add_edge("A", "C", length=600.0, name="Long St")
    G.add_edge("C", "B", length=600.0, name="Long St")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.0009)])
    assert m["edge_ids"] == set()
    assert m["n_skipped"] == 1
    # but a hop OBSERVED along the way (fixes every ~130 m, all on the edge)
    # matches normally. Two lone fixes 1.1 km apart would NOT: an in-block
    # jump beyond gap_base_m is missing observation, not walked ground.
    walk_up = [(la * 0.001, 0.0) for la in range(0, 10)] + [(0.01, 0.0)]
    m2 = trace.match_trace(G, walk_up)
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
    # Coverage evidence contains only goal passages; physical GPX distance is a
    # separate metric and includes the unnamed connector.
    assert m["edge_ids"] == {prog.edge_id(G, "A", "B", 0)}
    assert m["raw_m"] > 200.0
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


def test_match_prefers_named_street_over_parallel_unnamed_footway():
    # a GPS walk on a sidewalk next to a street should count as the street, not
    # an unnamed footway edge that merely happens to be shorter.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=100.0, name="Main St")
    G.add_edge("A", "B", length=20.0, highway="footway")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001)])
    assert m["edge_ids"] == {prog.edge_id(G, "A", "B", 0)}
    assert m["matched_m"] == 100.0
    assert m["named_m"] == 100.0


def test_match_handles_string_lengths_from_graphml():
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length="100.0", name="Main St")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001)])
    assert m["edge_ids"] == {prog.edge_id(G, "A", "B", 0)}
    assert m["matched_m"] == 100.0
    assert m["named_m"] == 100.0


def test_match_requires_full_edge_traversal_to_count():
    # walking only part of a block should not count that block as fully walked.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=100.0, name="Main St")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.0005)])
    assert m["edge_ids"] == set()
    assert m["matched_m"] == 50.0       # partial evidence is reported, not completed


def test_disjoint_gpx_segments_do_not_fill_unobserved_middle():
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=100.0, name="Main St")
    m = trace.match_trace_segments(G, [
        [(0.0, 0.0001), (0.0, 0.0002)],
        [(0.0, 0.0008), (0.0, 0.0009)],
    ])
    eid = prog.edge_id(G, "A", "B", 0)
    assert len(m["edge_spans"][eid]) == 2
    assert abs(m["matched_m"] - 20.0) < 1e-6
    assert eid not in m["edge_ids"]


def test_match_does_not_credit_parallel_unwalked_street():
    # Two parallel NAMED streets ~18 m apart (e.g. a second carriageway or a
    # neighbouring street). Walking ONE must never credit the other — the
    # all-candidates crediting only counts candidates near-tied with the nearest.
    G = nx.MultiGraph()
    G.add_node("A1", x=0.0, y=0.0)
    G.add_node("A2", x=0.001, y=0.0)
    G.add_node("B1", x=0.0, y=0.00016)        # ~18 m north of street A
    G.add_node("B2", x=0.001, y=0.00016)
    G.add_edge("A1", "A2", length=100.0, name="A Street")
    G.add_edge("B1", "B2", length=100.0, name="B Street")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001)])   # walk along A only
    assert m["edge_ids"] == {prog.edge_id(G, "A1", "A2", 0)}
    assert m["named_m"] == 100.0


def test_match_recovers_named_stairway_and_park_path():
    # Synthetic ground-truth walk: down a street, up a NAMED stairway, along a
    # NAMED park path, onto a second street. All four are named => required, so a
    # noisy GPS track must credit EVERY one — and must not be fooled by the
    # parallel unnamed sidewalk on the first block. This pins the coverage
    # guarantee for the hard feature types (stairways, park paths) corpus-wide.
    G = nx.MultiGraph()
    for i, x in enumerate([0.0, 0.0010, 0.0020, 0.0030, 0.0040]):
        G.add_node(i, x=x, y=0.0)
    G.add_edge(0, 1, length=111.3, name="Main Street", highway="residential")
    G.add_edge(0, 1, length=111.3, highway="footway")           # unnamed sidewalk decoy
    G.add_edge(1, 2, length=111.3, name="Tiled Steps", highway="steps")
    G.add_edge(2, 3, length=111.3, name="Canyon Path", highway="path")
    G.add_edge(3, 4, length=111.3, name="Second Street", highway="residential")

    m = trace.match_trace(G, _sample_walk(G, [0, 1, 2, 3, 4], seed=1))

    want = {
        prog.edge_id(G, 0, 1, trace._min_walkable_key(G, 0, 1)),  # Main St, not the footway
        prog.edge_id(G, 1, 2, 0),                                 # named stairway
        prog.edge_id(G, 2, 3, 0),                                 # named park path
        prog.edge_id(G, 3, 4, 0),                                 # Second St
    }
    assert want <= m["edge_ids"], f"missing: {want - m['edge_ids']}"
    assert m["named_m"] >= 111.3 * 4 * 0.90                       # observed intervals on all four


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


def _timed_gpx(tmp_path, legs):
    """Build a GPX from (lat, lon, epoch) triples in one trkseg."""
    pts = "".join(
        f'<trkpt lat="{la}" lon="{lo}"><time>'
        f'2026-07-21T{int(t)//3600:02d}:{int(t)//60%60:02d}:{int(t)%60:02d}Z'
        f"</time></trkpt>" for la, lo, t in legs)
    p = tmp_path / "ride.gpx"
    p.write_text('<gpx><trk><name>Mixed</name><trkseg>' + pts +
                 "</trkseg></trk></gpx>")
    return str(p)


def test_subway_station_fixes_do_not_credit_the_street_above():
    # A subway ride under a straight street yields fixes only near stations
    # (ON the street, ~700 m apart) and nothing between. The route between
    # station fixes hugs the street by construction (route ~= gap), but a
    # fix-free jump beyond gap_base_m is missing observation, never a
    # walked-through connector.
    G = nx.MultiGraph()
    for i in range(8):                        # 7 blocks of "Mission Street"
        G.add_node(i, x=i * 0.001, y=0.0)
        if i:
            G.add_edge(i - 1, i, length=111.32, name="Mission Street")
    stations = [(0.0, 0.0), (0.0, 0.0035), (0.0, 0.007)]   # ~390/390 m jumps
    fixes = []
    for lat, lon in stations:                 # 3 fixes idling at each station
        fixes += [(lat, lon), (lat, lon + 1e-5), (lat, lon + 2e-5)]
    m = trace.match_trace(G, fixes)
    covered = sum((hi - lo) * 111.32
                  for spans in m["edge_spans"].values()
                  for lo, hi in spans)
    assert covered <= 25.0, m["edge_spans"]   # station-vicinity slivers only
    assert m["raw_gap_m"] > 700.0             # the jumps are accounted as gaps


def test_vehicle_speed_run_is_stripped_and_segment_split(tmp_path):
    # Walk ~1.3 m/s, then a 20 m/s train ride, then walk again. The ride's
    # fixes must be removed and the track SPLIT there, so the matcher can
    # never credit street blocks from inside a train.
    legs = []
    t = 3600.0
    lon = 0.0
    for _ in range(20):                    # walking: 6 m steps every 5 s
        legs.append((0.0, lon, t)); lon += 6 / 111320.0; t += 5
    for _ in range(15):                    # riding: 100 m steps every 5 s
        legs.append((0.0, lon, t)); lon += 100 / 111320.0; t += 5
    for _ in range(20):                    # walking again
        legs.append((0.0, lon, t)); lon += 6 / 111320.0; t += 5
    path = _timed_gpx(tmp_path, legs)
    segments, stats = trace.strip_vehicle_runs(path)
    assert len(segments) == 2                       # split at the ride
    assert stats["vehicle_m"] > 1200.0              # ~1.4 km ride removed
    kept = sum(trace._geo_m(*a, *b) for s in segments
               for a, b in zip(s, s[1:]))
    assert kept < 300.0                             # only the walking remains


def test_untimed_gpx_passes_through_vehicle_filter(tmp_path):
    p = tmp_path / "plain.gpx"
    p.write_text(GPX_11)
    segments, stats = trace.strip_vehicle_runs(str(p))
    assert stats["vehicle_m"] == 0.0
    assert segments == trace.parse_gpx_segments(str(p))


def test_physical_distance_does_not_count_gpx_teleport():
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_node("C", x=0.050, y=0.0)
    G.add_edge("A", "B", length=111.0, name="Main Street")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.001), (0.0, 0.050)])
    assert 100.0 < m["raw_m"] < 120.0
    assert m["raw_gap_m"] > 5000.0
    assert m["n_raw_gaps"] == 1


def test_far_candidate_fixes_do_not_infer_connector_coverage():
    G = nx.MultiGraph()
    for i in range(4):
        G.add_node(i, x=i * 0.001, y=0.0)
    for i in range(3):
        G.add_edge(i, i + 1, length=111.0, name=f"Block {i}")
    # Both fixes are candidates (about 40 m from the street) but are not close
    # enough to prove that the intervening block was walked.
    m = trace.match_trace(G, [(0.00036, 0.0005), (0.00036, 0.0025)])
    middle = prog.edge_id(G, 1, 2, 0)
    assert middle not in m["edge_ids"]
    assert middle not in m["edge_spans"]


class _Geom:
    """Minimal stand-in for a shapely geometry: only .coords is read."""
    def __init__(self, coords):
        self.coords = coords


def test_hairpin_mouth_crossing_does_not_credit_the_whole_loop():
    # A 300 m U-court whose two ends sit 20 m apart: walking PAST its mouth
    # projects consecutive fixes onto frac~0 and frac~1 of the same edge. The
    # continuity gate must refuse to fill the 300 m between them.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.00018, y=0.0)
    G.add_edge("A", "B", length=300.0, name="Copper Court",
               geometry=_Geom([(0.0, 0.0), (0.0, 0.00126),
                               (0.00018, 0.00126), (0.00018, 0.0)]))
    fixes = [(-0.00005, -0.0002 + i * 0.000045) for i in range(14)]
    m = trace.match_trace(G, fixes)
    eid = prog.edge_id(G, "A", "B", 0)
    frac = sum(hi - lo for lo, hi in m["edge_spans"].get(eid, []))
    assert frac <= 0.2, frac
    assert m["matched_m"] <= 60.0, m["matched_m"]


def test_in_block_gps_dropout_leaves_the_middle_unobserved():
    # 270 m dropout INSIDE one 300 m block: only the two ends were observed.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0027, y=0.0)
    G.add_edge("A", "B", length=300.0, name="Longmarch Street")
    fixes = [(0.0, x * 1e-6) for x in (0, 45, 90, 135)] + \
            [(0.0, 0.0027 - x * 1e-6) for x in (135, 90, 45, 0)]
    m = trace.match_trace(G, fixes)
    eid = prog.edge_id(G, "A", "B", 0)
    frac = sum(hi - lo for lo, hi in m["edge_spans"].get(eid, []))
    assert frac <= 0.2, frac
    assert not prog.is_complete({"intervals": m["edge_spans"].get(eid, [])},
                                300.0)


def test_in_and_out_visit_records_only_the_traversed_range():
    # Poke 93 m into a 150 m dead-end block and come back out the same way:
    # the far 57 m were never observed and must stay uncovered.
    G = nx.MultiGraph()
    G.add_node("CW", x=-0.00072, y=0.0)
    G.add_node("O", x=0.0, y=0.0)
    G.add_node("CE", x=0.00072, y=0.0)
    G.add_node("N", x=0.0, y=0.001357)
    G.add_edge("CW", "O", length=80.0, name="Crossway")
    G.add_edge("O", "CE", length=80.0, name="Crossway")
    G.add_edge("O", "N", length=150.0, name="Stubb Street")
    up = [(y * 1e-6, 0.0) for y in range(0, 841, 60)]
    fixes = [(0.0, -0.00072 + i * 0.00009) for i in range(8)] + \
        up + list(reversed(up)) + \
        [(0.0, i * 0.00009) for i in range(8)]
    m = trace.match_trace(G, fixes)
    eid = prog.edge_id(G, "O", "N", 0)
    spans = m["edge_spans"].get(eid, [])
    frac = sum(hi - lo for lo, hi in spans)
    assert 0.45 <= frac <= 0.72, spans          # ~62% walked, never complete
    assert not prog.is_complete({"intervals": spans}, 150.0)


def test_implausible_detour_does_not_credit_its_endpoint_edges():
    # Two fixes 78 m apart whose only graph route is a 210 m U-detour: the
    # HMM may traverse it to stay alive, but NONE of the detour's edges —
    # including the entry/exit endpoint edges — are walked ground. They must
    # not appear as coverage, nor as full [[0,1]] spans in the store.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0007, y=0.0)
    G.add_node("C", x=0.0, y=-0.0006)
    G.add_node("D", x=0.0007, y=-0.0006)
    G.add_edge("A", "C", length=66.0, name="West Leg")
    G.add_edge("C", "D", length=78.0, name="South Leg")
    G.add_edge("D", "B", length=66.0, name="East Leg")
    m = trace.match_trace(G, [(0.0, 0.0), (0.0, 0.0007)])
    for u, v in (("A", "C"), ("C", "D"), ("D", "B")):
        eid = prog.edge_id(G, u, v, 0)
        assert eid not in m["edge_ids"], (u, v)
        assert eid not in m["edge_spans"], (u, v)


def test_far_parallel_trace_is_candidate_but_not_coverage_evidence():
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=111.0, name="Main Street")
    m = trace.match_trace(G, [(0.00036, 0.0), (0.00036, 0.001)])
    assert m["n_snapped"] == 2       # retained for HMM topology
    assert m["edge_ids"] == set()    # but >30 m away is not proof of walking it
    assert m["edge_spans"] == {}
