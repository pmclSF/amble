"""
export.py turn-by-turn + GPX. Directions are safety-relevant: a left/right swap
sends the walker the wrong way, so the maneuver classifier and the cue debounce
are checked at their exact boundaries.
"""
import xml.etree.ElementTree as ET

import networkx as nx
import pytest

from amble import export as exp
from amble.export import _bearing_ll, _maneuver, route_to_cues, route_to_gpx


def test_bearing_cardinals():
    # small offsets around (lat0, lon0); _bearing_ll(lon1, lat1, lon2, lat2)
    assert _bearing_ll(0, 0, 0, 0.001) == pytest.approx(0, abs=1)     # north
    assert _bearing_ll(0, 0, 0.001, 0) == pytest.approx(90, abs=1)    # east
    assert _bearing_ll(0, 0, 0, -0.001) == pytest.approx(180, abs=1)  # south
    assert _bearing_ll(0, 0, -0.001, 0) == pytest.approx(270, abs=1)  # west


@pytest.mark.parametrize("delta,phrase,sym", [
    (0, "Continue", "Straight"),
    (20, "Continue", "Straight"),      # boundary: still straight
    (30, "Slight right", "Right"),
    (-30, "Slight left", "Left"),
    (90, "Turn right", "Right"),
    (-90, "Turn left", "Left"),
    (150, "Sharp right", "Right"),
    (-150, "Sharp left", "Left"),
    (179, "U-turn", "Straight"),    # a reversal gets no left/right arrow
    (-179, "U-turn", "Straight"),
])
def test_maneuver_classification(delta, phrase, sym):
    assert _maneuver(delta) == (phrase, sym)


def _L_route():
    # Walk east along row0 then turn (left, north) up col2: A->B->C then C->D.
    G = nx.MultiGraph()
    G.add_node("A", x=0.000, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_node("C", x=0.002, y=0.0)
    G.add_node("D", x=0.002, y=0.001)
    G.add_edge("A", "B", length=1.0, name="Judah St")
    G.add_edge("B", "C", length=1.0, name="Judah St")
    G.add_edge("C", "D", length=1.0, name="46th Ave")
    route = [("A", "B", 0, False), ("B", "C", 0, False), ("C", "D", 0, False)]
    return G, route


def test_cues_emit_start_turn_end_and_skip_straight():
    G, route = _L_route()
    cues = route_to_cues(G, route)
    texts = [c[2] for c in cues]
    syms = [c[4] for c in cues]
    assert texts[0] == "Start on Judah St"
    assert syms[0] == "Start" and syms[-1] == "End"
    # B is a straight-through on the same street -> NO cue there
    assert not any("onto Judah" in t and "Turn" in t for t in texts)
    # the only maneuver is the left turn onto 46th Ave at C (heading 90 -> 0)
    turns = [c for c in cues if c[4] in ("Left", "Right")]
    assert len(turns) == 1
    assert turns[0][4] == "Left"
    assert "46th Ave" in turns[0][2]


def test_cues_guide_through_unnamed_turns_with_names_when_available():
    # Judah St, a turn through an UNNAMED connector, then a turn onto 46th Ave.
    # Both turns are well-separated, so both are kept: the named one carries the
    # name, the unnamed one is a bare "Turn left/right" so you're still guided.
    G = nx.MultiGraph()
    G.add_node("A", x=0.000, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_node("C", x=0.001, y=0.0010)   # ~110 m north: turn onto unnamed connector
    G.add_node("D", x=0.002, y=0.0010)   # turn east onto 46th Ave
    G.add_edge("A", "B", length=1.0, name="Judah St")
    G.add_edge("B", "C", length=1.0)                 # unnamed connector
    G.add_edge("C", "D", length=1.0, name="46th Ave")
    route = [("A", "B", 0, False), ("B", "C", 0, True), ("C", "D", 0, False)]
    cues = route_to_cues(G, route)
    texts = [c[2] for c in cues]
    assert any("46th Ave" in t for t in texts)              # named turn keeps the name
    assert any(c[3] == "" and c[4] in ("Left", "Right") for c in cues)  # unnamed turn still cued


def test_cue_debounce_collapses_slight_wiggle():
    # a near-straight connector with several SLIGHT (<50°) kinks close together
    # collapses — but a genuine turn would not (see next test).
    G = nx.MultiGraph()
    pts = {"A": (0.0000, 0.0), "B": (0.0001, 0.00001), "C": (0.0002, 0.0),
           "D": (0.0003, 0.00001), "E": (0.0010, 0.0)}
    for n, (x, y) in pts.items():
        G.add_node(n, x=x, y=y)
    for a, b in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")):
        G.add_edge(a, b, length=1.0)
    route = [(a, b, 0, False) for a, b in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"))]
    middle = [c for c in route_to_cues(G, route) if c[4] not in ("Start", "End")]
    assert len(middle) <= 1


def test_debounce_never_drops_a_sharp_turn():
    # a genuine 90° turn right after a short block must NOT be debounced away.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0001, y=0.0)        # ~9 m east on Main
    G.add_node("C", x=0.0001, y=0.0001)     # sharp left (north) onto Cross
    G.add_edge("A", "B", length=1.0, name="Main St")
    G.add_edge("B", "C", length=1.0, name="Cross St")
    cues = route_to_cues(G, route_with := [("A", "B", 0, False), ("B", "C", 0, False)])
    assert any("Cross St" in c[2] for c in cues)   # the sharp turn survives


def test_uturn_cue_emitted_for_same_street_dead_end():
    # walk up a named spur and back (dead-end out-and-back) -> "Turn around",
    # NOT swallowed as a same-street straight-through.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0, y=0.0010)
    G.add_edge("A", "B", length=1.0, name="Spur St")
    route = [("A", "B", 0, False), ("B", "A", 0, True)]
    cues = route_to_cues(G, route)
    assert any("Turn around" in c[2] for c in cues)
    assert all(c[4] != "Left" and c[4] != "Right"
               for c in cues if "Turn around" in c[2])   # no l/r arrow on a reversal


def test_edge_coords_oriented_u_to_v():
    G, _ = _L_route()
    coords = exp._edge_coords(G, "A", "B", 0)
    assert coords[0] == (G.nodes["A"]["x"], G.nodes["A"]["y"])
    # reversed direction flips orientation
    rev = exp._edge_coords(G, "B", "A", 0)
    assert rev[0] == (G.nodes["B"]["x"], G.nodes["B"]["y"])


def test_gpx_is_wellformed_and_cue_count_matches(tmp_path):
    G, route = _L_route()
    path = str(tmp_path / "r.gpx")
    _, npts = route_to_gpx(G, route, path, cues=True)
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(path).getroot()          # raises if malformed
    assert len(root.findall(".//g:trkpt", ns)) == npts > 0
    assert len(root.findall("g:wpt", ns)) == len(route_to_cues(G, route))


def test_gpx_cues_false_writes_no_waypoints(tmp_path):
    G, route = _L_route()
    path = str(tmp_path / "nocue.gpx")
    route_to_gpx(G, route, path, cues=False)
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    assert ET.parse(path).getroot().findall("g:wpt", ns) == []


def test_gpx_embeds_elevation_when_present(tmp_path):
    G, route = _L_route()
    for n, z in (("A", 0.0), ("B", 5.0), ("C", 10.0), ("D", 20.0)):
        G.nodes[n]["elevation"] = z
    path = str(tmp_path / "ele.gpx")
    route_to_gpx(G, route, path)
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    eles = [float(e.text) for e in ET.parse(path).getroot().findall(".//g:trkpt/g:ele", ns)]
    # EXACT per-trkpt sequence along A->B->C->D. A range check (min>=0/max<=20)
    # passed even if elevations were constant or attached to the wrong nodes;
    # this catches a mis-mapping because the order and values must both match.
    assert eles == [0.0, 5.0, 10.0, 20.0]


def test_progress_to_geojson_marks_walked(tmp_path):
    import json
    from amble import progress as prog
    G = nx.MultiGraph()
    G.add_node(1, x=0.0, y=0.0); G.add_node(2, x=0.001, y=0.0); G.add_node(3, x=0.002, y=0.0)
    G.add_edge(1, 2, length=100.0, name="A St"); G.add_edge(2, 3, length=100.0, name="B St")
    walked = {prog.edge_id(G, 1, 2, 0)}
    path = str(tmp_path / "m.geojson")
    exp.progress_to_geojson(G, walked, lambda u, v, k: prog.edge_id(G, u, v, k), path)
    feats = json.load(open(path))["features"]
    assert sum(1 for f in feats if f["properties"]["walked"]) == 1
