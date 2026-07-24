"""
render.build_layers — the pure geometry/colour split behind the map (no plotting,
so matplotlib isn't needed here). The map's job is to group edges by walk, colour
each walk distinctly and STABLY, and frame the walked area.
"""
import networkx as nx

from amble import render
from amble import progress as prog


def _g():
    G = nx.MultiGraph()
    for n, (x, y) in {1: (0.0, 0.0), 2: (0.0, 0.001),
                      3: (0.0, 0.002), 4: (0.001, 0.002)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge(1, 2, length=100.0, name="A St")
    G.add_edge(2, 3, length=100.0, name="B St")
    G.add_edge(3, 4, length=100.0)                       # stays unwalked
    return G


def test_build_layers_groups_by_walk_in_date_order_with_distinct_colours():
    G = _g()
    store = {"walked": {
        prog.edge_id(G, 2, 3, 0): {"date": "2026-06-07", "note": "Tuesday"},
        prog.edge_id(G, 1, 2, 0): {"date": "2026-06-06", "note": "Monday"},
    }}
    base, partial, layers, bbox = render.build_layers(G, store)
    assert len(base) == 1                                # only edge 3-4 unwalked
    assert [l["note"] for l in layers] == ["Monday", "Tuesday"]   # oldest first
    assert layers[0]["color"] != layers[1]["color"]
    assert layers[0]["km"] == 0.1 and layers[1]["km"] == 0.1
    minx, maxx, miny, maxy = bbox
    assert miny == 0.0 and maxy == 0.002                 # frames the walked nodes


def test_build_layers_keeps_an_earlier_walks_colour_when_a_new_one_is_added():
    # adding a later walk must not recolour the earlier ones (stable legend)
    G = _g()
    s1 = {"walked": {prog.edge_id(G, 1, 2, 0): {"date": "2026-06-06", "note": "Mon"}}}
    color_before = render.build_layers(G, s1)[2][0]["color"]
    s2 = {"walked": {
        prog.edge_id(G, 1, 2, 0): {"date": "2026-06-06", "note": "Mon"},
        prog.edge_id(G, 2, 3, 0): {"date": "2026-06-07", "note": "Tue"},
    }}
    layers = render.build_layers(G, s2)[2]
    mon = next(l for l in layers if l["note"] == "Mon")
    assert mon["color"] == color_before


def test_build_layers_separates_partial_blocks_from_done():
    # a block walked only part-way goes to partial_segs (the in-progress colour),
    # not the done layer; a fully-walked block stays in the done layer.
    G = _g()
    store = {"walked": {
        prog.edge_id(G, 1, 2, 0): {"intervals": [[0.0, 1.0]], "date": "x", "note": "done"},
        prog.edge_id(G, 2, 3, 0): {"intervals": [[0.0, 0.4]], "date": "x", "note": "half"},
    }}
    base, partial, layers, bbox = render.build_layers(G, store)
    assert len(partial) == 1                          # edge 2-3 is in progress
    assert sum(len(l["segs"]) for l in layers) == 1   # only edge 1-2 is done
    assert len(base) == 1                             # edge 3-4 untouched


def test_canonical_mode_draws_each_passage_once_keyed_on_coverage_id():
    # In the canonical model the base grid is the INVENTORY: one line per
    # passage. Parallel surfaces (named sidewalk shadow, unnamed sidewalk
    # alias) are never drawn, and walked evidence stored under the canonical
    # coverage_id lights up the representative.
    from amble import network
    G = nx.MultiGraph()
    for n, (x, y) in {"r0": (0.0, 0.0), "r1": (0.001, 0.0),
                      "s0": (0.0, 0.00008), "s1": (0.001, 0.00008),
                      "w0": (0.0, -0.00008), "w1": (0.001, -0.00008),
                      "x0": (0.0, 0.001), "x1": (0.001, 0.001)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge("r0", "r1", length=111, name="Main Street", highway="residential")
    G.add_edge("s0", "s1", length=111, name="Main Street", highway="footway")
    G.add_edge("w0", "w1", length=111, highway="footway", footway="sidewalk")
    G.add_edge("x0", "x1", length=111, name="Other Street", highway="residential")
    H, _audit = network.prepare_graph(G)
    cid = next(d["coverage_id"] for *_e, d in H.edges(data=True)
               if d.get("coverage_required")
               and d.get("coverage_name") == "Main Street")
    store = {"walked": {cid: {"intervals": [[0.0, 1.0]],
                              "date": "2026-01-01", "note": "w"}}}
    base, partial, layers, _bbox = render.build_layers(H, store)
    assert len(base) == 1                             # Other Street only
    assert partial == []
    assert sum(len(l["segs"]) for l in layers) == 1   # Main St rep, exactly once


def test_build_layers_no_walks_has_no_bbox():
    G = _g()
    base, partial, layers, bbox = render.build_layers(G, {"walked": {}})
    assert bbox is None and layers == [] and len(base) == 3


def test_redact_demotes_walked_segments_near_a_zone_to_base():
    # node 1 sits at (lat 0, lon 0). A 50 m zone there hides edge 1-2 (which
    # starts at node 1) but NOT edge 2-3 (nearest end ~110 m away).
    G = _g()
    store = {"walked": {
        prog.edge_id(G, 1, 2, 0): {"date": "2026-06-06", "note": "Mon"},
        prog.edge_id(G, 2, 3, 0): {"date": "2026-06-06", "note": "Mon"},
    }}
    base, partial, layers, bbox = render.build_layers(G, store, redact=[(0.0, 0.0, 50.0)])
    assert len(base) == 2                         # unwalked 3-4 + redacted 1-2
    assert len(layers) == 1 and layers[0]["km"] == 0.1   # only 2-3 remains shown
    assert bbox[2] >= 0.001                       # framed area excludes the home end


def test_redact_drops_a_fully_hidden_walk_entirely():
    G = _g()
    store = {"walked": {prog.edge_id(G, 1, 2, 0): {"date": "x", "note": "Mon"}}}
    base, partial, layers, bbox = render.build_layers(G, store, redact=[(0.0, 0.0, 50.0)])
    assert layers == [] and bbox is None and len(base) == 3   # nothing revealed


def test_redact_catches_an_edge_passing_through_a_zone():
    # endpoints are ~111 m out (outside a 50 m zone) but the edge passes through
    # the home point — exact point-to-segment distance must still hide it, or a
    # block straddling home would leak.
    G = nx.MultiGraph()
    G.add_node("A", x=-0.001, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_edge("A", "B", length=222.0, name="Home Block")
    store = {"walked": {prog.edge_id(G, "A", "B", 0): {"date": "x", "note": "w"}}}
    _, _, layers, bbox = render.build_layers(G, store, redact=[(0.0, 0.0, 50.0)])
    assert layers == [] and bbox is None
