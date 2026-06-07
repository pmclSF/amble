"""
network.py pure functions: nearest_node, the compact chunker, elevation math,
and the divided-way / partitioning helpers.
"""
import networkx as nx

from amble import network as net
from tests.helpers import coord_grid, _canon


def test_nearest_node_returns_closest_and_respects_lat_lon():
    G = nx.MultiGraph()
    G.add_node("near", x=-122.505, y=37.753)
    G.add_node("far", x=-122.480, y=37.760)
    G.add_edge("near", "far", length=1.0)
    # query right next to "near"; if x/y were swapped this would pick "far"
    assert net.nearest_node(G, 37.7531, -122.5049) == "near"


def test_chunk_contains_start_is_connected_and_sized():
    G = coord_grid(6)              # 60 edges, 1 m each
    chunk = net.chunk_from_node(G, (0, 0), target_km=0.01)  # 10 m target
    assert (0, 0) in chunk
    assert nx.is_connected(chunk)
    assert set(chunk.nodes) <= set(G.nodes)
    assert net.total_km(chunk) * 1000.0 >= 10.0 - 1e-9


def test_chunk_bias_keeps_start_present():
    # the start node must always land inside its own chunk (a bias that pushed
    # it out once caused a StopIteration downstream)
    G = coord_grid(6)
    for b in (0.0, 0.5, 1.0):
        assert (0, 0) in net.chunk_from_node(G, (0, 0), target_km=0.02, bias=b)


def test_chunk_larger_than_graph_returns_whole_component():
    G = coord_grid(3)
    chunk = net.chunk_from_node(G, (0, 0), target_km=999)
    assert chunk.number_of_edges() == G.number_of_edges()


def test_compute_effort_formula():
    G = nx.MultiGraph()
    G.add_node(1, x=0.0, y=0.0, elevation=10.0)
    G.add_node(2, x=0.001, y=0.0, elevation=25.0)   # +15 m
    G.add_edge(1, 2, length=100.0)
    net.compute_effort(G, climb_weight=8.0)
    d = G[1][2][0]
    assert d["rise"] == 15.0
    assert d["effort"] == 100.0 + 8.0 * 15.0


def test_route_elevation_stats_directional_and_deadhead_only():
    G = nx.MultiGraph()
    G.add_node(1, x=0.0, y=0.0, elevation=0.0)
    G.add_node(2, x=0.001, y=0.0, elevation=30.0)   # climb 30
    G.add_node(3, x=0.002, y=0.0, elevation=10.0)   # descend 20
    G.add_edge(1, 2, length=100.0)
    G.add_edge(2, 3, length=100.0)
    # walk 1->2 (up 30), 2->3 (down 20), then deadhead 3->2 (up 20)
    route = [(1, 2, 0, False), (2, 3, 0, False), (3, 2, 0, True)]
    st = net.route_elevation_stats(G, route)
    assert st["ascent_m"] == 50.0          # 30 (req) + 20 (deadhead)
    assert st["descent_m"] == 20.0
    assert st["deadhead_ascent_m"] == 20.0  # only the deadhead climb
    assert st["required_m"] == 200.0 and st["deadhead_m"] == 100.0


def test_route_elevation_stats_uses_exact_parallel_lengths():
    # a road (len 2) and a longer parallel stairway (len 5), both walked: the
    # reported required_m must be 7 (2 + 5), not min(2,5)*2 = 4 — using the
    # shortest parallel edge's length for both would undercount.
    G = nx.MultiGraph()
    G.add_node("X", x=0.0, y=0.0, elevation=0.0)
    G.add_node("Y", x=0.001, y=0.0, elevation=0.0)
    G.add_edge("X", "Y", length=2.0, name="road")     # key 0
    G.add_edge("X", "Y", length=5.0, name="stairs")   # key 1
    route = [("X", "Y", 0, False), ("Y", "X", 1, False)]
    st = net.route_elevation_stats(G, route)
    assert st["required_m"] == 7.0


def test_route_elevation_stats_handles_synthetic_key():
    # a duplicated edge carries a key absent from the graph; length must still
    # resolve via the parallel-edge minimum, not raise KeyError.
    G = nx.MultiGraph()
    G.add_node(1, x=0.0, y=0.0, elevation=0.0)
    G.add_node(2, x=0.001, y=0.0, elevation=0.0)
    G.add_edge(1, 2, length=100.0)
    st = net.route_elevation_stats(G, [(1, 2, 999, True)])  # key 999 not in G
    assert st["deadhead_m"] == 100.0


def _meter_node(G, n, xm, ym):
    import math
    G.add_node(n, x=-122.44 + xm / (111320 * math.cos(math.radians(37.77))),
               y=37.77 + ym / 110540)


def _divided(name1=None, name2=None, hw2=None):
    """A 4-cycle: two parallel 100 m rails 15 m apart, joined at both ends."""
    G = nx.MultiGraph()
    for n, (x, y) in {"W0": (0, 0), "W1": (0, 100),
                      "E0": (15, 0), "E1": (15, 100)}.items():
        _meter_node(G, n, x, y)
    a = {"length": 100.0}
    if name1:
        a["name"] = name1
    b = {"length": 100.0}
    if name2:
        b["name"] = name2
    if hw2:
        b["highway"] = hw2
    G.add_edge("W0", "W1", **a)
    G.add_edge("E0", "E1", **b)
    G.add_edge("W0", "E0", length=15.0)
    G.add_edge("W1", "E1", length=15.0)
    return G


def test_collapse_drops_one_unnamed_twin_and_stays_connected():
    G = _divided()                       # both unnamed parallel paths
    H, n = net.collapse_divided_ways(G)
    assert n == 1
    assert nx.is_connected(H)
    assert H.number_of_edges() == G.number_of_edges() - 1


def test_collapse_keeps_a_named_street():
    G = _divided(name1="Beach Path")     # one rail named, one unnamed
    H, n = net.collapse_divided_ways(G)
    assert n == 1
    kept = {d.get("name") for _, _, d in H.edges(data=True)}
    assert "Beach Path" in kept          # the named rail survived


def test_collapse_never_deletes_a_named_way():
    # two named carriageways of a divided road -> keep BOTH (walk both sides).
    # Deleting one would silently remove a must-walk street from the 100% total.
    for n1, n2 in (("Upper Great Hwy", "Upper Great Hwy"), ("A St", "B St")):
        G = _divided(name1=n1, name2=n2)
        _, n = net.collapse_divided_ways(G)
        assert n == 0


def test_collapse_preserves_total_named_length():
    # on a graph of only named ways, collapse must not lose any named distance.
    G = _divided(name1="A St", name2="A St")
    before = sum(d.get("length", 0.0) for _, _, d in G.edges(data=True)
                 if d.get("name"))
    H, _ = net.collapse_divided_ways(G)
    after = sum(d.get("length", 0.0) for _, _, d in H.edges(data=True)
                if d.get("name"))
    assert after == before


def test_collapse_keeps_highway_path_trails():
    # two parallel unnamed footpaths tagged highway=path (park trails) must NOT
    # be deduped — the user wants to walk park trails.
    G = _divided()
    for u, v, k in list(G.edges(keys=True)):
        if G[u][v][k].get("length") == 100.0:
            G[u][v][k]["highway"] = "path"
    _, n = net.collapse_divided_ways(G)
    assert n == 0


def test_collapse_preserves_stairways():
    G = _divided(name1="Hill St", hw2="steps")  # road + parallel stairway
    H, n = net.collapse_divided_ways(G)
    assert n == 0
    assert any(net._is_steps(d) for _, _, d in H.edges(data=True))


def test_collapse_leaves_lone_streets_alone():
    G = coord_grid(4)                    # a grid has no parallel twins
    _, n = net.collapse_divided_ways(G)
    assert n == 0


def test_largest_component_and_total_km():
    G = nx.MultiGraph()
    G.add_edge("a", "b", length=1000.0)              # component 1 (1 edge)
    G.add_edge("x", "y", length=500.0)
    G.add_edge("y", "z", length=500.0)               # component 2 (2 edges)
    big = net.largest_component(G)
    assert set(big.nodes) == {"x", "y", "z"}
    assert net.total_km(G) == 2.0


def test_iter_walk_chunks_partitions_all_edges():
    G = coord_grid(5)
    seen = []
    for _i, chunk in net.iter_walk_chunks(G, target_km=0.005):
        assert nx.is_connected(chunk)
        seen.extend(_canon(u, v, k) for u, v, k in chunk.edges(keys=True))
    # every edge covered exactly once across all chunks (a true partition)
    assert sorted(seen) == sorted(_canon(u, v, k) for u, v, k in G.edges(keys=True))
