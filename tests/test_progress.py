"""
progress.py is the multi-year dataset's source of truth. Key behaviours:
geometry-stable edge_id (survives re-fetch + matches the same street across
neighbourhood fetches), order-independence, deadhead skipping, named-only stats.
"""
import networkx as nx

from amble import progress as prog


def _g(edges):
    """Build a small MultiGraph with coords; edges = list of (u,v,attrs)."""
    G = nx.MultiGraph()
    coords = {1: (0.0, 0.0), 2: (0.0, 0.001), 3: (0.0, 0.002), 4: (0.0, 0.003),
              "A": (0.0, 0.0), "B": (0.0, 0.001)}
    for u, v, attrs in edges:
        for n in (u, v):
            x, y = coords[n]
            G.add_node(n, x=x, y=y)
        G.add_edge(u, v, **attrs)
    return G


def test_edge_id_is_order_independent():
    G = _g([(1, 2, {"length": 100.0, "name": "A St"})])
    assert prog.edge_id(G, 1, 2, 0) == prog.edge_id(G, 2, 1, 0)


def test_edge_id_distinguishes_parallel_keys():
    G = _g([(1, 2, {"length": 100.0, "name": "Main St"}),
            (1, 2, {"length": 140.0, "name": "Vulcan Steps"})])
    assert prog.edge_id(G, 1, 2, 0) != prog.edge_id(G, 1, 2, 1)


def test_edge_id_does_not_alias_distinct_streets():
    # two physically distinct same-named blocks <1 m apart must NOT collide,
    # or walking one would falsely mark the other "done".
    G = nx.MultiGraph()
    G.add_node(1, x=-122.4, y=37.770); G.add_node(2, x=-122.4, y=37.771)
    G.add_node(3, x=-122.400003, y=37.770003); G.add_node(4, x=-122.400002, y=37.771001)
    G.add_edge(1, 2, length=111.0, name="Main St")
    G.add_edge(3, 4, length=112.0, name="Main St")
    assert prog.edge_id(G, 1, 2, 0) != prog.edge_id(G, 3, 4, 0)


def test_mark_route_walked_skips_deadhead():
    G = _g([(1, 2, {"length": 100.0, "name": "A St"}),
            (2, 3, {"length": 100.0, "name": "B St"})])
    store = {"walked": {}}
    route = [(1, 2, 0, False), (2, 3, 0, False), (3, 2, 99, True)]  # synthetic deadhead
    n = prog.mark_route_walked(store, G, route)
    assert n == 2, "deadhead must not count as newly covered"
    assert len(store["walked"]) == 2
    assert prog.edge_id(G, 2, 3, 0) in store["walked"]


def test_mark_route_walked_is_idempotent():
    G = _g([(1, 2, {"length": 100.0, "name": "A St"})])
    store = {"walked": {}}
    route = [(1, 2, 0, False)]
    assert prog.mark_route_walked(store, G, route) == 1
    assert prog.mark_route_walked(store, G, route) == 0
    assert len(store["walked"]) == 1


def _named_graph():
    return _g([(1, 2, {"length": 100.0, "name": "A St"}),
               (2, 3, {"length": 300.0, "name": "B St"})])


def test_remaining_subgraph_excludes_walked():
    G = _named_graph()
    store = {"walked": {prog.edge_id(G, 1, 2, 0): {"date": "2026-06-06"}}}
    R = prog.remaining_subgraph(G, store)
    remaining = {prog.edge_id(G, u, v, k) for u, v, k in R.edges(keys=True)}
    assert prog.edge_id(G, 1, 2, 0) not in remaining
    assert prog.edge_id(G, 2, 3, 0) in remaining


def test_stats_exact_and_empty_graph_guard():
    G = _named_graph()
    store = {"walked": {prog.edge_id(G, 1, 2, 0): {"date": "x"}}}   # 100 m walked
    s = prog.stats(G, store)
    assert s["total_km"] == 0.4 and s["walked_km"] == 0.1 and s["pct_done"] == 25.0
    assert s["total_edges"] == 2 and s["walked_edges"] == 1
    empty = prog.stats(nx.MultiGraph(), {"walked": {}})
    assert empty["pct_done"] == 0.0


def test_stats_counts_only_named_ways():
    G = _g([(1, 2, {"length": 100.0, "name": "Real St"}),
            (3, 4, {"length": 900.0})])            # unnamed connector
    s = prog.stats(G, {"walked": {}})
    assert s["total_edges"] == 1 and s["total_km"] == 0.1


def test_is_required_corridor_counts_canonical_only():
    from amble.equivalents import canonical_name
    # the canonical name of a declared corridor is required; its redundant
    # aliases are optional connectors (you walk the strip once, not four times).
    assert prog.is_required({"name": "Great Highway"}) is True
    assert prog.is_required({"name": "Sunset Dunes"}) is False
    assert prog.is_required({"name": "Lower Great Highway Trail"}) is False
    # an ordinary street is still required; unnamed is still not.
    assert prog.is_required({"name": "Noriega Street"}) is True
    assert prog.is_required({}) is False
    assert canonical_name("Sunset Dunes") == "Great Highway"


def test_stats_counts_a_corridor_once():
    # a corridor canonical + a parallel alias + a normal street. Only the
    # canonical line and the normal street count toward 100%; the alias is
    # excluded, so the corridor is one must-walk, not two.
    G = nx.MultiGraph()
    for n, (x, y) in {1: (0, 0), 2: (0, .001), 3: (.0002, 0),
                      4: (.0002, .001), 5: (0, .002)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge(1, 2, length=100.0, name="Great Highway")    # canonical -> required
    G.add_edge(3, 4, length=100.0, name="Sunset Dunes")     # alias    -> NOT required
    G.add_edge(2, 5, length=100.0, name="Noriega Street")   # normal   -> required
    s = prog.stats(G, {"walked": {}})
    assert s["total_edges"] == 2 and s["total_km"] == 0.2


def test_store_roundtrip_and_missing_file(tmp_path):
    path = tmp_path / "p.json"
    assert prog.load_store(str(path)) == {"walked": {}}
    store = {"walked": {"a~b|X|5": {"date": "2026-06-06", "note": "hi"}}}
    prog.save_store(store, str(path))
    assert prog.load_store(str(path)) == store


def test_rekey_store_carries_across_renumbering():
    Gold = _g([(1, 2, {"length": 100.0, "name": "A St"})])
    Gnew = nx.MultiGraph()
    Gnew.add_node("X", x=0.0, y=0.0000003); Gnew.add_node("Y", x=0.0, y=0.0010003)
    Gnew.add_edge("X", "Y", length=102.0, name="A St")     # same street, new ids + drift
    store = {"walked": {prog.edge_id(Gold, 1, 2, 0): {"date": "x"}}}
    new, mig, lost, merged = prog.rekey_store(store, Gold, Gnew)
    assert mig == 1 and lost == 0
    assert prog.edge_id(Gnew, "X", "Y", 0) in new["walked"]


def test_rekey_store_merges_not_overwrites_on_collision():
    # two old parallel edges match (same name + midpoint) one new edge: keep one,
    # report the other as merged — never silently overwrite (which would lose a record).
    Gold = nx.MultiGraph()
    Gold.add_node(1, x=0.0, y=0.0); Gold.add_node(2, x=0.0, y=0.001)
    Gold.add_edge(1, 2, length=100.0, name="A St")          # key 0
    Gold.add_edge(1, 2, length=100.0, name="A St")          # key 1, same geometry
    Gnew = _g([(1, 2, {"length": 100.0, "name": "A St"})])
    store = {"walked": {prog.edge_id(Gold, 1, 2, 0): {"date": "x"},
                        prog.edge_id(Gold, 1, 2, 1): {"date": "y"}}}
    new, mig, lost, merged = prog.rekey_store(store, Gold, Gnew)
    assert mig == 1 and merged == 1 and len(new["walked"]) == 1


def test_rekey_store_tolerant_to_realistic_drift():
    # a re-fetch with ~5 m coord jitter + 10% length change must still re-key.
    Gold = _g([(1, 2, {"length": 100.0, "name": "A St"})])     # node 2 at (x=0, y=0.001)
    Gnew = nx.MultiGraph()
    Gnew.add_node("X", x=0.00004, y=0.00004); Gnew.add_node("Y", x=0.00004, y=0.00104)
    Gnew.add_edge("X", "Y", length=110.0, name="A St")          # ~5 m off, +10% length
    store = {"walked": {prog.edge_id(Gold, 1, 2, 0): {"date": "x"}}}
    new, mig, lost, merged = prog.rekey_store(store, Gold, Gnew)
    assert mig == 1 and lost == 0


def test_rekey_store_does_not_match_a_different_street():
    # a far / differently-named street must NOT be matched (no false re-key).
    Gold = _g([(1, 2, {"length": 100.0, "name": "A St"})])
    Gnew = nx.MultiGraph()
    Gnew.add_node("X", x=0.01, y=0.01); Gnew.add_node("Y", x=0.01, y=0.011)
    Gnew.add_edge("X", "Y", length=100.0, name="B St")         # different name + far
    store = {"walked": {prog.edge_id(Gold, 1, 2, 0): {"date": "x"}}}
    _, mig, lost, merged = prog.rekey_store(store, Gold, Gnew)
    assert mig == 0 and lost == 1
