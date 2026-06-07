"""Shared test builders and a strong Eulerian-trail validity check."""
from collections import Counter
import networkx as nx


def coord_grid(n, length=1.0):
    """
    n x n grid MultiGraph with real-ish coordinates so bearing/elevation code
    works. Node (r, c) sits at lon=c*0.001, lat=r*0.001. Avenues (vertical) are
    named "colC"; streets (horizontal) "rowR".
    """
    G = nx.MultiGraph()
    for r in range(n):
        for c in range(n):
            G.add_node((r, c), x=c * 0.001, y=r * 0.001)
    for r in range(n):
        for c in range(n):
            if c < n - 1:
                G.add_edge((r, c), (r, c + 1), length=length, name=f"row{r}")
            if r < n - 1:
                G.add_edge((r, c), (r + 1, c), length=length, name=f"col{c}")
    return G


def _canon(u, v, k):
    a, b = sorted((u, v), key=str)
    return (a, b, k)


def assert_valid_trail(G, result):
    """
    Assert the solved route is a genuine Eulerian trail of G:

    1. CONTIGUOUS — each edge starts where the previous one ended.
    2. EXACTLY-ONCE — every original edge (by parallel key) appears exactly once
       among the non-deadhead entries (deadheads are the duplicates).

    This is strictly stronger than the original _covers_all (at-least-once by
    frozenset), which could not see a teleporting or edge-doubling trail.
    """
    route = result["route"]
    assert route, "empty route"
    for (u, v, k, _d), (u2, v2, k2, _d2) in zip(route, route[1:]):
        assert v == u2, f"non-contiguous trail: edge ends at {v} but next starts at {u2}"
    walked_once = Counter(_canon(u, v, k) for (u, v, k, d) in route if not d)
    original = Counter(_canon(u, v, k) for u, v, k in G.edges(keys=True))
    assert walked_once == original, (
        "non-deadhead traversals must equal the original edges exactly; "
        f"missing={original - walked_once}, extra={walked_once - original}"
    )
