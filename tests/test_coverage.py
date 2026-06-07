"""
coverage.plan_coverage — Rural-Postman coverage of named ways over the full
graph. A named-only subgraph fractures into disconnected islands (named ways
connect *through* unnamed pavement), so plan_coverage must stitch those islands
with connectors into ONE contiguous walk that converges to 100% of named ways.
"""
import networkx as nx

from amble import coverage, progress as prog
from amble.progress import edge_id


def _two_islands():
    # named edge A-B and named edge C-D are separate "islands"; an UNNAMED edge
    # B-C is the only thing joining them.
    G = nx.MultiGraph()
    for n, (x, y) in {"A": (0, 0), "B": (0.001, 0), "C": (0.002, 0),
                      "D": (0.003, 0)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge("A", "B", length=100.0, name="X St")
    G.add_edge("C", "D", length=100.0, name="Y St")
    G.add_edge("B", "C", length=50.0)            # unnamed connector
    return G


def test_coverage_stitches_islands_into_one_contiguous_walk():
    G = _two_islands()
    sol = coverage.plan_coverage(G, set(), start="A", target_km=999)
    # both named islands covered
    covered = {tuple(sorted((u, v), key=str)) for (u, v, k, d) in sol["route"] if not d}
    assert ("A", "B") in covered and ("C", "D") in covered
    # ONE contiguous route (no teleporting between islands)
    assert all(v == u2 for (u, v, k, d), (u2, v2, k2, d2)
               in zip(sol["route"], sol["route"][1:]))
    # the unnamed connector is deadhead, not coverage
    assert sol["required_m"] == 200.0


def test_coverage_converges_to_100_percent():
    G = _two_islands()
    store = {"walked": {}}
    for _ in range(10):
        sol = coverage.plan_coverage(G, prog.walked_id_set(store), start="A", target_km=999)
        if sol["required_m"] < 1:
            break
        prog.mark_route_walked(store, G, sol["route"])
    named = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("name")]
    walked = prog.walked_id_set(store)
    assert all(edge_id(G, u, v, k) in walked for (u, v, k) in named)   # every named way done


def test_coverage_empty_when_all_named_walked():
    G = _two_islands()
    done = {edge_id(G, u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("name")}
    sol = coverage.plan_coverage(G, done, start="A")
    assert sol["required_m"] == 0.0 and sol["route"] == []


def test_coverage_does_not_require_corridor_aliases():
    # The planner must target only the canonical corridor line (Great Highway),
    # not its redundant alias (Sunset Dunes), so you walk the strip once.
    G = nx.MultiGraph()
    for n, (x, y) in {"A": (0, 0), "B": (0.001, 0),
                      "C": (0.0, 0.0002), "D": (0.001, 0.0002)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge("A", "B", length=100.0, name="Great Highway")   # canonical
    G.add_edge("C", "D", length=100.0, name="Sunset Dunes")    # alias (parallel)
    G.add_edge("A", "C", length=20.0)                          # connector
    sol = coverage.plan_coverage(G, set(), start="A", target_km=999)
    names = {(G[u][v][k].get("name") if (v in G[u] and k in G[u][v]) else None)
             for (u, v, k, d) in sol["route"] if not d}
    assert "Great Highway" in names          # canonical corridor is covered
    assert "Sunset Dunes" not in names        # alias is not a separate must-walk
    assert sol["required_m"] == 100.0         # the corridor counts once


def test_coverage_reaches_island_beyond_initial_cutoff():
    # The adaptive Dijkstra cutoff seeds at 300 m and escalates (x6, then no
    # cap). With every island within the seed the escalation branch never runs;
    # here Y St sits behind a 500 m unnamed connector — only reachable once the
    # cutoff escalates past 300 m. If that fallback regressed, the far island
    # would be stranded and required_m would drop to 100.
    G = nx.MultiGraph()
    for n, (x, y) in {"A": (0, 0), "B": (0.001, 0),
                      "C": (0.006, 0), "D": (0.007, 0)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge("A", "B", length=100.0, name="X St")
    G.add_edge("B", "C", length=500.0)            # long connector, beyond 300 m seed
    G.add_edge("C", "D", length=100.0, name="Y St")
    sol = coverage.plan_coverage(G, set(), start="A", target_km=999)
    covered = {tuple(sorted((u, v), key=str)) for (u, v, k, d) in sol["route"] if not d}
    assert ("A", "B") in covered and ("C", "D") in covered  # far island reached
    assert sol["required_m"] == 200.0
