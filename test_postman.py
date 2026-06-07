"""Verify the Chinese Postman solver on graphs with hand-computable answers."""
import math
import networkx as nx
from amble.postman import solve_route


def _covers_all(G, route):
    """Every original edge appears at least once in the route."""
    walked = set()
    for u, v, key, _dead in route:
        walked.add(frozenset((u, v)) if u != v else (u, v))
    for u, v in G.edges():
        if frozenset((u, v)) not in walked:
            return False
    return True


def test_even_cycle_no_repeats():
    # 4-node square, every node degree 2 -> already Eulerian, zero deadhead.
    G = nx.MultiGraph()
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        G.add_edge(a, b, length=1.0)
    r = solve_route(G, open_route=False)
    assert r["n_odd_nodes"] == 0
    assert math.isclose(r["deadhead_m"], 0.0)
    assert math.isclose(r["required_m"], 4.0)
    assert math.isclose(r["efficiency"], 1.0)
    assert _covers_all(G, r["route"])
    print("PASS even cycle: deadhead=0, total=4.0, efficiency=1.0")


def test_square_with_diagonal_closed():
    # square (4 unit edges) + diagonal A-C of length sqrt(2).
    # odd nodes: A(1) and C(3). cheapest pairing = the diagonal itself.
    G = nx.MultiGraph()
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        G.add_edge(a, b, length=1.0)
    G.add_edge(1, 3, length=math.sqrt(2))
    r = solve_route(G, open_route=False)
    assert r["n_odd_nodes"] == 2
    assert math.isclose(r["deadhead_m"], math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(r["required_m"], 4 + math.sqrt(2), rel_tol=1e-9)
    assert _covers_all(G, r["route"])
    print(f"PASS square+diagonal (closed): deadhead={r['deadhead_m']:.4f} "
          f"(expected {math.sqrt(2):.4f})")


def test_dead_end_must_backtrack():
    # path A-B-C: both ends are dead ends (degree 1). A closed walk must
    # traverse each edge twice (out and back). total = 4, deadhead = 2.
    G = nx.MultiGraph()
    G.add_edge("A", "B", length=1.0)
    G.add_edge("B", "C", length=1.0)
    r = solve_route(G, open_route=False)
    assert math.isclose(r["deadhead_m"], 2.0)
    assert math.isclose(r["total_m"], 4.0)
    assert _covers_all(G, r["route"])
    print("PASS dead-end (closed): every edge walked twice, total=4.0")


def test_open_route_beats_closed_on_path():
    # Same path A-B-C, but OPEN: start at A, end at C, walk each edge once.
    # deadhead should drop to 0, total = 2.
    G = nx.MultiGraph()
    G.add_edge("A", "B", length=1.0)
    G.add_edge("B", "C", length=1.0)
    r = solve_route(G, open_route=True)
    assert math.isclose(r["deadhead_m"], 0.0), r["deadhead_m"]
    assert math.isclose(r["total_m"], 2.0)
    assert r["endpoints"][0] != r["endpoints"][1]
    assert _covers_all(G, r["route"])
    print(f"PASS open route on path: deadhead=0, endpoints={r['endpoints']}")


def test_open_saves_on_grid():
    # 3x3 grid graph: open route should be <= closed route in total distance.
    G0 = nx.grid_2d_graph(3, 3)
    G = nx.MultiGraph()
    for u, v in G0.edges():
        G.add_edge(u, v, length=1.0)
    closed = solve_route(G, open_route=False)
    opened = solve_route(G, open_route=True)
    assert _covers_all(G, closed["route"])
    assert _covers_all(G, opened["route"])
    # exact hand-computed optima (not just open <= closed): a suboptimal open
    # route of 15 would slip past the inequality but fails these.
    assert math.isclose(closed["total_m"], 16.0), closed["total_m"]
    assert math.isclose(opened["total_m"], 14.0), opened["total_m"]
    print(f"PASS 3x3 grid: closed total={closed['total_m']:.1f}, "
          f"open total={opened['total_m']:.1f} "
          f"(open efficiency {opened['efficiency']:.2%})")


def test_parallel_streets_both_walked():
    # two distinct streets between same intersections (e.g. a road and a
    # parallel stairway). Both must be covered.
    G = nx.MultiGraph()
    G.add_edge("X", "Y", length=2.0, name="road")
    G.add_edge("X", "Y", length=3.0, name="stairs")
    r = solve_route(G, open_route=True)
    # both DISTINCT parallel streets (keys 0 and 1) must be covered — counting
    # X-Y traversals isn't enough, since walking key 0 twice would also reach 2.
    keys = {k for u, v, k, d in r["route"]
            if frozenset((u, v)) == frozenset(("X", "Y"))}
    assert keys == {0, 1}, keys
    print("PASS parallel streets: both the road and the stairway are covered")


if __name__ == "__main__":
    test_even_cycle_no_repeats()
    test_square_with_diagonal_closed()
    test_dead_end_must_backtrack()
    test_open_route_beats_closed_on_path()
    test_open_saves_on_grid()
    test_parallel_streets_both_walked()
    print("\nAll solver tests passed.")
