"""
solve_route(force_start=...) is new and central to the --start feature: the
open-route solver otherwise ignores the start node entirely, so force_start is
the only thing making the walk begin at the bakery.
"""
import math
import networkx as nx

from amble.postman import solve_route
from tests.helpers import coord_grid, assert_valid_trail


def test_forced_start_is_first_node():
    G = coord_grid(4)
    res = solve_route(G, open_route=True, force_start=(2, 1))
    assert res["route"][0][0] == (2, 1)
    assert res["endpoints"][0] == (2, 1)
    assert_valid_trail(G, res)


def test_forced_corner_costs_nothing_extra():
    # A corner of the free open route is already an endpoint -> no added deadhead.
    G = coord_grid(3)
    free = solve_route(G, open_route=True)
    corner = free["endpoints"][0]
    forced = solve_route(G, open_route=True, force_start=corner)
    assert forced["route"][0][0] == corner
    assert math.isclose(forced["total_m"], free["total_m"], rel_tol=1e-9)


def test_forced_interior_adds_minimal_connector():
    # Forcing an even-degree interior start: the exact minimal total is 15.0 on
    # the 3x3 grid (free optimum 14.0 + a single 1 m connector to the sweep). A
    # weak ">= free" check passed even for the pre-fix suboptimal route, so pin
    # the exact value — it regresses if the even-force_start handling breaks.
    G = coord_grid(3)
    free = solve_route(G, open_route=True)
    forced = solve_route(G, open_route=True, force_start=(1, 1))
    assert math.isclose(free["total_m"], 14.0, rel_tol=1e-9)
    assert math.isclose(forced["total_m"], 15.0, rel_tol=1e-9)
    assert math.isclose(forced["required_m"], free["required_m"], rel_tol=1e-9)
    assert_valid_trail(G, forced)


def test_even_force_start_picks_cheaper_closed_route():
    # An EVEN-degree forced start can't be an open-trail endpoint without
    # duplicating a path to it; sometimes a CLOSED circuit from there is cheaper.
    # solve_route must compare and take the cheaper (postman.py even-branch).
    # Lollipop: cycle A-B-C-D-A (all even) + tail A-F, forced start C (even, far
    # from the odd nodes A & F). Closed-from-C deadhead is 1.0 (retrace the tail);
    # the open route would duplicate ~2.0 reaching an endpoint. Both routes are
    # valid trails preserving required_m, so ONLY this exact-deadhead assertion
    # catches a regression that drops the open-vs-closed comparison.
    G = nx.MultiGraph()
    for n, (x, y) in {"A": (0, 0), "B": (0, .001), "C": (.001, .001),
                      "D": (.001, 0), "F": (-.001, 0)}.items():
        G.add_node(n, x=x, y=y)
    for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "F")]:
        G.add_edge(a, b, length=1.0)
    sol = solve_route(G, open_route=True, force_start="C")
    assert sol["route"][0][0] == "C"
    assert math.isclose(sol["deadhead_m"], 1.0)   # optimal; pre-fix open route was 2.0
    assert_valid_trail(G, sol)


def test_forced_start_not_in_graph_does_not_crash():
    G = coord_grid(3)
    res = solve_route(G, open_route=True, force_start=("not", "a", "node"))
    assert_valid_trail(G, res)   # falls back to a normal valid route


def test_force_start_odd_leaf_is_optimal():
    import math
    G = nx.MultiGraph()
    for n, (x, y) in {1: (0, 0), 0: (0, -0.001), 4: (0.001, 0), 5: (-0.001, 0)}.items():
        G.add_node(n, x=x, y=y)
    G.add_edge(1, 0, length=4.0); G.add_edge(1, 4, length=4.0); G.add_edge(1, 5, length=5.0)
    sol = solve_route(G, open_route=True, force_start=0)
    assert sol["endpoints"][0] == 0
    assert math.isclose(sol["deadhead_m"], 4.0)   # optimal — was 12 (post-hoc duplication)
