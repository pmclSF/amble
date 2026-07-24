"""
Tests for postman._euler_trail_straight (the straight-preferring Eulerian
extraction) and, crucially, the trail ORDERING. total_m is computed
analytically and a plain at-least-once coverage check can't see ordering, so a
non-contiguous or edge-doubling trail would be invisible without assert_valid_trail.
"""
import networkx as nx
import pytest

from amble import postman
from amble.postman import solve_route, _bearing
from tests.helpers import coord_grid, assert_valid_trail, _canon


def test_trail_valid_on_grid():
    G = coord_grid(4)
    assert_valid_trail(G, solve_route(G, open_route=True))
    assert_valid_trail(G, solve_route(G, open_route=False))


def test_valid_trail_helper_catches_non_contiguous():
    # The helper must actually fail on a broken trail, or it proves nothing.
    G = coord_grid(3)
    res = solve_route(G, open_route=True)
    broken = dict(res)
    r = list(res["route"])
    r[2], r[5] = r[5], r[2]  # shuffle => teleport between non-adjacent edges
    broken["route"] = r
    with pytest.raises(AssertionError):
        assert_valid_trail(G, broken)


def test_valid_trail_helper_catches_dropped_edge():
    G = coord_grid(3)
    res = solve_route(G, open_route=True)
    broken = dict(res)
    broken["route"] = [e for e in res["route"] if e[3]] + \
        [e for e in res["route"] if not e[3]][:-1]  # drop one required edge
    with pytest.raises(AssertionError):
        assert_valid_trail(G, broken)


def test_dead_end_duplicates_are_deadhead_not_extra_coverage():
    # path A-B-C closed: each edge walked twice, the 2nd time as deadhead.
    G = nx.MultiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.001, y=0.0)
    G.add_node("C", x=0.002, y=0.0)
    G.add_edge("A", "B", length=1.0, name="Main")
    G.add_edge("B", "C", length=1.0, name="Main")
    res = solve_route(G, open_route=False)
    assert_valid_trail(G, res)  # each original edge exactly once as non-deadhead
    deadhead = [e for e in res["route"] if e[3]]
    assert len(deadhead) == 2, "the two return traversals must be flagged deadhead"


def test_parallel_keys_both_in_trail():
    G = nx.MultiGraph()
    G.add_node("X", x=0.0, y=0.0)
    G.add_node("Y", x=0.001, y=0.0)
    G.add_edge("X", "Y", length=2.0, name="road")    # key 0
    G.add_edge("X", "Y", length=3.0, name="stairs")  # key 1
    res = solve_route(G, open_route=True)
    keys = {k for (u, v, k, d) in res["route"] if frozenset((u, v)) == frozenset(("X", "Y"))}
    assert keys == {0, 1}, f"both parallel streets must be covered, got keys {keys}"


def test_deadhead_duplicate_keeps_exact_physical_parallel_key():
    G = nx.MultiGraph()
    G.add_edge("A", "B", key=0, length=100.0, name="long")
    G.add_edge("A", "B", key=5, length=50.0, name="medium")
    G.add_edge("A", "B", key=7, length=10.0, name="short")
    result = solve_route(G, open_route=False)
    dead = [e for e in result["route"] if e[3]]
    assert dead == [("B", "A", 7, True)] or dead == [("A", "B", 7, True)]


def _sharp_turns(G, route):
    n = 0
    for (a, b, _k, _d), (c, d, _k2, _d2) in zip(route, route[1:]):
        b1, b2 = _bearing(G, a, b), _bearing(G, c, d)
        if b1 is None or b2 is None:
            continue
        if abs(((b2 - b1 + 180) % 360) - 180) > 30:
            n += 1
    return n


def test_straight_preferring_has_strictly_fewer_turns_than_networkx():
    G = coord_grid(5)
    straight = solve_route(G, open_route=True)["route"]
    orig = postman._euler_trail_straight
    try:
        postman._euler_trail_straight = \
            lambda H, start: list(nx.eulerian_path(H, source=start, keys=True))
        arbitrary = solve_route(G, open_route=True)["route"]
    finally:
        postman._euler_trail_straight = orig
    # STRICT: the straight-preferring trail must have strictly FEWER sharp turns
    # (measured 19 vs 38 on a 5x5 grid). A <= check passed even if the
    # straightener did nothing (equal counts), so it proved nothing.
    assert _sharp_turns(G, straight) < _sharp_turns(G, arbitrary)
