"""
straightline.plan_boustrophedon — the snake router that walks whole streets of
ONE orientation end to end. Unlike the CPP solver it deliberately covers a
subset (one axis, named streets), so we check contiguity + that it covers
exactly the chosen orientation, not full coverage.
"""
from collections import Counter

from amble import straightline
from amble.straightline import plan_boustrophedon, _dominant_axes
from tests.helpers import coord_grid, _canon


def _assert_contiguous(route):
    for (u, v, k, _d), (u2, v2, k2, _d2) in zip(route, route[1:]):
        assert v == u2, f"snake jumped from {v} to {u2}"


def test_dominant_axes_on_aligned_grid():
    ns, ew = _dominant_axes(coord_grid(5))
    assert min(ns, 180 - ns) < 5          # ~north-south
    assert abs(ew - 90) < 5               # ~east-west


def test_snake_is_contiguous_and_covers_one_orientation():
    G = coord_grid(5)                     # cols run N-S ("colC"), rows E-W ("rowR")
    sol = plan_boustrophedon(G, target_km=999, axis="ns", straight_km=2.0)
    _assert_contiguous(sol["route"])

    required = Counter(_canon(u, v, k) for (u, v, k, d) in sol["route"] if not d)
    col_edges = {_canon(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                 if d.get("name", "").startswith("col")}
    # every N-S edge is walked once as a "street"; nothing walked twice as street
    assert set(required) == col_edges
    assert max(required.values()) == 1
    # no E-W ("row") edge is walked as a street (those are for another day)
    row_edges = {_canon(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                 if d.get("name", "").startswith("row")}
    assert not (set(required) & row_edges)


def test_snake_counts_whole_streets():
    G = coord_grid(5)
    sol = plan_boustrophedon(G, target_km=999, axis="ns", straight_km=2.0)
    assert sol["n_streets"] == 5          # five whole avenues (col0..col4)


def test_snake_ew_axis_picks_rows():
    G = coord_grid(5)
    sol = plan_boustrophedon(G, target_km=999, axis="ew", straight_km=2.0)
    required = {_canon(u, v, k) for (u, v, k, d) in sol["route"] if not d}
    row_edges = {_canon(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                 if d.get("name", "").startswith("row")}
    assert required == row_edges


def test_snake_target_limits_streets_walked():
    G = coord_grid(8)
    full = plan_boustrophedon(G, target_km=999, axis="ns", straight_km=3.0)
    small = plan_boustrophedon(G, target_km=0.001, axis="ns", straight_km=3.0)
    assert full["n_streets"] == 8       # all 8 avenues with a huge target
    # a tiny target walks exactly ONE whole avenue (a street is never split).
    # ">= 1 and < full" passed for any 1..7; pin the exact floor instead.
    assert small["n_streets"] == 1
