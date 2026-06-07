"""
Consolidated hill + named-scope pieces: stair-preferred ascent, highest-node,
named-only remaining set, and the snake honoring already-walked streets.
"""
import networkx as nx

from amble import progress as prog
from amble.contour import stair_preferred_ascent, highest_node, _z
from amble.straightline import plan_boustrophedon
from amble.progress import edge_id
from tests.helpers import coord_grid


def test_ascent_prefers_named_stairway_over_road():
    G = nx.MultiGraph()
    for n, (x, y, z) in {"B": (0, 0, 0), "T": (0.002, 0, 40)}.items():
        G.add_node(n, x=x, y=y, elevation=z)
    # a named stairway B->S->T
    G.add_node("S", x=0.001, y=0.0, elevation=20)
    G.add_edge("B", "S", length=50.0, highway="steps", name="Vulcan Stairway")
    G.add_edge("S", "T", length=50.0, highway="steps", name="Vulcan Stairway")
    # an unnamed road B->R->T of similar length but no steps
    G.add_node("R", x=0.001, y=0.001, elevation=20)
    G.add_edge("B", "R", length=55.0)
    G.add_edge("R", "T", length=55.0)
    path = stair_preferred_ascent(G, "B", "T")
    assert "S" in path and "R" not in path     # climbed the stairway


def test_unnamed_steps_are_not_preferred():
    # noisy/unnamed steps get no special treatment — a shorter road wins
    G = nx.MultiGraph()
    for n, (x, y, z) in {"B": (0, 0, 0), "T": (0.002, 0, 40)}.items():
        G.add_node(n, x=x, y=y, elevation=z)
    G.add_node("S", x=0.001, y=0.0, elevation=20)
    G.add_edge("B", "S", length=200.0, highway="steps")   # unnamed steps, long
    G.add_edge("S", "T", length=200.0, highway="steps")
    G.add_node("R", x=0.001, y=0.001, elevation=20)
    G.add_edge("B", "R", length=40.0)
    G.add_edge("R", "T", length=40.0)
    assert "R" in stair_preferred_ascent(G, "B", "T")


def test_highest_node():
    G = nx.MultiGraph()
    for n, z in (("a", 5.0), ("b", 50.0), ("c", 20.0)):
        G.add_node(n, x=0.0, y=0.0, elevation=z)
    G.add_edge("a", "b", length=1.0)
    G.add_edge("b", "c", length=1.0)
    assert highest_node(G) == "b"
    assert _z(G, "b") == 50.0


def test_remaining_required_only_excludes_unnamed():
    G = nx.MultiGraph()
    for n,(x,y) in {1:(0,0),2:(0,.001),3:(0,.002),4:(0,.003)}.items(): G.add_node(n,x=x,y=y)
    G.add_edge(1, 2, length=10.0, name="Real St")
    G.add_edge(3, 4, length=10.0)               # unnamed connector
    R = prog.remaining_subgraph(G, {"walked": {}}, required_only=True)
    assert R.number_of_edges() == 1


def test_snake_skips_already_walked_streets():
    G = coord_grid(5)
    full = plan_boustrophedon(G, target_km=999, axis="ns", straight_km=2.0)
    assert full["n_streets"] == 5
    done = {edge_id(G, u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
            if d.get("name") == "col0"}
    after = plan_boustrophedon(G, target_km=999, axis="ns", straight_km=2.0, done=done)
    assert after["n_streets"] == 4               # col0 already done, skipped
