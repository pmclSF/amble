"""
contour.py — hill handling.

Contour bands, descent-ordering, and an explicit spiral all did worse than
plain efficient coverage, because the straight-preferring Euler trail already
rides a hill's contour ring-roads as long swirls and puts the unavoidable turns
on the spurs. So hills don't get a special router — they get
``efficient`` coverage started at the TOP, plus an optional stair-preferred
"approach" path so the GPX can guide you up to the summit first.

What survives here: finding the top of the day's coverage, and a gentle
named-stairway-preferring climb to it.
"""

from __future__ import annotations

import networkx as nx

from .network import _is_steps
from .postman import _min_parallel_edge


def _z(G, n):
    return float(G.nodes[n].get("elevation", 0.0))


def highest_node(G):
    """The top of the day's coverage — where a hill walk should start."""
    return max(G.nodes, key=lambda n: _z(G, n))


def stair_preferred_ascent(G, start, summit, stair_discount=0.6,
                           climb_penalty=4.0, weight="length"):
    """
    Node path from ``start`` up to ``summit`` with a GENTLE preference for NAMED
    stairways on the climb (roads are fine). Directed costs make going up cost a
    mild climb penalty, but going up a *named* stairway (Vulcan-class) is
    discounted; descending is cheap. Unnamed/noisy steps get no special
    treatment — we don't trust that data. Returns the node path, or None.
    """
    D = nx.DiGraph()
    for u, v, k, d in G.edges(keys=True, data=True):
        L = d.get(weight, 0.0)
        named_stair = _is_steps(d) and bool(d.get("name"))
        for a, b in ((u, v), (v, u)):
            climb = max(0.0, _z(G, b) - _z(G, a))
            cost = L * stair_discount if named_stair else L + climb_penalty * climb
            if D.has_edge(a, b):
                D[a][b]["w"] = min(D[a][b]["w"], cost)
            else:
                D.add_edge(a, b, w=cost)
    try:
        return nx.shortest_path(D, start, summit, weight="w")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def ascent_edges(G, node_path, weight="length"):
    """Turn a node path into (u, v, key) edges via the shortest parallel edge."""
    out = []
    for x, y in zip(node_path[:-1], node_path[1:]):
        k, _ = _min_parallel_edge(G, x, y, weight)
        out.append((x, y, k))
    return out
