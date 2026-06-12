"""
straightline.py — a boustrophedon ("lawnmower") router meant to be easy to
follow on foot.

Walking a city is not a mail-truck mileage problem. The Chinese-Postman solver
(postman.py) minimises distance but produces a block-by-block zig-zag that
crosses itself and is impossible to keep your place on — you get lost and end up
re-walking streets. This planner does the followable thing instead:

  * Pick ONE orientation — the N-S avenues OR the E-W streets, never both in the
    same walk (you do the cross-streets on a separate day).
  * Walk each whole street END TO END in a single straight shot.
  * Snake to the neighbouring parallel street one block over and walk it back.

You turn about twice per street instead of once per block, the route barely
crosses itself, and the mental model is trivial: "46th end to end, over a block,
45th end to end, ...". Output is the SAME dict shape solve_route returns, so
export/progress treat it identically. The cross-street connector blocks are the
only deadhead, and they get covered for real on the other-orientation walk.
"""

from __future__ import annotations

import math
import networkx as nx

from .postman import _bearing, _min_parallel_edge


# --------------------------------------------------------------------------- #
# grid orientation
# --------------------------------------------------------------------------- #
def _dominant_axes(G):
    """
    The two perpendicular street-grid bearings (deg, folded to [0,180)),
    returned as (ns_axis, ew_axis) where ns_axis is the one closer to true
    north-south.

    A street grid has TWO orthogonal orientations. Their *doubled* angles are
    180 deg apart and cancel, so we use the circular mean of *quadrupled* angles
    (N-S at 0 and E-W at 90 both map to 0 under x4), which recovers the grid
    orientation mod 90 even for a rotated grid.
    """
    mx = my = 0.0
    for u, v, _k in G.edges(keys=True):
        b = _bearing(G, u, v)
        if b is None:
            continue
        a = math.radians((b % 180.0) * 4.0)
        mx += math.cos(a)
        my += math.sin(a)
    phi = (math.degrees(math.atan2(my, mx)) / 4.0) % 90.0
    theta0, theta1 = phi, phi + 90.0      # the two grid axes in [0,180)

    def ns_ness(t):           # 0 = perfectly N-S, 90 = perfectly E-W
        return min(t, 180.0 - t)
    return (theta0, theta1) if ns_ness(theta0) <= ns_ness(theta1) else (theta1, theta0)


def _axis_dist(angle, axis):
    """Angular distance (0..90) of a folded bearing to a grid axis."""
    return abs(((angle - axis + 90.0) % 180.0) - 90.0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minkey(G, u, v, weight="length"):
    k, _ = _min_parallel_edge(G, u, v, weight)
    return k


def _enu(G, n, lat0, lon0):
    x = (float(G.nodes[n]["x"]) - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (float(G.nodes[n]["y"]) - lat0) * 110540.0
    return x, y


def _component_walk(comp, weight="length"):
    """
    A contiguous edge walk covering EVERY edge of one street-component, plus its
    two endpoints. A normal street is a simple path -> walked straight end to
    end. A rare branchy component falls back to the tested CPP solver so coverage
    is never dropped.
    Returns (edges[list of (u,v,key)], endpoint_a, endpoint_b).
    """
    ends = [n for n in comp if comp.degree(n) == 1]
    simple = len(ends) == 2 and all(comp.degree(n) <= 2 for n in comp)
    if simple:
        npath = nx.shortest_path(comp, ends[0], ends[1])
        edges = [(a, b, _minkey(comp, a, b, weight))
                 for a, b in zip(npath, npath[1:])]
        return edges, ends[0], ends[1]
    from .postman import solve_route
    sol = solve_route(comp, weight=weight, open_route=True)
    edges = [(u, v, k) for (u, v, k, _d) in sol["route"]]
    a, b = sol["endpoints"]
    return edges, a, b


def _reverse(edges):
    return [(v, u, k) for (u, v, k) in reversed(edges)]


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def _name_key(data):
    nm = data.get("name")
    if isinstance(nm, list):
        return tuple(sorted(nm))
    return nm


def plan_boustrophedon(G, start=None, target_km=8.0, axis="ns", weight="length",
                       straight_km=1.5, done=None):
    """
    Snake-walk one orientation. ``axis`` is "ns" (avenues) or "ew" (streets).
    A grid-aligned band ``straight_km`` long (in the walk direction) is taken
    starting at ``start``; every street crossing that band is walked end to end
    (so each straight shot is the same ~``straight_km`` and their ends line up,
    making the jog to the next street exactly one block). Streets are added
    perpendicular-outward until ~``target_km`` is covered. Returns the
    solve_route dict shape (+ 'n_streets').
    """
    if G.number_of_edges() == 0:
        return {"route": [], "required_m": 0.0, "deadhead_m": 0.0, "total_m": 0.0,
                "efficiency": 1.0, "n_odd_nodes": 0, "endpoints": (None, None),
                "n_streets": 0, "axis_deg": 0.0}

    ns_axis, ew_axis = _dominant_axes(G)
    target = ns_axis if axis == "ns" else ew_axis
    other = ew_axis if axis == "ns" else ns_axis

    if start is None or start not in G:
        start = next(iter(G.nodes))
    rad = math.radians(target)
    da = (math.sin(rad), math.cos(rad))     # unit vector ALONG the street (E, N)
    dp = (math.cos(rad), -math.sin(rad))    # unit vector PERPENDICULAR (E, N)
    lat0, lon0 = float(G.nodes[start]["y"]), float(G.nodes[start]["x"])

    def proj(n):
        E = (float(G.nodes[n]["x"]) - lon0) * math.cos(math.radians(lat0)) * 111320.0
        N = (float(G.nodes[n]["y"]) - lat0) * 110540.0
        return E * da[0] + N * da[1], E * dp[0] + N * dp[1]

    # orient the band toward the bulk of the area (away from the ocean)
    nn = G.number_of_nodes()
    ca = sum(proj(n)[0] for n in G.nodes) / nn
    cp = sum(proj(n)[1] for n in G.nodes) / nn
    sa = 1.0 if ca >= 0 else -1.0
    sp = 1.0 if cp >= 0 else -1.0
    H = straight_km * 1000.0

    # nodes inside the along-street band [0, H] from the start. TOL absorbs
    # float noise so nodes at the start's own cross-level (proj == 0 +/- eps)
    # aren't dropped, which would clip the first/last edge off each street.
    TOL = 1.0
    slab = {n for n in G.nodes if -TOL <= sa * proj(n)[0] <= H + TOL}

    # chosen-orientation, NAMED edges within the band, grouped by street name so
    # each whole avenue is one straight shot (unnamed footway scraps are left for
    # another day — they'd fragment the snake).
    from collections import defaultdict
    from .progress import edge_id, is_required
    done = done or set()
    by_name = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        if u in slab and v in slab and edge_id(G, u, v, k) not in done:
            b = _bearing(G, u, v)
            if b is None:
                continue
            if _axis_dist(b % 180.0, target) < _axis_dist(b % 180.0, other):
                if is_required(G[u][v][k]):     # canonical named ways only
                    by_name[_name_key(G[u][v][k])].append((u, v, k))

    streets = []
    for nm, es in by_name.items():
        sub = nx.MultiGraph()
        for (u, v, k) in es:
            sub.add_node(u, **G.nodes[u])
            sub.add_node(v, **G.nodes[v])
            sub.add_edge(u, v, key=k, **G[u][v][k])
        for nodes in nx.connected_components(sub):
            comp = sub.subgraph(nodes).copy()
            edges, ea, eb = _component_walk(comp, weight)
            if not edges:
                continue
            length = sum(min((d.get(weight, 0.0) for d in comp[u][v].values()),
                             default=0.0) for (u, v, k) in edges)
            pos = sp * (sum(proj(n)[1] for n in nodes) / len(nodes))
            streets.append({"name": nm, "edges": edges, "ends": (ea, eb),
                            "len": length, "pos": pos, "nodes": set(nodes)})
    if not streets:
        return plan_boustrophedon(G, start, target_km,
                                  "ew" if axis == "ns" else "ns", weight, straight_km)

    streets.sort(key=lambda s: s["pos"])   # from the start side outward

    # 3. choose the street nearest the start, then sweep outward (toward the side
    #    with more to cover) accumulating whole streets up to the target
    if start is not None and start in G:
        sx, sy = _enu(G, start, lat0, lon0)
        i0 = min(range(len(streets)),
                 key=lambda i: min((sx - _enu(G, n, lat0, lon0)[0]) ** 2
                                   + (sy - _enu(G, n, lat0, lon0)[1]) ** 2
                                   for n in streets[i]["nodes"]))
    else:
        i0 = 0
    left_len = sum(s["len"] for s in streets[:i0])
    right_len = sum(s["len"] for s in streets[i0 + 1:])
    step = 1 if right_len >= left_len else -1

    chosen, total_len, i = [], 0.0, i0
    while 0 <= i < len(streets):
        chosen.append(streets[i])
        total_len += streets[i]["len"]
        if total_len >= target_km * 1000.0:
            break
        i += step

    # 4. walk each chosen street end to end, entering at the nearer end (this is
    #    what makes it snake), with a shortest-path connector (deadhead) between
    cur = start if (start is not None and start in G) else chosen[0]["ends"][0]
    first = cur
    route, deadhead_m, required_m = [], 0.0, 0.0
    for st in chosen:
        dist, paths = nx.single_source_dijkstra(G, cur, weight=weight)
        ea, eb = st["ends"]
        entry = ea if dist.get(ea, math.inf) <= dist.get(eb, math.inf) else eb
        if entry != cur and entry in paths:
            p = paths[entry]
            for x, y in zip(p[:-1], p[1:]):
                k = _minkey(G, x, y, weight)
                deadhead_m += G[x][y][k].get(weight, 0.0)
                route.append((x, y, k, True))
        edges = st["edges"] if entry == ea else _reverse(st["edges"])
        for (u, v, k) in edges:
            required_m += min((d.get(weight, 0.0) for d in G[u][v].values()),
                              default=0.0)
            route.append((u, v, k, False))
        cur = eb if entry == ea else ea

    total_m = required_m + deadhead_m
    return {
        "route": route,
        "required_m": required_m,
        "deadhead_m": deadhead_m,
        "total_m": total_m,
        "efficiency": (required_m / total_m) if total_m else 1.0,
        "n_odd_nodes": 0,
        "endpoints": (first, cur),
        "n_streets": len(chosen),
        "axis_deg": round(target, 1),
    }
