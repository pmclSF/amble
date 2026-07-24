"""
postman.py — Undirected Chinese Postman (Route Inspection) solver.

Given an undirected (multi)graph whose edges carry a numeric ``length``
attribute, produce a closed walk that traverses every edge AT LEAST once
while minimising the total length walked. Real street networks always have
odd-degree nodes (dead ends, T-junctions, the tops of staircases), so some
repetition is forced; this solver finds the least possible.

The algorithm (classic, polynomial-time for the undirected case):

  1. Find all odd-degree nodes. There is always an even number of them.
  2. Compute shortest-path distances between every pair of odd nodes
     (one Dijkstra per odd node — O(V) runs, not O(V^2)).
  3. Find a MINIMUM-WEIGHT PERFECT MATCHING on the odd nodes. Each matched
     pair tells us which odd nodes to "connect" by walking an existing path
     a second time.
  4. Duplicate the edges along each matched shortest path. The graph now has
     all-even degree => an Eulerian circuit exists.
  5. Extract the Eulerian circuit. That ordered edge list is the route.

The amount of duplicated length is the unavoidable "deadhead" distance.
"""

from __future__ import annotations

import itertools
import math
import networkx as nx


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _min_parallel_edge(G: nx.MultiGraph, u, v, weight: str = "length"):
    """Return (key, length) of the shortest edge between u and v in a MultiGraph."""
    best_key, best_len = None, float("inf")
    for key, data in G[u][v].items():
        w = data.get(weight, 1.0)
        if w < best_len:
            best_key, best_len = key, w
    return best_key, best_len


def required_length(G: nx.MultiGraph, weight: str = "length") -> float:
    """Total length of every edge that MUST be walked (each counted once)."""
    return sum(d.get(weight, 1.0) for _, _, d in G.edges(data=True))


def _bearing(G, u, v):
    """
    Compass bearing in degrees (0-360) walking from node u to node v, or None if
    the nodes lack 'x'/'y' coordinates (e.g. abstract test graphs) — callers then
    fall back to a name-only straightness preference.
    """
    try:
        y1 = math.radians(float(G.nodes[u]["y"]))
        y2 = math.radians(float(G.nodes[v]["y"]))
        dlon = math.radians(float(G.nodes[v]["x"]) - float(G.nodes[u]["x"]))
    except (KeyError, ValueError, TypeError):
        return None
    x = math.sin(dlon) * math.cos(y2)
    y = math.cos(y1) * math.sin(y2) - math.sin(y1) * math.cos(y2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _edge_names(data):
    nm = data.get("name")
    if isinstance(nm, list):
        return set(nm)
    return {nm} if nm else set()


def _euler_trail_straight(H, start):
    """
    An Eulerian trail/circuit of H starting at ``start`` that prefers to go
    STRAIGHT through each intersection — staying on the same-named street, else
    the smallest turn angle. Hierholzer's algorithm is correct for *any*
    next-edge choice, so steering that choice toward straight continuations
    yields a route with the same total length but far fewer turns (nicer to walk
    and to give directions for). Returns an ordered list of (u, v, key).
    """
    adj = {n: [] for n in H.nodes}
    edges = []
    for u, v, k in H.edges(keys=True):
        eid = len(edges)
        edges.append((u, v, k))
        adj[u].append((eid, v, k))
        adj[v].append((eid, u, k))
    used = [False] * len(edges)

    stack = [start]
    edge_stack = []          # edge used to arrive at each non-start stack node
    trail = []
    while stack:
        v = stack[-1]
        inc_b, inc_names = None, set()
        if edge_stack:
            pu, _, pk = edge_stack[-1]
            inc_b = _bearing(H, pu, v)
            inc_names = _edge_names(H[pu][v][pk])
        best, best_score = None, None
        for eid, w, k in adj[v]:
            if used[eid]:
                continue
            same = bool(inc_names & _edge_names(H[v][w][k]))
            dev = 0.0
            if inc_b is not None:
                out_b = _bearing(H, v, w)
                if out_b is not None:
                    dev = abs(((out_b - inc_b + 180.0) % 360.0) - 180.0)  # 0=straight
            score = dev - (1000.0 if same else 0.0)  # same street always wins
            if best_score is None or score < best_score:
                best, best_score = (eid, w, k), score
        if best is None:
            stack.pop()
            if edge_stack:
                trail.append(edge_stack.pop())
        else:
            eid, w, k = best
            used[eid] = True
            stack.append(w)
            edge_stack.append((v, w, k))
    trail.reverse()
    return trail


# --------------------------------------------------------------------------- #
# main solver
# --------------------------------------------------------------------------- #
def solve_route(
    G: nx.MultiGraph,
    weight: str = "length",
    source=None,
    open_route: bool = True,
    force_start=None,
    _skip_even=False,
):
    """
    Solve the undirected Chinese Postman Problem on a CONNECTED MultiGraph.

    Parameters
    ----------
    G          : connected undirected MultiGraph with numeric ``weight`` on edges
    weight     : edge attribute holding length (default "length")
    source     : node to start at for a CLOSED circuit (ignored for open routes)
    force_start: node the route MUST begin at. For an open route this overrides
                 the solver's free choice of start endpoint; if that node isn't
                 already a natural endpoint, the minimal connector path to the
                 nearest one is duplicated (the real walk from where you stand to
                 where the efficient sweep begins). For a closed circuit it just
                 sets where the loop opens.
    open_route : if True, allow the walk to START and END at DIFFERENT nodes
                 (a "route" rather than a closed "circuit"). This is almost
                 always cheaper for real walks — you rarely need to loop back
                 to your start. The solver automatically picks the two endpoints
                 that save the most backtracking. If False, returns a closed
                 circuit that ends where it began.

    Returns
    -------
    dict with:
      route        : list of (u, v, key, is_deadhead) in traversal order
      required_m   : metres of unique street (the stuff you actually want to cover)
      deadhead_m   : extra metres walked twice (the unavoidable backtracking)
      total_m      : required_m + deadhead_m
      efficiency   : required_m / total_m  (1.0 == perfect, no repeats)
      n_odd_nodes  : how many odd-degree nodes existed before balancing
      endpoints    : (start, end) nodes of the route
    """
    if G.number_of_edges() == 0:
        return {
            "route": [], "required_m": 0.0, "deadhead_m": 0.0,
            "total_m": 0.0, "efficiency": 1.0, "n_odd_nodes": 0,
            "endpoints": (None, None),
        }
    if not nx.is_connected(G):
        raise ValueError(
            "Graph is not connected. Split it into components first "
            "(see network.connected_chunks)."
        )

    # An EVEN-degree force_start can't be a trail endpoint without duplicating a
    # path to it — but sometimes a CLOSED circuit starting there is cheaper than
    # that open route. Solve both and keep the lower-deadhead one. (Odd starts are
    # handled optimally inside the matching, so they skip this.)
    if (open_route and not _skip_even and force_start is not None
            and force_start in G and G.degree(force_start) % 2 == 0):
        opened = solve_route(G, weight, source, True, force_start, _skip_even=True)
        closed = solve_route(G, weight, force_start, False, force_start)
        return closed if closed["deadhead_m"] < opened["deadhead_m"] - 1e-9 else opened

    req_m = required_length(G, weight)

    # 1. odd-degree nodes ---------------------------------------------------- #
    odd_nodes = [n for n, deg in G.degree() if deg % 2 == 1]
    if len(odd_nodes) > 600:
        import sys
        print(f"  [warning] {len(odd_nodes)} odd nodes — the O(k^3) matching may "
              f"take a while; consider a smaller --target-km or area.",
              file=sys.stderr)

    # work on a copy we can augment with duplicate edges
    H = nx.MultiGraph(G)

    deadhead_m = 0.0
    freed = []  # odd nodes left unpaired -> become the open route's endpoints
    if odd_nodes:
        # 2. shortest paths between odd nodes (one Dijkstra per odd node) ---- #
        dist, paths = {}, {}
        odd_set = set(odd_nodes)
        for a in odd_nodes:
            d, p = nx.single_source_dijkstra(G, a, weight=weight)
            dist[a] = {b: d[b] for b in odd_set if b in d}
            paths[a] = {b: p[b] for b in odd_set if b in p}

        # 3. minimum-weight matching ---------------------------------------- #
        # networkx MAXIMISES weight, so transform true cost c -> (max_d - c) + 1
        # which is positive and strictly decreasing in c. A hypothetical
        # zero-cost edge maps to the largest weight (max_d + 1).
        max_d = max(
            (dist[a][b] for a in odd_nodes for b in dist[a] if a != b),
            default=1.0,
        )
        zero_cost_w = max_d + 1.0

        K = nx.Graph()
        for a, b in itertools.combinations(odd_nodes, 2):
            if b in dist[a]:
                K.add_edge(a, b, match_weight=(max_d - dist[a][b]) + 1.0)

        if open_route and len(odd_nodes) >= 2:
            # Two dummy terminals at zero cost. In the matching they either pair
            # with each other (=> closed circuit is optimal) or each "free" one
            # odd node, which becomes a route endpoint with NO duplicated path.
            t1, t2 = ("__T1__", "__T2__")
            if force_start is not None and force_start in odd_set:
                # FORCE force_start to be a freed endpoint, chosen optimally by
                # the matcher: t1 frees only force_start, t2 frees the best other,
                # and there's no t1-t2 escape. This avoids the post-hoc path
                # duplication (step 5), which produced sub-optimal deadhead.
                K.add_edge(t1, force_start, match_weight=zero_cost_w)
                for n in odd_nodes:
                    if n != force_start:
                        K.add_edge(t2, n, match_weight=zero_cost_w)
            else:
                for n in odd_nodes:
                    K.add_edge(t1, n, match_weight=zero_cost_w)
                    K.add_edge(t2, n, match_weight=zero_cost_w)
                K.add_edge(t1, t2, match_weight=zero_cost_w)

        matching = nx.max_weight_matching(
            K, maxcardinality=True, weight="match_weight"
        )

        # 4. duplicate edges along each matched real path ------------------- #
        terminals = {"__T1__", "__T2__"}
        for a, b in matching:
            if a in terminals and b in terminals:
                continue  # T1-T2 matched: no freed endpoints, closed route
            if a in terminals or b in terminals:
                freed.append(b if a in terminals else a)
                continue  # freed odd node, no duplication
            node_path = paths[a][b]
            deadhead_m += dist[a][b]
            for x, y in zip(node_path[:-1], node_path[1:]):
                key, _ = _min_parallel_edge(G, x, y, weight)
                data = dict(G[x][y][key])
                data["deadhead"] = True
                data["_source_key"] = key
                H.add_edge(x, y, **data)

    # 5. (optional) force the route to START at a specific node -------------- #
    # If force_start isn't already one of the two trail endpoints, make it one
    # by duplicating the shortest path to the nearer endpoint. That added length
    # is genuine: it's the walk from your chosen start to where the cheapest
    # sweep begins, so we count it as deadhead.
    if force_start is not None and force_start in H:
        cur_odd = [n for n, deg in H.degree() if deg % 2 == 1]
        if len(cur_odd) == 2 and force_start not in cur_odd:
            d, p = nx.single_source_dijkstra(G, force_start, weight=weight)
            tgt = min(cur_odd, key=lambda n: d.get(n, float("inf")))
            node_path = p[tgt]
            deadhead_m += d[tgt]
            for x, y in zip(node_path[:-1], node_path[1:]):
                key, _ = _min_parallel_edge(G, x, y, weight)
                data = dict(G[x][y][key])
                data["deadhead"] = True
                data["_source_key"] = key
                H.add_edge(x, y, **data)

    # 6. Eulerian path or circuit (straight-preferring) --------------------- #
    remaining_odd = [n for n, deg in H.degree() if deg % 2 == 1]
    if remaining_odd:
        if force_start is not None and force_start in remaining_odd:
            start = force_start
        else:
            start = freed[0] if freed else remaining_odd[0]
    else:
        start = force_start if (force_start is not None and force_start in H) else source
        if start is None or start not in H:
            start = next(iter(H.nodes))
    circuit_edges = _euler_trail_straight(H, start)
    # recompute true endpoints from the path itself (robust)
    endpoints = ((circuit_edges[0][0], circuit_edges[-1][1])
                 if circuit_edges else (start, start))

    route = []
    for u, v, key in circuit_edges:
        is_dead = bool(H[u][v][key].get("deadhead", False))
        # A duplicate has a synthetic MultiGraph key in H.  Export the exact
        # physical source key so GPX/GeoJSON never fall back to a different
        # parallel carriageway, road, or staircase.
        out_key = H[u][v][key].get("_source_key", key)
        route.append((u, v, out_key, is_dead))

    total_m = req_m + deadhead_m
    return {
        "route": route,
        "required_m": req_m,
        "deadhead_m": deadhead_m,
        "total_m": total_m,
        "efficiency": (req_m / total_m) if total_m else 1.0,
        "n_odd_nodes": len(odd_nodes),
        "endpoints": endpoints,
    }
