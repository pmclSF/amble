"""
coverage.py — Rural-Postman coverage of the must-walk (named) ways.

The earlier efficient planner induced a NAMED-ONLY subgraph and ran the Chinese
Postman on its largest connected component. But named ways connect *through*
unnamed paths, so the named-only graph fractures into islands (Buena Vista: 22
of them) — and the planner could only ever reach the biggest, stranding the
rest and disagreeing with the progress denominator.

This solves the right problem: the **Rural Postman Problem** — cover a REQUIRED
subset of edges (the unwalked named ways) while using the FULL connected graph
for connectors. Named islands are stitched together with shortest paths through
unnamed pavement, so one walk is geographically continuous and every named way
is reachable. Output is the solve_route dict shape; required = named coverage,
everything else (connectors, backtracking) = deadhead.
"""

from __future__ import annotations

import networkx as nx

from .postman import solve_route, _min_parallel_edge
from .progress import edge_id, is_required


def _canon(u, v, k):
    a, b = sorted((u, v), key=str)
    return (a, b, k)


def _len(G, u, v, k, weight):
    if v in G[u] and k in G[u][v]:
        return G[u][v][k].get(weight, 0.0)
    return min((d.get(weight, 0.0) for d in G[u][v].values()), default=0.0)


def plan_coverage(G, walked_ids, start=None, target_km=8.0, weight="length"):
    """
    Rural-Postman coverage of unwalked NAMED ways near ``start``, using the full
    connected graph ``G`` for connectors. ``start`` may be the highest unwalked
    point (hills, "work down") or your location. Returns the solve_route dict
    shape (route, required_m = named covered, deadhead_m = connectors+repeats,
    total_m, efficiency, endpoints, n_streets).

    ``target_km`` is APPROXIMATE — it bounds the named + connector estimate, but
    the Chinese-Postman pass adds some backtracking on top, so the actual walk
    can run ~1.2-1.8x the target where named ways are sparse islands (a park).
    Dial it down for a shorter day.
    """
    req = {_canon(u, v, k): (u, v, k)
           for u, v, k, d in G.edges(keys=True, data=True)
           if is_required(d) and edge_id(G, u, v, k) not in walked_ids}
    if not req:
        return {"route": [], "required_m": 0.0, "deadhead_m": 0.0, "total_m": 0.0,
                "efficiency": 1.0, "n_odd_nodes": 0, "endpoints": (None, None),
                "n_streets": 0}

    if start is None or start not in G:
        start = next(iter(req.values()))[0]

    # Grow the day INCREMENTALLY, bounding TOTAL walking (named + connectors) by
    # the target: repeatedly hop to the nearest unwalked named edge, adding the
    # connector path and the edge, until ~target_km of total. This keeps the day
    # continuous AND a sensible length even where named ways are sparse islands.
    inf = float("inf")
    target_m = max(target_km, 0.1) * 1000.0
    # Grow to ~0.65x the target of NAMED+connectors; the Chinese-Postman pass
    # below then adds backtracking, landing the actual walk near target_km. (It's
    # still approximate — see the docstring.)
    grow_m = 0.65 * target_m
    H = nx.MultiGraph()
    H.add_node(start, **G.nodes[start])
    blob = {start}
    required_set, total = set(), 0.0
    remaining = dict(req)
    cut_seed = 300.0                     # adaptive Dijkstra cutoff (grows if islands are far)
    while total < grow_m and remaining:
        # Bound the Dijkstra to ~the day's scale: the next island is almost
        # always near the growing blob, so a cutoff explores only the local area
        # (big speedup on a city-size graph). Fall back to a full search only if
        # nothing is within the cutoff (a far straggler at the end of the day).
        def _nearest(cut):
            dd_, pp_ = nx.multi_source_dijkstra(G, blob, weight=weight, cutoff=cut)
            bc, be, bd = None, None, inf
            for c, (u, v, k) in remaining.items():
                du, dv = dd_.get(u, inf), dd_.get(v, inf)
                nn, ddist = (u, du) if du <= dv else (v, dv)
                if ddist < bd:
                    bc, be, bd = c, nn, ddist
            return bc, be, bd, pp_
        best_c = None
        for cut in (cut_seed, cut_seed * 6.0, None):   # seeded by the last hop
            best_c, best_entry, best_d, paths = _nearest(cut)
            if best_c is not None:
                cut_seed = max(300.0, 3.0 * best_d)
                break
        if best_c is None:
            break                       # nothing else reachable
        # cap the overshoot: don't tack on a far/huge island once we already have
        # a day's worth (the connector is walked ~twice). The first island is
        # always taken so a walk is never empty.
        u, v, k = remaining[best_c]
        projected = total + 2.0 * best_d + _len(G, u, v, k, weight)
        if required_set and projected > 1.8 * target_m:
            break
        remaining.pop(best_c)
        for x, y in zip(paths[best_entry][:-1], paths[best_entry][1:]):
            kk, _ = _min_parallel_edge(G, x, y, weight)
            if not H.has_edge(x, y, kk):
                H.add_node(x, **G.nodes[x]); H.add_node(y, **G.nodes[y])
                H.add_edge(x, y, key=kk, **G[x][y][kk])
                # a connector into an island is typically walked twice (in & out)
                total += 2.0 * _len(G, x, y, kk, weight)
        H.add_node(u, **G.nodes[u]); H.add_node(v, **G.nodes[v])
        H.add_edge(u, v, key=k, **G[u][v][k])
        required_set.add(best_c)
        total += _len(G, u, v, k, weight)
        blob |= set(paths[best_entry]) | {u, v}

    # 4. Chinese-Postman the connected structure, starting at the top/your spot.
    sol = solve_route(H, weight=weight, open_route=True,
                      force_start=start if start in H else None)

    # 5. re-flag: a traversal counts as COVERAGE only if it's a required named
    #    edge on its first pass; connectors and repeats are deadhead.
    covered, route, req_m, dead_m = set(), [], 0.0, 0.0
    for (u, v, k, _solver_dead) in sol["route"]:
        c = _canon(u, v, k)
        L = _len(G, u, v, k, weight)
        if not _solver_dead and c in required_set and c not in covered:
            covered.add(c); req_m += L; route.append((u, v, k, False))
        else:
            dead_m += L; route.append((u, v, k, True))

    total = req_m + dead_m
    names = set()
    for (u, v, k, dd) in route:
        if dd:
            continue
        nm = G[u][v][k].get("name") if (v in G[u] and k in G[u][v]) else None
        if isinstance(nm, list):
            nm = nm[0] if nm else None
        if nm:
            names.add(nm)
    return {
        "route": route, "required_m": req_m, "deadhead_m": dead_m,
        "total_m": total, "efficiency": (req_m / total) if total else 1.0,
        "n_odd_nodes": sol.get("n_odd_nodes", 0),
        "endpoints": (route[0][0], route[-1][1]) if route else (start, start),
        "n_streets": len(names),
    }
