"""
progress.py — remember which streets you have already walked, across years.

Edges are identified by a UNIQUE key (see edge_id): sorted OSM node ids + key.
It's collision-free (so walking one street never falsely marks another "done"),
and stable for the same .graphml. Re-fetching changes node ids, so progress is
carried across a re-fetch by `migrate` (a tolerant MATCH on name + nearest
midpoint, see rekey_store/_best_match), and rolled up city-wide by `total` the
same way — fuzzy matching is right there, but wrong for identity.

The store is plain JSON — easy to inspect, back up, or edit by hand.
"""

from __future__ import annotations

import json
import os
import datetime as _dt
import networkx as nx


def _edge_name(data) -> str:
    nm = data.get("name")
    if isinstance(nm, list):
        nm = nm[0] if nm else ""
    return nm or ""


def edge_id(G, u, v, key) -> str:
    """
    A UNIQUE, collision-free identity for an edge: sorted OSM node ids + the
    parallel key. Stable as long as you reuse the same .graphml. (G is accepted
    for signature symmetry with the other edge helpers but isn't needed here.)

    Why not a geometry hash: a lossy hash (rounded coords + length bucket) ALIASES
    distinct streets, so walking one would falsely mark another "done" — a silent,
    permanent coverage hole. Identity must be exact. Re-fetch durability is
    handled separately by `migrate` (a tolerant MATCH on name + nearest midpoint,
    see rekey_store/_best_match) — fuzzy matching is right there, wrong for identity.
    """
    a, b = sorted((u, v), key=str)
    return f"{a}-{b}-{key}"


def is_required(data) -> bool:
    """
    The project's must-walk set is every NAMED way (street, alley, stairway,
    named park path). Unnamed ways (sidewalks, crossings, desire lines, minor
    connectors) are walked only as connectors and don't count toward 100%.

    Exception: a declared CORRIDOR (e.g. the Great Highway's four overlapping
    names — see equivalents.py) counts ONCE. Only its canonical name is required;
    the redundant aliases are optional connectors, so you walk the strip once
    instead of four times. Everything not declared keeps the plain "named" rule.
    """
    from .equivalents import canonical_name
    name = _edge_name(data)
    return bool(name) and canonical_name(name) == name


def _midpoint(G, u, v):
    return ((float(G.nodes[u]["y"]) + float(G.nodes[v]["y"])) / 2.0,
            (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"])) / 2.0)


def _geo_dist_m(p, q):
    import math
    dlat = (q[0] - p[0]) * 110540.0
    dlon = (q[1] - p[1]) * 111320.0 * math.cos(math.radians((p[0] + q[0]) / 2.0))
    return math.hypot(dlat, dlon)


def _edge_index(G):
    """name -> list of (edge_id, midpoint, length), for tolerant matching."""
    from collections import defaultdict
    idx = defaultdict(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        idx[_edge_name(d)].append(
            (edge_id(G, u, v, k), _midpoint(G, u, v), d.get("length", 0.0)))
    return idx


def _best_match(name, mid, length, index, tol_m=40.0):
    """The new edge whose name matches and whose midpoint is nearest within
    tol_m (and length within ~35%) — a tolerant window, not a hard bin, so it
    survives sub-metre coordinate drift and length recalcs across a re-fetch."""
    best, best_d = None, tol_m
    for (eid, nmid, nlen) in index.get(name, ()):
        d = _geo_dist_m(mid, nmid)
        if d < best_d and abs(nlen - length) <= max(20.0, 0.35 * length):
            best, best_d = eid, d
    return best


def rekey_store(store: dict, Gold, Gnew):
    """
    Carry a store across a re-fetch: re-key walked records from Gold's node-ids to
    Gnew's by tolerant geometry MATCH (name + nearest midpoint). Returns
    (new_store, migrated, not_found, merged). Two old records matching one new
    edge are MERGED (kept once), never silently overwritten.
    """
    new_idx = _edge_index(Gnew)
    old_geo = {edge_id(Gold, u, v, k):
               (_edge_name(d), _midpoint(Gold, u, v), d.get("length", 0.0))
               for u, v, k, d in Gold.edges(keys=True, data=True)}
    walked, migrated, not_found, merged = {}, 0, 0, 0
    for key, rec in store.get("walked", {}).items():
        info = old_geo.get(key)
        match = _best_match(*info, new_idx) if info else None
        if match is None:
            not_found += 1
        elif match in walked:
            merged += 1
        else:
            walked[match] = rec
            migrated += 1
    return {"walked": walked}, migrated, not_found, merged


def load_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"walked": {}}


def save_store(store: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


def mark_route_walked(store: dict, G, route, when: str | None = None, note=""):
    """Mark every NON-deadhead edge in a solved route as walked."""
    when = when or _dt.date.today().isoformat()
    n = 0
    for (u, v, key, dead) in route:
        if dead:
            continue  # a repeat of a street already covered by its first pass
        eid = edge_id(G, u, v, key)
        if eid not in store["walked"]:
            store["walked"][eid] = {"date": when, "note": note}
            n += 1
    return n


def mark_edges_walked(store: dict, edge_ids, when: str | None = None, note=""):
    when = when or _dt.date.today().isoformat()
    n = 0
    for eid in edge_ids:
        if eid not in store["walked"]:
            store["walked"][eid] = {"date": when, "note": note}
            n += 1
    return n


def walked_id_set(store: dict) -> set:
    return set(store.get("walked", {}).keys())


def remaining_subgraph(G: nx.MultiGraph, store: dict,
                       required_only: bool = False) -> nx.MultiGraph:
    """
    A MultiGraph of edges you have NOT yet walked. With ``required_only`` it
    keeps just the unwalked NAMED ways — the coverage target — which is what the
    routers chew on (they use the full graph for connectors separately).
    """
    walked = walked_id_set(store)
    R = nx.MultiGraph()
    for u, v, key, data in G.edges(keys=True, data=True):
        if required_only and not is_required(data):
            continue
        if edge_id(G, u, v, key) not in walked:
            R.add_node(u, **G.nodes[u])
            R.add_node(v, **G.nodes[v])
            R.add_edge(u, v, key=key, **data)
    return R


def stats(G: nx.MultiGraph, store: dict) -> dict:
    """Progress over the must-walk set (named ways only); 100% = every named
    way walked. Unnamed connectors are ignored here."""
    walked = walked_id_set(store)
    total_m = walked_m = 0.0
    total_e = walked_e = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        if not is_required(data):
            continue
        L = data.get("length", 0.0)
        total_m += L
        total_e += 1
        if edge_id(G, u, v, key) in walked:
            walked_m += L
            walked_e += 1
    return {
        "total_km": total_m / 1000.0,
        "walked_km": walked_m / 1000.0,
        "remaining_km": (total_m - walked_m) / 1000.0,
        "total_edges": total_e,
        "walked_edges": walked_e,
        "pct_done": (walked_m / total_m * 100.0) if total_m else 0.0,
    }
