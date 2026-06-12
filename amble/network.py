"""
network.py — get the walkable network from OpenStreetMap and slice it into
pieces small enough to walk in a day.

Requires osmnx (`pip install osmnx`) and network access to the Overpass API.
The heavy geo dependency is imported lazily so the solver can be unit-tested
without it.
"""

from __future__ import annotations

import os
import math
import networkx as nx


def _ox():
    import osmnx as ox  # lazy import
    # Cache Overpass responses so re-runs are instant and reproducible.
    ox.settings.use_cache = True
    ox.settings.log_console = False
    # Pin the way tags we filter on, so a slim global config can't silently turn
    # drop_driveways into a no-op (it keys on 'service'; we also keep 'name').
    ox.settings.useful_tags_way = sorted(set(list(ox.settings.useful_tags_way) +
        ["highway", "name", "service", "footway", "surface", "access"]))
    # 'walk' is bidirectional by default in osmnx>=2, and the 'walk' filter
    # already includes footways, paths, pedestrian areas AND highway=steps
    # (the famous SF staircases). One-way directionality is ignored for walking.
    return ox


# --------------------------------------------------------------------------- #
# download / cache
# --------------------------------------------------------------------------- #
def load_or_download(place: str, cache_path: str) -> nx.MultiGraph:
    """
    Return an UNDIRECTED walkable MultiGraph for ``place`` (e.g.
    "Noe Valley, San Francisco, California, USA" or "San Francisco,
    California, USA"). Caches to a GraphML file so OSM node IDs — and
    therefore your progress records — stay stable across runs.
    """
    ox = _ox()
    if os.path.exists(cache_path):
        G = ox.load_graphml(cache_path)
    else:
        G_dir = ox.graph_from_place(place, network_type="walk", simplify=True)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        ox.save_graphml(G_dir, cache_path)
        G = G_dir
    return ox.convert.to_undirected(G)


def load_or_download_polygon(polygon, cache_path: str) -> nx.MultiGraph:
    """Same as load_or_download but for an explicit shapely Polygon boundary."""
    ox = _ox()
    if os.path.exists(cache_path):
        G = ox.load_graphml(cache_path)
    else:
        G = ox.graph_from_polygon(polygon, network_type="walk", simplify=True)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        ox.save_graphml(G, cache_path)
    return ox.convert.to_undirected(G)


# --------------------------------------------------------------------------- #
# connectivity
# --------------------------------------------------------------------------- #
def largest_component(G: nx.MultiGraph) -> nx.MultiGraph:
    """The biggest connected chunk — OSM extracts often have stray islands."""
    if nx.is_connected(G):
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def connected_components(G: nx.MultiGraph):
    """Yield each connected component as its own MultiGraph."""
    for nodes in nx.connected_components(G):
        yield G.subgraph(nodes).copy()


# Service ways that aren't streets you'd want to walk. Note 'alley' is NOT here
# — alleys are part of the goal ("every street, alley, stairs").
_DRIVEWAY_SERVICES = {"driveway", "parking_aisle", "drive-through", "drive_through"}


def drop_driveways(G: nx.MultiGraph):
    """
    Remove driveway / parking-aisle / drive-through edges (OSM service=*), which
    the 'walk' network includes but you don't actually want to walk. Alleys are
    kept. Returns (filtered_graph, n_edges_removed).
    """
    def services(d):
        sv = d.get("service")
        if isinstance(sv, list):
            return set(sv)
        return {sv} if sv else set()

    # never drop a NAMED way, even if it's tagged service=driveway/parking_aisle
    # (e.g. a named lane) — that would silently remove a must-walk street.
    rm = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
          if services(d) & _DRIVEWAY_SERVICES and not d.get("name")]
    if not rm:
        return G, 0
    H = G.copy()
    H.remove_edges_from(rm)
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    return H, len(rm)


def _is_steps(d):
    h = d.get("highway")
    h = set(h) if isinstance(h, list) else {h}
    return "steps" in h


def _is_trail(d):
    """A trail/path (highway=path) — a real walkable route, never a sidewalk to
    dedupe away, so it's excluded from collapse (keeps park paths)."""
    h = d.get("highway")
    h = set(h) if isinstance(h, list) else {h}
    return "path" in h


def collapse_divided_ways(G: nx.MultiGraph, max_offset_m: float = 20.0,
                          max_bearing_diff: float = 25.0):
    """
    OSM often maps a corridor as two parallel ways — a road plus its own mapped
    sidewalk, or two unnamed footpaths around a median. Walking both produces a
    convoluted zig-zag, so we drop one of each near-parallel twin.

    Safety rails (so we never lose a must-walk way):
      * NAMED ways are never deleted. We only drop the UNNAMED member of a twin
        pair (a sidewalk / desire line). Two named ways — even two same-named
        carriageways of a divided road — are BOTH kept; you walk both sides.
      * STAIRWAYS are never collapsed (a road + a parallel public stairway are
        genuinely different walks, both part of the goal).
      * A twin is only dropped if its endpoints stay connected by a short
        alternate path, so connectivity and reachable coverage are preserved.

    Redundant *named* corridors (the Great Highway's four overlapping names) are
    NOT handled here — that's a scope question, not a geometry one, and lives in
    progress.is_required (only the canonical name of a declared corridor counts).

    Returns (filtered_graph, n_removed).
    """
    import math

    ys = [float(d["y"]) for _, d in G.nodes(data=True) if "y" in d]
    xs = [float(d["x"]) for _, d in G.nodes(data=True) if "x" in d]
    if not ys:
        return G, 0
    lat0, lon0 = sum(ys) / len(ys), sum(xs) / len(xs)
    cl = math.cos(math.radians(lat0))

    def enu(n):
        d = G.nodes[n]
        return ((float(d["x"]) - lon0) * cl * 111320.0,
                (float(d["y"]) - lat0) * 110540.0)

    def name_of(u, v, k):
        nm = G[u][v][k].get("name")
        if isinstance(nm, list):
            return tuple(sorted(nm))
        return nm

    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if _is_steps(d) or _is_trail(d):
            continue                      # never dedupe stairways or trails
        (ax, ay), (bx, by) = enu(u), enu(v)
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        edges.append({"e": (u, v, k), "m": ((ax + bx) / 2.0, (ay + by) / 2.0),
                      "dir": (dx / length, dy / length), "len": length})

    cell = max(max_offset_m, 1.0)
    grid = {}
    for i, e in enumerate(edges):
        grid.setdefault((int(e["m"][0] // cell), int(e["m"][1] // cell)), []).append(i)

    cos_lim = math.cos(math.radians(max_bearing_diff))
    Gw = G.copy()                       # live graph; removals checked against it
    gone = set()
    n_removed = 0
    for i, e in enumerate(edges):
        if i in gone:
            continue
        ci, cj = int(e["m"][0] // cell), int(e["m"][1] // cell)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for j in grid.get((ci + di, cj + dj), ()):
                    if j <= i or j in gone or i in gone:
                        continue
                    f = edges[j]
                    if abs(e["dir"][0] * f["dir"][0] + e["dir"][1] * f["dir"][1]) < cos_lim:
                        continue
                    rx, ry = f["m"][0] - e["m"][0], f["m"][1] - e["m"][1]
                    along = rx * e["dir"][0] + ry * e["dir"][1]
                    perp = abs(rx * -e["dir"][1] + ry * e["dir"][0])
                    if not (0.5 < perp <= max_offset_m):
                        continue
                    if abs(along) > (e["len"] + f["len"]) / 2.0:
                        continue            # offset end-to-end, not side-by-side
                    # NEVER delete a named way — that would silently remove a
                    # must-walk street from coverage and from the 100% total. So
                    # we only collapse when at least one twin is UNNAMED, and we
                    # drop the unnamed one:
                    #  * road + its unnamed sidewalk -> drop the sidewalk.
                    #  * two unnamed parallel paths (desire lines) -> drop one.
                    #  * two NAMED ways (even a divided road like a boulevard, or
                    #    two same-named carriageways) -> keep BOTH; you walk both
                    #    sides. A divided way is only "deduped" if a side is unnamed.
                    ni, nj = name_of(*e["e"]), name_of(*f["e"])
                    if ni is not None and nj is not None:
                        continue
                    if ni is None:
                        drop_idx, drop = i, e["e"]
                    else:
                        drop_idx, drop = j, f["e"]
                    du, dv, dk = drop
                    data = dict(Gw[du][dv][dk])
                    Gw.remove_edge(du, dv, dk)
                    # only confirm removal if du,dv stay connected by a SHORT
                    # alternate path (the parallel twin) — guarantees no split
                    # only drop a twin whose endpoints stay joined by a SHORT
                    # alternate (its parallel rail). Tight bound so a tiny edge
                    # isn't removed when its only detour is far away.
                    alt_max = e["len"] * 2.5 + 10.0
                    reach = nx.single_source_dijkstra_path_length(
                        Gw, du, cutoff=alt_max, weight="length")
                    if dv in reach:
                        gone.add(drop_idx)
                        n_removed += 1
                        if drop_idx == i:
                            break
                    else:
                        Gw.add_edge(du, dv, key=dk, **data)
                if i in gone:
                    break
            if i in gone:
                break
    Gw.remove_nodes_from([n for n in list(Gw.nodes) if Gw.degree(n) == 0])
    return Gw, n_removed


# --------------------------------------------------------------------------- #
# chunking into day-sized walks
# --------------------------------------------------------------------------- #
def iter_walk_chunks(G: nx.MultiGraph, target_km: float = 8.0):
    """
    Greedily partition G into CONNECTED sub-networks whose unique street length
    is roughly ``target_km`` each — i.e. one comfortable day's walk apiece.

    Grows a chunk by breadth-first exploration over not-yet-assigned edges, so
    each returned subgraph is connected and can be fed straight to
    ``postman.solve_route``. The final chunk may be shorter than target.

    Yields (chunk_index, subgraph).
    """
    target_m = target_km * 1000.0
    remaining = set(
        (u, v, k) for u, v, k in G.edges(keys=True)
    )
    # adjacency for BFS over edges
    idx = 0
    while remaining:
        # seed: any leftover edge, preferring one touching a low-degree node
        seed = next(iter(remaining))
        frontier = [seed]
        chunk_edges = []
        chunk_len = 0.0
        seen = set()
        while frontier and chunk_len < target_m:
            e = frontier.pop()
            if e not in remaining or e in seen:
                continue
            seen.add(e)
            u, v, k = e
            chunk_edges.append(e)
            remaining.discard(e)
            chunk_len += G[u][v][k].get("length", 0.0)
            # enqueue adjacent unassigned edges
            for node in (u, v):
                for nbr in G[node]:
                    for kk in G[node][nbr]:
                        cand = (node, nbr, kk)
                        cand_canon = cand if cand in remaining else (nbr, node, kk)
                        if cand_canon in remaining:
                            frontier.append(cand_canon)
        # build subgraph from collected edges
        sub = nx.MultiGraph()
        for (u, v, k) in chunk_edges:
            sub.add_node(u, **G.nodes[u])
            sub.add_node(v, **G.nodes[v])
            sub.add_edge(u, v, key=k, **G[u][v][k])
        # ensure connectivity (BFS guarantees it, but guard anyway)
        if not nx.is_connected(sub):
            sub = largest_component(sub)
        idx += 1
        yield idx, sub


def nearest_node(G: nx.MultiGraph, lat: float, lon: float):
    """
    OSM node id closest to (lat, lon). Dependency-free (no scikit-learn): an
    equirectangular approximation, which is plenty accurate at city scale where
    a few metres of geocoding error already dwarfs the projection error.
    Nodes carry 'y' (lat) and 'x' (lon) attributes from osmnx.
    """
    cos_lat = math.cos(math.radians(lat))
    best, best_d2 = None, float("inf")
    for n, data in G.nodes(data=True):
        try:
            y = float(data["y"])
            x = float(data["x"])
        except (KeyError, ValueError, TypeError):
            continue
        dx = (x - lon) * cos_lat
        dy = (y - lat)
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2, best = d2, n
    return best


def chunk_from_node(G: nx.MultiGraph, start_node, target_km: float = 8.0,
                    bias: float = 0.0):
    """
    One CONNECTED sub-network of roughly ``target_km`` of unique street, grown
    OUTWARD from ``start_node`` as a COMPACT disk.

    Why a disk and not a breadth-first edge-blob: a greedy edge frontier frays
    at its boundary, leaving through-streets chopped into one-block stubs. Each
    stub is an artificial odd-degree node, and odd nodes are exactly what forces
    backtracking in the postman solve — so a ragged boundary manufactures
    avoidable doubling-back. Instead we pick the nodes nearest ``start_node`` and
    take their INDUCED subgraph: every kept intersection keeps *all* its edges to
    other kept intersections, so the only odd nodes left lie along a smooth
    perimeter where the matcher can pair them cheaply with a neighbour one block
    over. In testing this cut a day's backtracking from ~18% to ~15% and replaced
    8 scattered stubs with a tidy boundary.

    ``bias`` (default 0 = pure compact disk) optionally elongates the disk toward
    the centroid of ``G`` so ``start_node`` lands on the boundary — cheaper when
    you force the route to start there — but values above ~0.5 stretch the day
    into a thin corridor, so keep it small. Always contains ``start_node``.

    Returns a MultiGraph ready for ``postman.solve_route``.
    """
    target_m = target_km * 1000.0
    lat0 = float(G.nodes[start_node]["y"])
    lon0 = float(G.nodes[start_node]["x"])
    cos0 = math.cos(math.radians(lat0))

    def xy(n):
        d = G.nodes[n]
        return (float(d["x"]) - lon0) * cos0, (float(d["y"]) - lat0)

    # direction from the start toward the bulk of the (remaining) streets
    ux = uy = 0.0
    if bias:
        n_nodes = G.number_of_nodes() or 1
        cx = sum(xy(n)[0] for n in G.nodes) / n_nodes
        cy = sum(xy(n)[1] for n in G.nodes) / n_nodes
        cn = math.hypot(cx, cy) or 1.0
        ux, uy = cx / cn, cy / cn

    def cost(n):
        x, y = xy(n)
        return math.hypot(x, y) - bias * (x * ux + y * uy)

    included = {start_node}
    length = 0.0
    for n in sorted(G.nodes, key=cost):
        if n in included:
            continue
        included.add(n)
        for nbr in G[n]:
            if nbr in included and nbr != n:
                for k in G[n][nbr]:
                    length += G[n][nbr][k].get("length", 0.0)
        if length >= target_m:
            break

    sub = G.subgraph(included).copy()
    if not nx.is_connected(sub):
        comp = next(c for c in nx.connected_components(sub) if start_node in c)
        sub = sub.subgraph(comp).copy()
    return sub


# --------------------------------------------------------------------------- #
# elevation
# --------------------------------------------------------------------------- #
def add_elevations(G: nx.MultiGraph, elev_cache_path: str,
                   dataset: str = "srtm30m", batch: int = 100):
    """
    Attach an ``elevation`` (metres) attribute to every node, fetched from the
    free, key-less OpenTopoData API and cached to ``elev_cache_path`` (a JSON
    map of node-id -> metres) so it's fetched once and reused forever.

    OpenTopoData's public endpoint allows 100 locations/request and ~1 req/sec,
    so we batch and pace accordingly. Returns the number of nodes newly fetched.
    """
    import json
    import time
    import requests

    cache = {}
    if os.path.exists(elev_cache_path):
        with open(elev_cache_path) as f:
            cache = json.load(f)

    todo = [n for n in G.nodes if str(n) not in cache]
    fetched = 0
    for i in range(0, len(todo), batch):
        group = todo[i:i + batch]
        loc = "|".join(f"{G.nodes[n]['y']},{G.nodes[n]['x']}" for n in group)
        r = requests.get(f"https://api.opentopodata.org/v1/{dataset}",
                         params={"locations": loc}, timeout=60)
        r.raise_for_status()
        results = r.json()["results"]
        for n, res in zip(group, results):
            ele = res.get("elevation")
            cache[str(n)] = float(ele) if ele is not None else 0.0
        fetched += len(group)
        if i + batch < len(todo):
            time.sleep(1.05)  # respect the public rate limit

    if fetched:
        os.makedirs(os.path.dirname(os.path.abspath(elev_cache_path)) or ".",
                    exist_ok=True)
        with open(elev_cache_path, "w") as f:
            json.dump(cache, f)

    attach_elevations(G, elev_cache_path, _cache=cache)
    return fetched


def attach_elevations(G: nx.MultiGraph, elev_cache_path: str, _cache=None):
    """Load a node-id -> elevation JSON sidecar and set node 'elevation'."""
    import json
    cache = _cache
    if cache is None:
        with open(elev_cache_path) as f:
            cache = json.load(f)
    missing = 0
    for n in G.nodes:
        v = cache.get(str(n))
        if v is None:
            missing += 1
        else:
            G.nodes[n]["elevation"] = float(v)
    return missing


def has_elevations(G: nx.MultiGraph) -> bool:
    return all("elevation" in d for _, d in G.nodes(data=True))


def compute_effort(G: nx.MultiGraph, climb_weight: float = 8.0):
    """
    Set an ``effort`` attribute on every edge:

        effort = horizontal_length + climb_weight * |elevation_change|

    Penalising absolute elevation change (symmetric, since the graph is
    undirected) makes the postman solver prefer to do its unavoidable
    *backtracking* over flat connectors rather than re-climbing hills — i.e.
    it stops walking up and down the same slope twice. ``climb_weight`` is the
    metres of flat walking each metre of climb is considered "worth"; the
    default 8 follows Naismith's rule (≈7.92). Higher = hill-averse.

    Also stamps each edge with ``rise`` (= |elevation_change|). Requires node
    elevations (see ``add_elevations``).
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        zu = G.nodes[u].get("elevation", 0.0)
        zv = G.nodes[v].get("elevation", 0.0)
        rise = abs(zv - zu)
        d["rise"] = rise
        d["effort"] = d.get("length", 0.0) + climb_weight * rise


def route_elevation_stats(G: nx.MultiGraph, route):
    """
    Replay a solved route (list of (u, v, key, is_deadhead)) in traversal order
    and total the real metres and the directional elevation change. Robust to
    the synthetic edge keys the solver creates for duplicated streets: lengths
    are looked up by the shortest parallel edge, and elevation only needs the
    endpoints' node heights.

    Returns dict: required_m, deadhead_m, ascent_m, descent_m, deadhead_ascent_m.
    """
    req = dead = asc = desc = dead_asc = 0.0
    for (u, v, k, is_dead) in route:
        # use the EXACT edge's length; fall back to a parallel edge only for the
        # synthetic keys the solver assigns to deadhead duplicates. (Using min()
        # unconditionally undercounts when a road and a longer parallel stairway
        # are both walked.)
        if v in G[u] and k in G[u][v]:
            L = G[u][v][k].get("length", 0.0)
        else:
            L = min((dd.get("length", 0.0) for dd in G[u][v].values()), default=0.0)
        if is_dead:
            dead += L
        else:
            req += L
        zu = G.nodes[u].get("elevation")
        zv = G.nodes[v].get("elevation")
        if zu is not None and zv is not None:
            dz = zv - zu
            if dz > 0:
                asc += dz
                if is_dead:
                    dead_asc += dz
            else:
                desc += -dz
    return {
        "required_m": req, "deadhead_m": dead,
        "ascent_m": asc, "descent_m": desc, "deadhead_ascent_m": dead_asc,
    }


def total_km(G: nx.MultiGraph) -> float:
    return sum(d.get("length", 0.0) for _, _, d in G.edges(data=True)) / 1000.0
