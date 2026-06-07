#!/usr/bin/env python3
"""
amblecli.py — Amble command line.

Typical multi-year workflow:

  # 1. Pull one neighborhood's walkable network (streets, alleys, stairs)
  python amblecli.py fetch "Bernal Heights, San Francisco, California, USA" \
        --cache data/bernal.graphml

  # 2. See how big the job is and how far you've gotten
  python amblecli.py status --cache data/bernal.graphml --store data/progress.json

  # 3. Plan the next ~8 km walk over streets you haven't done yet
  python amblecli.py plan  --cache data/bernal.graphml --store data/progress.json \
        --target-km 8 --out walks/walk_001

  # ...go walk it, following walks/walk_001.gpx on your phone...

  # 4. Record it as done
  python amblecli.py done  --cache data/bernal.graphml --store data/progress.json \
        --route walks/walk_001.route.json

  # 5. Export a progress map any time
  python amblecli.py map   --cache data/bernal.graphml --store data/progress.json \
        --out walks/progress_map.geojson
"""
import argparse
import json
import os
import sys

from amble import network as net
from amble import postman
from amble import progress as prog
from amble import export as exp
from amble import straightline
from amble import contour
from amble import coverage
from amble.postman import _bearing


def _fmt(stats):
    return (f"{stats['walked_km']:.1f} / {stats['total_km']:.1f} km "
            f"({stats['pct_done']:.1f}%)  |  "
            f"{stats['walked_edges']}/{stats['total_edges']} segments")


def _warn_store_mismatch(G, store):
    """Catch the silent-0% trap: a progress store paired with the wrong cache, or
    a graph re-fetched so its node IDs changed. Warn instead of showing a bogus 0%."""
    walked = prog.walked_id_set(store)
    if not walked:
        return
    graph_ids = {prog.edge_id(G, u, v, k) for u, v, k in G.edges(keys=True)}
    match = len(walked & graph_ids)
    if match < 0.5 * len(walked):
        print(f"  [warning] only {match}/{len(walked)} recorded edges match this "
              f"cache. The store and .graphml may not be a pair, or the graph was "
              f"re-fetched (edge IDs are only stable for the same .graphml file).",
              file=sys.stderr)


def _straight_runs(G, route):
    """Longest turn-free runs in a route (the 'swirls' — ring roads on a hill),
    as (length_m, street-names) sorted longest first."""
    runs, cur = [], []
    for prev, e in zip(route, route[1:]):
        b1, b2 = _bearing(G, prev[0], prev[1]), _bearing(G, e[0], e[1])
        cur.append(prev)
        if b1 is None or b2 is None or abs(((b2 - b1 + 180) % 360) - 180) > 30:
            runs.append(cur); cur = []
    if route:
        cur.append(route[-1]); runs.append(cur)
    out = []
    for run in runs:
        L = sum(min((d.get("length", 0.0) for d in G[u][v].values()), default=0.0)
                for (u, v, k, _d) in run)
        names = set()
        for (u, v, k, _d) in run:
            nm = G[u][v].get(k, next(iter(G[u][v].values()), {})).get("name")
            if nm:
                names.add(nm[0] if isinstance(nm, list) else nm)
        if names:
            out.append((L, ", ".join(sorted(names))[:48]))
    return sorted(out, key=lambda r: -r[0])


def _describe_node(G, n):
    """Human-readable '(lat, lon) — Street A / Street B' for a route endpoint."""
    if n is None or n not in G:
        return "unknown"
    data = G.nodes[n]
    lat, lon = data.get("y"), data.get("x")
    names = set()
    for nbr in G[n]:
        for k in G[n][nbr]:
            nm = G[n][nbr][k].get("name")
            if isinstance(nm, list):
                names.update(nm)
            elif nm:
                names.add(nm)
    where = " / ".join(sorted(names)) if names else "unnamed path"
    gmap = f"https://www.google.com/maps?q={lat},{lon}" if lat is not None else ""
    return f"({lat:.5f}, {lon:.5f}) — {where}\n      {gmap}"


def cmd_fetch(a):
    G = net.load_or_download(a.place, a.cache)
    G = net.largest_component(G)
    print(f"Fetched '{a.place}': {G.number_of_nodes()} intersections, "
          f"{G.number_of_edges()} segments, {net.total_km(G):.1f} km of walking.")
    print(f"Cached to {a.cache}")


def cmd_status(a):
    G = net.largest_component(net.load_or_download(a.place or "", a.cache)
                              if a.place else _load_cached(a.cache))
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    print(_fmt(prog.stats(G, store)))


def cmd_remaining(a):
    """Punch-list: the named ways you still need to walk, longest first."""
    from collections import defaultdict
    G = net.largest_component(_load_cached(a.cache))
    walked = prog.walked_id_set(prog.load_store(a.store))
    rem = defaultdict(float)
    for u, v, k, d in G.edges(keys=True, data=True):
        nm = d.get("name")
        if not nm or prog.edge_id(G, u, v, k) in walked:
            continue
        nm = nm[0] if isinstance(nm, list) else nm
        rem[nm] += min((dd.get("length", 0.0) for dd in G[u][v].values()), default=0.0)
    if not rem:
        print("Nothing remaining — every named way is walked.")
        return
    print(f"{len(rem)} named ways remaining ({sum(rem.values())/1000:.1f} km):")
    for nm, L in sorted(rem.items(), key=lambda x: -x[1]):
        print(f"  {L:6.0f} m  {nm}")


def cmd_migrate(a):
    """Carry a progress store across a RE-FETCH: re-key it from the old cache's
    node ids to the new cache's, matching each walked edge by name + nearest
    midpoint (prog.rekey_store/_best_match). Run once after re-fetching."""
    Gold = net.largest_component(_load_cached(a.from_cache))
    Gnew = net.largest_component(_load_cached(a.to_cache))
    store, migrated, lost, merged = prog.rekey_store(
        prog.load_store(a.store), Gold, Gnew)
    prog.save_store(store, a.store)
    print(f"Migrated {migrated} records to the new cache "
          f"({lost} not found in the new graph, {merged} merged into a shared edge). "
          f"Now at {_fmt(prog.stats(Gnew, store))}")


def cmd_total(a):
    """City-wide progress across neighborhoods. Boundary streets (same street in
    two fetches, slightly different coords) are deduped by tolerant geometry —
    name + nearest midpoint — so they're counted once."""
    from collections import defaultdict
    reps = defaultdict(list)            # name -> [ [midpoint, length, walked], ... ]
    for cache, store_path in a.area:
        G = net.largest_component(_load_cached(cache))
        walked = prog.walked_id_set(prog.load_store(store_path))
        for u, v, k, d in G.edges(keys=True, data=True):
            if not d.get("name"):
                continue
            name, mid = prog._edge_name(d), prog._midpoint(G, u, v)
            length = d.get("length", 0.0)
            w = prog.edge_id(G, u, v, k) in walked
            hit = next((r for r in reps[name]
                        if prog._geo_dist_m(mid, r[0]) <= 40.0
                        and abs(r[1] - length) <= max(20.0, 0.35 * length)), None)
            if hit:
                hit[2] = hit[2] or w     # the same street counted once; OR walked
            else:
                reps[name].append([mid, length, w])
    segs = [r for rs in reps.values() for r in rs]
    total_km = sum(r[1] for r in segs) / 1000.0
    walked_km = sum(r[1] for r in segs if r[2]) / 1000.0
    pct = (walked_km / total_km * 100.0) if total_km else 0.0
    print(f"City-wide: {walked_km:.1f} / {total_km:.1f} km ({pct:.1f}%)  |  "
          f"~{len(segs)} named segments across {len(a.area)} areas (boundary-deduped)")


def _load_cached(cache):
    import osmnx as ox
    G = ox.convert.to_undirected(ox.load_graphml(cache))
    # driveways / parking aisles aren't streets you'd walk; alleys & park paths
    # are kept. Done here so every command (plan/status/map/done) agrees on the
    # set of streets, so progress can actually reach 100%.
    G, _ = net.drop_driveways(G)
    # collapse divided ways (e.g. the two "Upper Great Hwy" carriageways, or a
    # path split around a median) so you don't walk both sides convolutedly.
    G, _ = net.collapse_divided_ways(G)
    return G


def cmd_plan(a):
    G = net.largest_component(_load_cached(a.cache))
    use_elev = bool(a.elev)
    if use_elev:
        if not os.path.exists(a.elev):
            raise SystemExit(
                f"No elevation data at {a.elev}. Run:  python amblecli.py elevation "
                f"--cache {a.cache} --elev {a.elev}")
        missing = net.attach_elevations(G, a.elev)
        if missing:
            print(f"  (note: {missing} nodes lack elevation; treated as flat)")
    store = prog.load_store(a.store)
    walked = prog.walked_id_set(store)
    _warn_store_mismatch(G, store)
    # the must-walk set is the NAMED ways; unnamed paths are connectors only
    Rreq = prog.remaining_subgraph(G, store, required_only=True)
    if Rreq.number_of_edges() == 0:
        print("Nothing left — every named way is walked. Congratulations.")
        return
    user_node = lat = lon = None
    if a.start:
        try:
            lat, lon = (float(x) for x in a.start.split(","))
        except ValueError:
            raise SystemExit('--start must be "lat,lon", e.g. --start "37.7531,-122.5050"')
        user_node = net.nearest_node(G, lat, lon)

    approach = []
    mode = a.style
    if a.style == "snake":
        sol = straightline.plan_boustrophedon(
            G, start=user_node, target_km=a.target_km,
            axis=a.axis, straight_km=a.straight_km, done=walked)
        # Snake covers one orientation's grid streets; it can't reach the curved
        # / other-axis / leftover named ways. Once it can only scrape up a little
        # (grid mostly done, or only non-grid streets remain near here), hand off
        # to Rural-Postman coverage so progress always advances to 100%.
        if sol.get("required_m", 0.0) < min(1000.0, 0.25 * a.target_km * 1000.0):
            print("(snake found little here — covering remaining named ways instead)")
            sol = coverage.plan_coverage(G, walked, start=user_node,
                                         target_km=a.target_km)
            mode = "efficient"            # report the mode that actually ran
    else:
        # Rural-Postman coverage of named ways over the FULL connected graph
        # (parks / hills) — named islands are stitched with connectors, never
        # stranded. Start at the top to work downhill.
        if a.start_at_top:
            if not use_elev:
                raise SystemExit("--start-at-top needs elevations (pass --elev)")
            force_start = contour.highest_node(Rreq) if Rreq.number_of_nodes() else None
        else:
            force_start = user_node
        sol = coverage.plan_coverage(G, walked, start=force_start, target_km=a.target_km)
        # optional stair-preferred climb from your start up to the top
        if a.start_at_top and user_node is not None and force_start is not None:
            path = contour.stair_preferred_ascent(G, user_node, force_start)
            if path and len(path) > 1:
                approach = [(u, v, k, True)
                            for (u, v, k) in contour.ascent_edges(G, path)]
                print(f"Approach: climb to the top ({_describe_node(G, force_start)})")

    route = approach + sol["route"]
    route_graph = G
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    gpx, npts = exp.route_to_gpx(route_graph, route, a.out + ".gpx")
    exp.route_to_geojson(route_graph, route, a.out + ".geojson")
    n_cues = len(exp.route_to_cues(route_graph, route))
    with open(a.out + ".route.json", "w") as f:
        json.dump([[u, v, k, d] for (u, v, k, d) in route], f)

    st = net.route_elevation_stats(route_graph, route)
    total = st["required_m"] + st["deadhead_m"]
    eff = (st["required_m"] / total) if total else 1.0
    jog = "connectors" if mode == "snake" else "backtracking"
    print(f"Planned walk -> {a.out}.gpx ({npts} track points, "
          f"{n_cues} turn-by-turn cues)")
    if mode == "snake":
        print(f"  mode           : snake — {sol.get('n_streets', 0)} whole "
              f"{a.axis.upper()} streets, end to end")
    print(f"  named ways     : {st['required_m']/1000:.2f} km")
    print(f"  {jog:<14} : {st['deadhead_m']/1000:.2f} km "
          f"({1-eff:.0%} of the walk)")
    print(f"  total on foot  : {total/1000:.2f} km")
    if use_elev:
        print(f"  elevation gain : {st['ascent_m']:.0f} m up / "
              f"{st['descent_m']:.0f} m down")
    if mode == "efficient":
        for L, names in _straight_runs(route_graph, route)[:3]:
            print(f"  long swirl     : {L:.0f} m  ({names})")
    s, e = sol["endpoints"]
    print(f"  START : {_describe_node(route_graph, route[0][0] if route else s)}")
    print(f"  END   : {_describe_node(route_graph, e)}")


def cmd_done(a):
    G = net.largest_component(_load_cached(a.cache))
    store = prog.load_store(a.store)
    with open(a.route) as f:
        route = [tuple(x) for x in json.load(f)]
    n = prog.mark_route_walked(store, G, route, when=a.date, note=a.note or "")
    prog.save_store(store, a.store)
    print(f"Marked {n} new segments walked. Now at {_fmt(prog.stats(G, store))}")


def cmd_elevation(a):
    G = net.largest_component(_load_cached(a.cache))
    print(f"Fetching elevations for {G.number_of_nodes()} nodes "
          f"(dataset={a.dataset}); cached to {a.elev} ...")
    fetched = net.add_elevations(G, a.elev, dataset=a.dataset)
    eles = [d["elevation"] for _, d in G.nodes(data=True) if "elevation" in d]
    lo, hi = (min(eles), max(eles)) if eles else (0, 0)
    print(f"Done. {fetched} newly fetched, {len(eles)} total. "
          f"Range {lo:.0f}–{hi:.0f} m.")


def cmd_map(a):
    G = net.largest_component(_load_cached(a.cache))
    store = prog.load_store(a.store)
    exp.progress_to_geojson(
        G, prog.walked_id_set(store), lambda u, v, k: prog.edge_id(G, u, v, k), a.out
    )
    print(f"Progress map -> {a.out}  ({_fmt(prog.stats(G, store))})")
    print("Open it at https://geojson.io to see walked vs remaining.")


def main():
    p = argparse.ArgumentParser(prog="amble")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="download a place's walkable network")
    f.add_argument("place")
    f.add_argument("--cache", required=True)
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("status", help="show progress")
    s.add_argument("--cache", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--place", default=None)
    s.set_defaults(func=cmd_status)

    pl = sub.add_parser("plan", help="plan the next walk over unwalked streets")
    pl.add_argument("--cache", required=True)
    pl.add_argument("--store", required=True)
    pl.add_argument("--out", required=True, help="output path prefix")
    pl.add_argument("--target-km", type=float, default=8.0)
    pl.add_argument("--style", choices=["snake", "efficient"], default="snake",
                    help="snake = walk whole streets of one orientation end to "
                         "end (easy to follow, the default); efficient = "
                         "minimal-distance coverage (best for parks/irregular areas)")
    pl.add_argument("--axis", choices=["ns", "ew"], default="ns",
                    help="snake mode: walk the N-S streets or the E-W streets "
                         "(do the two orientations on separate days)")
    pl.add_argument("--straight-km", type=float, default=1.4,
                    help="snake mode: length of each straight shot before the "
                         "one-block jog to the next parallel street")
    pl.add_argument("--start", default=None,
                    help='begin the walk at "lat,lon" (snake starts at the '
                         'nearest street corner; efficient anchors the chunk there)')
    pl.add_argument("--loop", action="store_true",
                    help="force start==end (default: open route, cheaper)")
    pl.add_argument("--start-at-top", action="store_true",
                    help="efficient mode on hills: start at the highest unwalked "
                         "point and work down (climbs up via named stairways). "
                         "Needs --elev.")
    pl.add_argument("--chunk-bias", type=float, default=0.0,
                    help="0 = compact disk (default); small positive values "
                         "elongate the day toward unwalked area so a forced "
                         "start is cheaper. Above ~0.5 makes thin corridors.")
    pl.add_argument("--elev", default=None,
                    help="elevation sidecar JSON (from `elevation`); enables "
                         "hill-aware routing that avoids re-climbing slopes")
    pl.add_argument("--climb-weight", type=float, default=8.0,
                    help="metres of flat walking each metre of climb is 'worth' "
                         "when minimising backtracking (default 8 ≈ Naismith)")
    pl.set_defaults(func=cmd_plan)

    el = sub.add_parser("elevation",
                        help="fetch node elevations (free, no API key) for a cache")
    el.add_argument("--cache", required=True)
    el.add_argument("--elev", required=True, help="sidecar JSON to write")
    el.add_argument("--dataset", default="srtm30m",
                    help="OpenTopoData dataset (srtm30m, aster30m, ...)")
    el.set_defaults(func=cmd_elevation)

    d = sub.add_parser("done", help="record a planned route as walked")
    d.add_argument("--cache", required=True)
    d.add_argument("--store", required=True)
    d.add_argument("--route", required=True, help="the .route.json from `plan`")
    d.add_argument("--date", default=None)
    d.add_argument("--note", default=None)
    d.set_defaults(func=cmd_done)

    m = sub.add_parser("map", help="export a walked-vs-remaining GeoJSON map")
    m.add_argument("--cache", required=True)
    m.add_argument("--store", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_map)

    rm = sub.add_parser("remaining", help="list the named ways you still need to walk")
    rm.add_argument("--cache", required=True)
    rm.add_argument("--store", required=True)
    rm.set_defaults(func=cmd_remaining)

    mg = sub.add_parser("migrate", help="carry a store across a re-fetch (old cache -> new cache)")
    mg.add_argument("--from-cache", required=True, dest="from_cache")
    mg.add_argument("--to-cache", required=True, dest="to_cache")
    mg.add_argument("--store", required=True)
    mg.set_defaults(func=cmd_migrate)

    tot = sub.add_parser("total", help="city-wide progress across neighborhoods")
    tot.add_argument("--area", nargs=2, action="append", required=True,
                     metavar=("CACHE", "STORE"),
                     help="a cache + store pair; repeat --area for each neighborhood")
    tot.set_defaults(func=cmd_total)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
