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
from amble import trace
from amble import render
from amble import export as exp
from amble import straightline
from amble import contour
from amble import coverage
from amble.postman import _bearing


def _fmt(stats):
    return (f"{stats['walked_km']:.1f} / {stats['total_km']:.1f} km "
            f"({stats['pct_done']:.1f}%)  |  "
            f"{stats['walked_edges']}/{stats['total_edges']} segments")


def _today():
    import datetime
    return datetime.date.today().isoformat()


def _fmt_date(iso):
    """ISO date -> a human label for the map, e.g. '2026-06-07' -> 'June 7, 2026'."""
    import datetime
    try:
        d = datetime.date.fromisoformat(iso)
    except (ValueError, TypeError):
        return iso or ""
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _slug(text):
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    return "".join(keep).strip("-").replace("--", "-") or "walk"


def _date_suffix(iso):
    """ISO date -> a filename tag, e.g. '2026-06-25' -> 'Jun 25, 2026'. The year
    matters: this is a multi-year project, so a same-named walk one year later
    must not collide with this one."""
    import datetime
    try:
        d = datetime.date.fromisoformat(iso)
    except (ValueError, TypeError):
        return ""
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _date_from_filename(path):
    """Best-effort ISO date from the dated filenames AMble creates."""
    import datetime
    import re
    # Accept the tool's own collision-suffixed names too: '... - Jun 25, 2026 (2).gpx'
    m = re.search(r"- ([A-Z][a-z]{2}) (\d{1,2}), (\d{4})(?:\s*\(\d+\))?(?:\.[^.]+)?$",
                  os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(" ".join(m.groups()), "%b %d %Y").date().isoformat()
    except ValueError:
        return None


def _dated_walk_path(path, note, iso_date):
    """Target path for an imported GPX: '<note> - <Mon D>.gpx', beside the source,
    so recorded tracks self-describe by date instead of piling up as ' copy'.
    Returns None if no date is available or the file is already named that way."""
    import re
    suffix = _date_suffix(iso_date)
    if not suffix:
        return None
    base = os.path.splitext(os.path.basename(note))[0]
    base = re.sub(r"\s*-\s*[A-Z][a-z]{2}\s+\d{1,2}(?:,?\s+\d{4})?(?:\s*\(\d+\))?$",
                  "", base).strip()  # avoid doubling, incl. ' (2)' suffixed names
    base = base.replace(os.sep, "-")
    ext = os.path.splitext(path)[1] or ".gpx"
    target = os.path.join(os.path.dirname(path), f"{base} - {suffix}{ext}")
    if os.path.abspath(target) == os.path.abspath(path):
        return None
    return target


def _rename_walk_file(path, note, iso_date):
    """Rename an imported GPX to its dated name; disambiguate on collision so a
    second walk that shares a name+date never clobbers the first. Returns the
    final path (the original, unchanged, on any error)."""
    target = _dated_walk_path(path, note, iso_date)
    if not target:
        return path
    if os.path.exists(target):
        stem, ext = os.path.splitext(target)
        i = 2
        while os.path.exists(f"{stem} ({i}){ext}"):
            i += 1
        target = f"{stem} ({i}){ext}"
    try:
        os.rename(path, target)
        return target
    except OSError:
        return path


def _redact_zones(a):
    """Privacy zones (lat, lon, radius_m) to hide from the map. Read from a
    gitignored privacy file (default maps/privacy.json) plus any --hide flags, so
    a home address never lives in committed code. File format:
        {"hide": [{"lat": .., "lon": .., "radius_m": 200, "label": "home"}]}
    """
    zones = []
    path = getattr(a, "privacy", None)
    if path and os.path.exists(path):
        with open(path) as f:
            for z in json.load(f).get("hide", []):
                zones.append((float(z["lat"]), float(z["lon"]),
                              float(z.get("radius_m", 200.0))))
    for h in getattr(a, "hide", None) or []:
        lat, lon = (float(x) for x in h.split(","))
        zones.append((lat, lon, float(getattr(a, "hide_radius", 200.0))))
    return zones


def _progress_update(G, store, date=None):
    """A short 'how far have I come' blurb: the given day's new distance (if any)
    plus the running total and city coverage. Printed after import/done."""
    summ = prog.walk_summary(G, store)
    st = prog.stats(G, store)
    lines = []
    today = next((d for d in summ["days"] if d["date"] == date), None)
    if today:
        lines.append(f"  {today['date']}: covered {today['km']:.2f} km of new "
                     f"streets ({today['named_km']:.2f} km named)")
    lines.append(f"  total: {summ['total_km']:.1f} km of streets over "
                 f"{summ['n_days']} day(s)  ·  city coverage "
                 f"{st['pct_done']:.1f}% ({st['walked_km']:.1f}/"
                 f"{st['total_km']:.0f} km named)")
    return "\n".join(lines)


def _warn_store_mismatch(G, store):
    """Catch the silent-0% trap: a progress store paired with the wrong cache, or
    a graph re-fetched so its node IDs changed. Warn instead of showing a bogus 0%."""
    expected = store.get("inventory_fingerprint")
    actual = G.graph.get("amble_inventory_fingerprint")
    if expected and actual and expected != actual:
        print("  [warning] progress store was built against a different canonical "
              "inventory; rebuild or migrate before trusting completeness.", file=sys.stderr)
    walked = prog.walked_id_set(store)
    if not walked:
        return
    graph_ids = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        graph_ids.add(prog.edge_id(G, u, v, k))
        graph_ids.add(prog.legacy_edge_id(u, v, k))
        aliases = d.get("coverage_alias_ids") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        graph_ids.update(str(x) for x in aliases)
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
    if a.polygon:
        from shapely.geometry import shape
        with open(a.polygon) as f:
            obj = json.load(f)
        if obj.get("type") == "FeatureCollection":
            if len(obj.get("features", [])) != 1:
                raise SystemExit("--polygon FeatureCollection must contain exactly one feature")
            obj = obj["features"][0]
        if obj.get("type") == "Feature":
            obj = obj["geometry"]
        G = net.load_or_download_polygon(shape(obj), a.cache)
        label = a.polygon
    else:
        if not a.place:
            raise SystemExit("fetch needs a PLACE or --polygon zone.geojson")
        G = net.load_or_download(a.place, a.cache)
        label = a.place
    G, audit = net.prepare_graph(G)
    print(f"Fetched '{label}': {G.number_of_nodes()} intersections, "
          f"{G.number_of_edges()} segments, {net.total_km(G):.1f} km of walking.")
    pa = audit["passages"]
    print(f"Canonical inventory: {pa['targets']} public named passages, "
          f"{pa['canonical_m']/1000:.1f} km ({pa['aliases']} duplicate surfaces merged; "
          f"{pa.get('observation_aliases', 0)} sidewalk observation aliases; "
          f"{pa.get('id_collisions_resolved', 0)} ID collisions resolved).")
    print(f"Cached to {a.cache}")


def cmd_status(a):
    if a.place:
        G, _ = net.prepare_graph(net.load_or_download(a.place or "", a.cache))
    else:
        G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    print(_fmt(prog.stats(G, store)))
    summ = prog.walk_summary(G, store)
    if summ["n_days"]:
        last = summ["days"][-1]
        print(f"  {summ['total_km']:.1f} km of streets covered over "
              f"{summ['n_days']} day(s); latest {last['date']} "
              f"(+{last['km']:.2f} km)")


def cmd_audit(a):
    """Audit the OSM snapshot against the canonical public-passage policy."""
    import collections
    import osmnx as ox
    raw = ox.convert.to_undirected(ox.load_graphml(a.cache))
    present = {key for *_e, d in raw.edges(data=True) for key in d}
    present |= {key for _n, d in raw.nodes(data=True) for key in d}
    G, audit = net.prepare_graph(raw)
    pa = audit["passages"]
    classes = collections.Counter(
        d.get("coverage_class") for *_e, d in G.edges(data=True)
        if prog.is_required(d))
    print(f"Canonical targets: {pa['targets']} passages, "
          f"{pa['canonical_m']/1000:.1f} km; {pa['aliases']} duplicate surfaces merged; "
          f"{pa.get('observation_aliases', 0)} sidewalk observation aliases; "
          f"{pa.get('id_collisions_resolved', 0)} identity collisions resolved")
    print("Classes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(classes.items(), key=lambda kv: str(kv[0]))))
    fr = pa.get("foot_restricted_targets", {})
    if fr.get("targets"):
        print(f"Foot-restricted targets (evidence via parallel sidepath): "
              f"{fr['targets']} passages, {fr['m']/1000:.1f} km")
    nm = pa.get("divided_near_miss", {})
    if nm.get("pairs"):
        print(f"[review] possible divided-road pairs left UNMERGED "
              f"(45-90 m apart): {nm['pairs']} pairs, {nm['m']/1000:.2f} km")
    sh = pa.get("intersection_shards", {})
    if sh.get("targets"):
        print(f"Intersection shards absorbed (sub-20 m same-name street "
              f"stubs): {sh['targets']} pieces, {sh['m']/1000:.2f} km")
    pr = pa.get("pier_area_rings_not_required", {})
    if pr.get("edges"):
        print(f"Pier polygon rings kept walkable but NOT required: "
              f"{pr['edges']} edges, {pr['m']/1000:.2f} km")
    # Targets you cannot reach on foot from the main network are a standing
    # decision, not an error — surface them so nothing hides in an islet.
    import networkx as nx
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(comps) > 1:
        rows = []
        for comp in comps[1:]:
            sub = G.subgraph(comp)
            km = sum(d.get("length", 0.0)
                     for *_e, d in sub.edges(data=True)
                     if prog.is_required(d)) / 1000
            if km >= 0.05:
                name = next((d.get("coverage_name") for *_e, d in
                             sub.edges(data=True)
                             if d.get("coverage_name")), "?")
            else:
                continue
            rows.append((km, len(comp), name))
        rows.sort(reverse=True)
        if rows:
            total = sum(r[0] for r in rows)
            print(f"Required distance OUTSIDE the main component: "
                  f"{total:.1f} km in {len(rows)} islands — top: " +
                  "; ".join(f"{n} ({km:.1f} km)" for km, _c, n in rows[:4]))
    for reason, item in sorted(audit["removed"].items()):
        print(f"Excluded {reason}: {item['edges']} edges, {item['m']/1000:.2f} km")
    critical = {"foot", "footway", "sidewalk", "amenity", "parking", "indoor",
                "building", "layer", "level", "man_made", "barrier", "railway",
                "public_transport", "area"}
    retained = set(filter(None,
        str(raw.graph.get("amble_retained_way_tags", "")).split("|")))
    retained |= set(filter(None,
        str(raw.graph.get("amble_retained_node_tags", "")).split("|")))
    # Old snapshots lack an explicit schema, so actual tag occurrence is the
    # only evidence available. New snapshots validate what the downloader
    # requested, even when no current OSM feature happens to use a given tag.
    missing = sorted(critical - (retained or present))
    if missing:
        print("[warning] cache predates canonical tag retention; re-fetch to classify: "
              + ", ".join(missing))
    if raw.graph.get("amble_osm_filter") != net.COVERAGE_FILTER_VERSION:
        print(f"[warning] cache predates the {net.COVERAGE_FILTER_VERSION} "
              "download filter; re-fetch — it is missing sidewalk-separate "
              "roadways, foot-restricted carriageways, piers, and "
              "walk-unreachable components (e.g. Treasure Island)")


def cmd_remaining(a):
    """Punch-list: the named ways you still need to walk, longest first."""
    from collections import defaultdict
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    walked = prog.completed_id_set(G, store)
    rem = defaultdict(float)
    for u, v, k, d in G.edges(keys=True, data=True):
        nm = d.get("name")
        if not prog.is_required(d) or prog.edge_id(G, u, v, k) in walked:
            continue
        nm = nm[0] if isinstance(nm, list) else nm
        rem[nm] += d.get("length", 0.0)
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
    Gold = _load_cached(a.from_cache)
    Gnew = _load_cached(a.to_cache)
    import shutil
    backup = a.store + ".pre-migrate.bak"
    if os.path.exists(a.store):
        shutil.copy2(a.store, backup)
    store, migrated, lost, merged = prog.rekey_store(
        prog.load_store(a.store), Gold, Gnew)
    # The store now describes the NEW cache's inventory: restamp, or every
    # later command would warn about a fingerprint built against the old one.
    store["model"] = Gnew.graph.get("amble_model", store.get("model"))
    store["inventory_fingerprint"] = Gnew.graph.get("amble_inventory_fingerprint")
    prog.save_store(store, a.store)
    print(f"Migrated {migrated} records to the new cache "
          f"({lost} not found in the new graph, {merged} merged into a shared edge). "
          f"Now at {_fmt(prog.stats(Gnew, store))}")
    if os.path.exists(backup):
        print(f"Original store backed up to {backup}; unmatched evidence was preserved.")


def cmd_total(a):
    """City-wide progress across neighborhoods. Boundary streets (same street in
    two fetches, slightly different coords) are deduped by tolerant geometry —
    name + nearest midpoint — so they're counted once."""
    from collections import defaultdict
    reps = defaultdict(list)            # name -> [ [midpoint, length, fraction], ... ]
    for cache, store_path in a.area:
        G = _load_cached(cache)
        store = prog.load_store(store_path)
        for u, v, k, d in G.edges(keys=True, data=True):
            if not prog.is_required(d):
                continue
            name, mid = prog._edge_name(d), prog._midpoint(G, u, v)
            length = d.get("length", 0.0)
            rec = prog._combined_record(prog._records_for_edge(G, u, v, k, store))
            frac = prog.coverage_frac(rec) if rec is not None else 0.0
            hit = next((r for r in reps[name]
                        if prog._geo_dist_m(mid, r[0]) <= 40.0
                        and abs(r[1] - length) <= max(20.0, 0.35 * length)), None)
            if hit:
                hit[2] = max(hit[2], frac)  # same boundary block: best evidence wins
            else:
                reps[name].append([mid, length, frac])
    segs = [r for rs in reps.values() for r in rs]
    total_km = sum(r[1] for r in segs) / 1000.0
    walked_km = sum(r[1] * r[2] for r in segs) / 1000.0
    pct = (walked_km / total_km * 100.0) if total_km else 0.0
    print(f"City-wide: {walked_km:.1f} / {total_km:.1f} km ({pct:.1f}%)  |  "
          f"~{len(segs)} named segments across {len(a.area)} areas (boundary-deduped)")


def _load_cached(cache):
    import osmnx as ox
    G = ox.convert.to_undirected(ox.load_graphml(cache))
    # Keep the detailed public pedestrian network for routing, while annotating
    # one canonical target per named public passage.  Sidewalks/carriageways are
    # representations of a block, not separate accomplishments.
    G, _audit = net.prepare_graph(G)
    return G


def _planning_component(G, store, start_node=None):
    """Choose one routable component without hiding targets from global status.

    The inventory may contain small disconnected public components.  Status and
    maps retain all of them; a single walk must use one connected component.  A
    user-specified start wins when that component has remaining work, otherwise
    the component with the most remaining canonical distance is selected.
    """
    done = prog.completed_id_set(G, store)
    components = list(net.connected_components(G))
    if len(components) == 1:
        return G

    def remaining_m(H):
        return sum(float(d.get("length", 0.0) or 0.0)
                   for u, v, k, d in H.edges(keys=True, data=True)
                   if prog.is_required(d) and prog.edge_id(H, u, v, k) not in done)

    if start_node is not None:
        containing = next((H for H in components if start_node in H), None)
        if containing is not None and remaining_m(containing) > 0:
            return containing
    return max(components, key=remaining_m)


def cmd_plan(a):
    G_all = _load_cached(a.cache)
    use_elev = bool(a.elev)
    if use_elev:
        if not os.path.exists(a.elev):
            raise SystemExit(
                f"No elevation data at {a.elev}. Run:  python amblecli.py elevation "
                f"--cache {a.cache} --elev {a.elev}")
        missing = net.attach_elevations(G_all, a.elev)
        if missing:
            print(f"  (note: {missing} nodes lack elevation; treated as flat)")
    store = prog.load_store(a.store)
    # Only completed canonical passages leave the planning target.  A partial
    # block must remain available until the same completion rule used by status
    # and maps considers it done.
    walked_all = prog.completed_id_set(G_all, store)
    _warn_store_mismatch(G_all, store)
    Rall = prog.remaining_subgraph(G_all, store, required_only=True)
    if Rall.number_of_edges() == 0:
        print("Nothing left — every named way is walked. Congratulations.")
        return
    user_node = lat = lon = None
    if a.start:
        try:
            lat, lon = (float(x) for x in a.start.split(","))
        except ValueError:
            raise SystemExit('--start must be "lat,lon", e.g. --start "37.7531,-122.5050"')
        user_node = net.nearest_node(G_all, lat, lon)

    G = _planning_component(G_all, store, user_node)
    if user_node is not None and user_node not in G:
        user_node = None
        print("  (note: the requested start is disconnected from remaining work; "
              "planning the largest remaining component)")
    walked = {eid for eid in walked_all}
    Rreq = prog.remaining_subgraph(G, store, required_only=True)
    default_start = user_node
    if default_start is None and Rreq.number_of_edges():
        default_start = next(iter(Rreq.edges()))[0]

    approach = []
    mode = a.style
    if a.style == "snake":
        sol = straightline.plan_boustrophedon(
            G, start=default_start, target_km=a.target_km,
            axis=a.axis, straight_km=a.straight_km, done=walked)
        # Snake covers one orientation's grid streets; it can't reach the curved
        # / other-axis / leftover named ways. Once it can only scrape up a little
        # (grid mostly done, or only non-grid streets remain near here), hand off
        # to Rural-Postman coverage so progress always advances to 100%.
        if sol.get("required_m", 0.0) < min(1000.0, 0.25 * a.target_km * 1000.0):
            print("(snake found little here — covering remaining named ways instead)")
            sol = coverage.plan_coverage(G, walked, start=default_start,
                                         target_km=a.target_km,
                                         open_route=not a.loop)
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
        sol = coverage.plan_coverage(G, walked, start=force_start,
                                     target_km=a.target_km,
                                     open_route=not a.loop)
        # optional stair-preferred climb from your start up to the top
        if a.start_at_top and user_node is not None and force_start is not None:
            path = contour.stair_preferred_ascent(G, user_node, force_start)
            if path and len(path) > 1:
                approach = [(u, v, k, True)
                            for (u, v, k) in contour.ascent_edges(G, path)]
                print(f"Approach: climb to the top ({_describe_node(G, force_start)})")

    route = approach + sol["route"]
    if a.loop and route and route[-1][1] != route[0][0]:
        import networkx as nx
        try:
            back = nx.shortest_path(G, route[-1][1], route[0][0], weight="length")
            for x, y in zip(back, back[1:]):
                k, _ = postman._min_parallel_edge(G, x, y, "length")
                route.append((x, y, k, True))
            sol["endpoints"] = (route[0][0], route[-1][1])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise SystemExit("Cannot close this route within the public routing component")
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
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    with open(a.route) as f:
        route = [tuple(x) for x in json.load(f)]
    invalid = [(u, v, k) for u, v, k, _d in route
               if u not in G or v not in G[u] or k not in G[u][v]]
    if invalid:
        raise SystemExit(f"Route does not match this cache ({len(invalid)} unknown edges); "
                         "no progress was changed")
    n = prog.mark_route_walked(store, G, route, when=a.date, note=a.note or "")
    if G.graph.get("amble_inventory_fingerprint"):
        store.setdefault("schema_version", 2)
        store.setdefault("model", G.graph.get("amble_model"))
        store.setdefault("inventory_fingerprint",
                         G.graph.get("amble_inventory_fingerprint"))
    prog.save_store(store, a.store)
    print(f"Marked {n} new segments walked. Now at {_fmt(prog.stats(G, store))}")
    print(_progress_update(G, store, a.date or _today()))


def cmd_import(a):
    """Map-match a RECORDED .gpx track log (from your phone) and mark it walked.
    Unlike `done`, which records a route `plan` made, this matches a noisy GPS
    trace onto the network — snapping each fix to the nearest street and joining
    consecutive fixes along the shortest path, skipping GPS dropouts."""
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    import hashlib
    with open(a.gpx, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    prior = store.get("imports", {}).get(digest)
    if prior and not a.force:
        print(f"Already imported this exact GPX on {prior.get('date', '?')} "
              f"as {prior.get('note', '(unlabelled)')!r}; no changes made.")
        return
    segments, veh = trace.strip_vehicle_runs(a.gpx)
    if veh["vehicle_m"]:
        print(f"  [vehicle] stripped {veh['vehicle_m']/1000:.2f} km of "
              f"sustained >13 km/h movement ({veh['n_vehicle_fixes']} fixes) — "
              "a recording left running on transit/in a car is not walking")
    if not segments:
        print(f"No track points found in {a.gpx} — nothing to import.")
        return
    note = a.note or trace.track_name(a.gpx) or os.path.basename(a.gpx)
    m = trace.match_trace_segments(G, segments)
    st0 = prog.stats(G, store)
    prog.record_spans(store, m["edge_spans"], when=a.date, note=note)
    st1 = prog.stats(G, store)
    store.setdefault("imports", {})[digest] = {
        "date": a.date or _today(), "note": note,
        "matcher": "canonical-hmm-v1", "source": os.path.basename(a.gpx),
        "raw_m": m.get("raw_m", 0.0), "assigned_m": m.get("matched_m", 0.0),
        "required_assigned_m": m.get("named_m", 0.0),
        "n_points": m.get("n_points", 0),
        "n_unmatched": m.get("n_unmatched", 0),
        "n_skipped": m.get("n_skipped", 0),
        "raw_gap_m": m.get("raw_gap_m", 0.0),
        "n_raw_gaps": m.get("n_raw_gaps", 0),
        "near_target_trace_m": m.get("near_target_trace_m", 0.0),
        "vehicle_m": veh["vehicle_m"], "n_vehicle_fixes": veh["n_vehicle_fixes"],
        "new_complete_m": max(0.0, (st1["complete_km"] - st0["complete_km"]) * 1000),
    }
    store["schema_version"] = 2
    store["model"] = "canonical-passages-v1"
    store["inventory_fingerprint"] = G.graph.get("amble_inventory_fingerprint")
    prog.save_store(store, a.store)
    print(f"Imported {a.gpx!r} as \"{note}\"")
    print(f"  {m['n_points']} GPS fixes -> {m['n_snapped']} on-street, "
          f"{m.get('n_unmatched', 0)} unmatched, "
          f"{m['n_skipped']} transitions skipped (GPS gaps)")
    print(f"  physical GPX : {m.get('raw_m', 0.0)/1000:.2f} km")
    if m.get("n_raw_gaps", 0):
        print(f"  ignored gaps : {m.get('raw_gap_m', 0.0)/1000:.2f} km across "
              f"{m['n_raw_gaps']} discontinuities (not presumed walked)")
    print(f"  assigned     : {m['matched_m']/1000:.2f} km of canonical passages "
          f"({m['named_m']/1000:.2f} km required) across "
          f"{len(m['edge_spans'])} blocks")
    print(f"  near targets : {m.get('near_target_trace_m', 0.0)/1000:.2f} km of "
          "physical trace stayed within 30 m of a canonical passage")
    print(f"  store: +{st1['covered_km']-st0['covered_km']:.2f} km named coverage, "
          f"+{st1['complete_edges']-st0['complete_edges']} blocks completed "
          f"({st1['partial_edges']} partially walked)")
    print(_progress_update(G, store, a.date or _today()))
    if a.map:
        out = os.path.join(a.maps_dir, f"{a.date or _today()}-{_slug(note)}.jpg")
        render.render_coverage(G, store, out, focus=False, mode="total",
                               redact=_redact_zones(a),
                               date_label=_fmt_date(a.date or _today()))
        print(f"  map -> {out}")
    new_path = _rename_walk_file(a.gpx, note, a.date or _today())
    if os.path.abspath(new_path) != os.path.abspath(a.gpx):
        print(f"  renamed -> {new_path}")


def cmd_rebuild(a):
    """Reprocess original GPXs into a new canonical, versioned progress store."""
    import hashlib
    if os.path.exists(a.store) and not a.force:
        raise SystemExit(f"Refusing to overwrite {a.store}; pass --force or choose a new path")
    G = _load_cached(a.cache)
    store = {"schema_version": 2, "model": "canonical-passages-v1",
             "inventory_fingerprint": G.graph.get("amble_inventory_fingerprint"),
             "walked": {}, "imports": {}}
    passage_audit = G.graph.get("amble_passage_audit", {})
    print(f"Canonical inventory: {prog.stats(G, store)['total_edges']} targets, "
          f"{prog.stats(G, store)['total_km']:.1f} km; "
          f"{passage_audit.get('observation_aliases', 0)} sidewalk aliases; "
          f"{passage_audit.get('id_collisions_resolved', 0)} ID collisions resolved")
    for path in a.gpx:
        segments, veh = trace.strip_vehicle_runs(path)
        if veh["vehicle_m"]:
            print(f"  [vehicle] {os.path.basename(path)}: stripped "
                  f"{veh['vehicle_m']/1000:.2f} km of >13 km/h movement "
                  f"({veh['n_vehicle_fixes']} fixes)")
        if not segments:
            print(f"  skip {path}: no track points")
            continue
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest in store["imports"]:
            print(f"  skip {path}: exact duplicate")
            continue
        note = trace.track_name(path) or os.path.splitext(os.path.basename(path))[0]
        date = _date_from_filename(path)
        if not date:
            date = _today()
            print(f"  [warning] {os.path.basename(path)}: no ' - Mon D, YYYY' "
                  f"in filename; diary date falls back to today ({date})")
        before = prog.stats(G, store)
        m = trace.match_trace_segments(G, segments)
        prog.record_spans(store, m["edge_spans"], when=date, note=note)
        store["imports"][digest] = {
            "date": date, "note": note, "matcher": "canonical-hmm-v1",
            "source": os.path.basename(path), "raw_m": m.get("raw_m", 0.0),
            "assigned_m": m.get("matched_m", 0.0),
            "required_assigned_m": m.get("named_m", 0.0),
            "n_points": m.get("n_points", 0),
            "n_unmatched": m.get("n_unmatched", 0),
            "n_skipped": m.get("n_skipped", 0),
            "raw_gap_m": m.get("raw_gap_m", 0.0),
            "n_raw_gaps": m.get("n_raw_gaps", 0),
            "near_target_trace_m": m.get("near_target_trace_m", 0.0),
            "vehicle_m": veh["vehicle_m"],
            "n_vehicle_fixes": veh["n_vehicle_fixes"],
        }
        after = prog.stats(G, store)
        store["imports"][digest]["new_complete_m"] = max(
            0.0, (after["complete_km"] - before["complete_km"]) * 1000)
        print(f"  {os.path.basename(path)}: GPX {m.get('raw_m',0)/1000:.2f} km, "
              f"near targets {m.get('near_target_trace_m',0)/1000:.2f} km, "
              f"assigned unique {m['named_m']/1000:.2f} km, "
              f"new complete +{after['complete_km']-before['complete_km']:.2f} km, "
              f"unmatched {m.get('n_unmatched',0)}/{m['n_points']}"
              + (f", ignored gaps {m['raw_gap_m']/1000:.2f} km"
                 if m.get("n_raw_gaps", 0) else ""))
    prog.save_store(store, a.store)
    print(f"Rebuilt -> {a.store}  ({_fmt(prog.stats(G, store))})")


def cmd_review(a):
    """Adversarially review persisted per-GPX matching diagnostics."""
    store = prog.load_store(a.store)
    imports = list(store.get("imports", {}).values())
    if not imports:
        print("No versioned GPX import diagnostics in this store; rebuild from original GPXs.")
        return
    imports.sort(key=lambda x: (x.get("date", ""), x.get("source", "")))
    total_raw = sum(float(x.get("raw_m", 0.0)) for x in imports)
    total_assigned = sum(float(x.get("required_assigned_m",
                                       x.get("assigned_m", 0.0))) for x in imports)
    total_new = sum(float(x.get("new_complete_m", 0.0)) for x in imports)
    print(f"GPX review: {len(imports)} files; {total_raw/1000:.2f} km contiguous "
          f"physical, {total_assigned/1000:.2f} km per-walk unique assigned, "
          f"{total_new/1000:.2f} km newly completed")
    for item in imports:
        raw = float(item.get("raw_m", 0.0))
        assigned = float(item.get("required_assigned_m",
                                  item.get("assigned_m", 0.0)))
        near = float(item.get("near_target_trace_m", 0.0))
        points = int(item.get("n_points", 0) or 0)
        unmatched = int(item.get("n_unmatched", 0) or 0)
        flags = []
        if float(item.get("vehicle_m", 0) or 0) > 0:
            flags.append(f"vehicle {float(item.get('vehicle_m', 0))/1000:.2f}km stripped")
        if item.get("n_raw_gaps", 0):
            flags.append(f"GPS gaps {float(item.get('raw_gap_m', 0))/1000:.2f}km")
        if raw and assigned > raw * 1.05:
            flags.append("ASSIGNED>PHYSICAL")
        if points and unmatched / points >= 0.20:
            flags.append(f"unmatched {unmatched/points:.0%}")
        if raw and near / raw < 0.60:
            flags.append(f"only {near/raw:.0%} near inventory")
        if near and assigned > near * 1.20:
            flags.append("assigned exceeds close trace: inspect HMM inference")
        if near >= 1000 and assigned / near < 0.65:
            flags.append(f"unique/near {assigned/near:.0%}: repeats or fragmented evidence")
        label = item.get("source") or item.get("note") or "unknown GPX"
        print(f"  {item.get('date', '?')}  {label}: physical {raw/1000:.2f} km, "
              f"near {near/1000:.2f}, assigned {assigned/1000:.2f}, "
              f"new complete {float(item.get('new_complete_m', 0))/1000:.2f}"
              + ("  [" + "; ".join(flags) + "]" if flags else ""))


def cmd_render(a):
    """Render the coverage map (whole city, walked total) to a JPEG on demand."""
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    date = a.date or _today()
    out = a.out or os.path.join(a.maps_dir, f"coverage_{date}.jpg")
    render.render_coverage(G, store, out, title=a.title, focus=a.focus,
                           mode=("by_walk" if a.by_walk else "total"),
                           redact=_redact_zones(a), date_label=_fmt_date(date))
    print(f"Coverage map -> {out}  ({_fmt(prog.stats(G, store))})")


def cmd_elevation(a):
    G = _load_cached(a.cache)
    print(f"Fetching elevations for {G.number_of_nodes()} nodes "
          f"(dataset={a.dataset}); cached to {a.elev} ...")
    fetched = net.add_elevations(G, a.elev, dataset=a.dataset)
    eles = [d["elevation"] for _, d in G.nodes(data=True) if "elevation" in d]
    lo, hi = (min(eles), max(eles)) if eles else (0, 0)
    print(f"Done. {fetched} newly fetched, {len(eles)} total. "
          f"Range {lo:.0f}–{hi:.0f} m.")


def cmd_map(a):
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    exp.progress_to_geojson(
        G, prog.completed_id_set(G, store), lambda u, v, k: prog.edge_id(G, u, v, k), a.out
    )
    print(f"Progress map -> {a.out}  ({_fmt(prog.stats(G, store))})")
    print("Open it at https://geojson.io to see walked vs remaining.")


def cmd_log(a):
    """A walk diary: per-day new street covered, with running totals."""
    G = _load_cached(a.cache)
    store = prog.load_store(a.store)
    _warn_store_mismatch(G, store)
    summ = prog.walk_summary(G, store)
    st = prog.stats(G, store)
    if not summ["days"]:
        print("No walks recorded yet.")
        return
    print(f"Walk log — {summ['total_km']:.1f} km of streets covered over "
          f"{summ['n_days']} day(s)\n")
    for d in summ["days"]:
        walks = ", ".join(sorted(d["notes"])) if d["notes"] else f"{d['edges']} segments"
        print(f"  {d['date']}   {d['km']:6.2f} km  ({d['named_km']:5.2f} km named)"
              f"   {walks}")
    print(f"\nCity coverage: {_fmt(st)}")


def main():
    p = argparse.ArgumentParser(prog="amble")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="download a place's walkable network")
    f.add_argument("place", nargs="?", help="OSM place name (or use --polygon)")
    f.add_argument("--polygon", help="GeoJSON Polygon/Feature defining the coverage zone")
    f.add_argument("--cache", required=True)
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("status", help="show progress")
    s.add_argument("--cache", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--place", default=None)
    s.set_defaults(func=cmd_status)

    au = sub.add_parser("audit", help="audit cache eligibility and canonical inventory")
    au.add_argument("--cache", required=True)
    au.set_defaults(func=cmd_audit)

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
    pl.add_argument("--elev", default=None,
                    help="elevation sidecar JSON (from `elevation`); adds a GPX "
                         "profile and enables --start-at-top")
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

    im = sub.add_parser("import",
                        help="map-match a recorded .gpx track log and mark it walked")
    im.add_argument("--cache", required=True)
    im.add_argument("--store", required=True)
    im.add_argument("--gpx", required=True, help="a recorded GPS track (.gpx)")
    im.add_argument("--date", default=None, help="walk date (default: today)")
    im.add_argument("--note", default=None,
                    help="default: the GPX track name, else the file name")
    im.add_argument("--force", action="store_true",
                    help="reprocess an exact GPX already recorded in this store")
    im.add_argument("--map", action="store_true",
                    help="also render a whole-city coverage JPEG into --maps-dir")
    im.add_argument("--maps-dir", default="maps", dest="maps_dir",
                    help="folder for rendered maps (default: maps/)")
    im.add_argument("--privacy", default="maps/privacy.json",
                    help="gitignored JSON of zones to hide from the map")
    im.add_argument("--hide", action="append", metavar="LAT,LON",
                    help="also hide a zone at this point (repeatable)")
    im.add_argument("--hide-radius", type=float, default=200.0, dest="hide_radius",
                    help="radius in metres for --hide zones (default 200)")
    im.set_defaults(func=cmd_import)

    rb = sub.add_parser("rebuild",
                        help="reprocess original GPXs into a new canonical store")
    rb.add_argument("--cache", required=True)
    rb.add_argument("--store", required=True, help="new progress JSON to create")
    rb.add_argument("--gpx", nargs="+", required=True, help="GPX files in walk order")
    rb.add_argument("--force", action="store_true", help="overwrite --store")
    rb.set_defaults(func=cmd_rebuild)

    rv = sub.add_parser("review",
                        help="flag suspicious distance/matching results in a rebuilt store")
    rv.add_argument("--store", required=True)
    rv.set_defaults(func=cmd_review)

    m = sub.add_parser("map", help="export a walked-vs-remaining GeoJSON map")
    m.add_argument("--cache", required=True)
    m.add_argument("--store", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_map)

    rn = sub.add_parser("render", help="render a coverage map to a JPEG (matplotlib)")
    rn.add_argument("--cache", required=True)
    rn.add_argument("--store", required=True)
    rn.add_argument("--out", default=None,
                    help="output path (default: maps/coverage_<date>.jpg)")
    rn.add_argument("--maps-dir", default="maps", dest="maps_dir",
                    help="folder for the dated default filename (default: maps/)")
    rn.add_argument("--title", default="",
                    help="optional header text (default: none)")
    rn.add_argument("--date", default=None,
                    help="date stamped on the map + in the filename (ISO yyyy-mm-dd)")
    rn.add_argument("--focus", action="store_true",
                    help="crop to the walked area (default: show the whole city)")
    rn.add_argument("--by-walk", action="store_true", dest="by_walk",
                    help="colour each walk separately (default: one walked total)")
    rn.add_argument("--privacy", default="maps/privacy.json",
                    help="gitignored JSON of zones to hide from the map")
    rn.add_argument("--hide", action="append", metavar="LAT,LON",
                    help="also hide a zone at this point (repeatable)")
    rn.add_argument("--hide-radius", type=float, default=200.0, dest="hide_radius",
                    help="radius in metres for --hide zones (default 200)")
    rn.set_defaults(func=cmd_render)

    lg = sub.add_parser("log", help="walk diary: per-day distance covered + totals")
    lg.add_argument("--cache", required=True)
    lg.add_argument("--store", required=True)
    lg.set_defaults(func=cmd_log)

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
