"""
export.py — turn a solved route into files you can actually use:

  * GPX  — load into a phone GPS app (OsmAnd, Gaia GPS, Organic Maps) and
           follow the track turn by turn.
  * GeoJSON — drop into https://geojson.io, QGIS, or a Leaflet map to see
              the route (deadhead segments shown separately) or your overall
              walked-vs-remaining progress.
"""

from __future__ import annotations

import json
import math
import xml.sax.saxutils as sx


def _first_name(data):
    nm = data.get("name")
    if isinstance(nm, list):
        return nm[0] if nm else ""
    return nm or ""


def _bearing_ll(lon1, lat1, lon2, lat2):
    y1, y2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(y2)
    y = math.cos(y1) * math.sin(y2) - math.sin(y1) * math.cos(y2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _maneuver(delta):
    """Map a signed heading change (deg, + = clockwise/right) to (phrase, sym)."""
    ad = abs(delta)
    if ad <= 20:
        return "Continue", "Straight"
    side = "right" if delta > 0 else "left"
    sym = "Right" if delta > 0 else "Left"
    if ad <= 50:
        return f"Slight {side}", sym
    if ad <= 135:
        return f"Turn {side}", sym
    if ad <= 170:
        return f"Sharp {side}", sym
    return "U-turn", "Straight"     # a reversal — no left/right arrow


def _cue_dist_m(a, b):
    dlat = (b[0] - a[0]) * 110540.0
    dlon = (b[1] - a[1]) * 111320.0 * math.cos(math.radians((a[0] + b[0]) / 2.0))
    return math.hypot(dlat, dlon)


def _ll_dist(a, b):                       # a, b = (lon, lat)
    dlat = (b[1] - a[1]) * 110540.0
    dlon = (b[0] - a[0]) * 111320.0 * math.cos(math.radians((a[1] + b[1]) / 2.0))
    return math.hypot(dlat, dlon)


def _span_bearing(coords, from_end, span_m=20.0):
    """
    Travel bearing over ~span_m of geometry at one end of an edge, rather than the
    final 2 points — OSM ways often hook into the intersection, so a 2-point
    bearing misreads a real turn as "Continue". from_end=True: approach bearing
    INTO coords[-1]; from_end=False: departure bearing OUT of coords[0].
    """
    seq = list(reversed(coords)) if from_end else coords
    node, far, acc = seq[0], seq[-1], 0.0
    for a, b in zip(seq, seq[1:]):
        acc += _ll_dist(a, b)
        far = b
        if acc >= span_m:
            break
    if node == far:
        return None
    return (_bearing_ll(far[0], far[1], node[0], node[1]) if from_end
            else _bearing_ll(node[0], node[1], far[0], far[1]))


def route_to_cues(G, route, min_gap_m=25.0):
    """
    Turn-by-turn cues from an ordered route, as a list of
    (lat, lon, instruction, street, sym).

    A cue is emitted at every real maneuver — turns onto NAMED streets get the
    name ("Turn left onto 45th Ave"); turns through UNNAMED park/connector
    geometry get a bare "Turn left" so you're still guided through a tangle.
    Direction reversals (dead-end out-and-backs) say "Turn around".

    To avoid a "cue storm" (several cues within 30 m where the route wiggles
    through micro-geometry), cues are then DEBOUNCED: a maneuver within
    ``min_gap_m`` of the previously-kept cue is dropped (Start and End always
    kept). This keeps well-separated turns — including a long unnamed run with
    several genuine turns — while collapsing close clusters to one cue.
    """
    if not route:
        return []
    start_name = next((_first_name(_edge_data(G, u, v, k))
                       for (u, v, k, _d) in route
                       if _first_name(_edge_data(G, u, v, k))), "")
    u0, v0, k0, _ = route[0]
    raw = [(G.nodes[u0]["y"], G.nodes[u0]["x"],
            f"Start on {start_name}" if start_name else "Start",
            start_name, "Start", 999.0)]
    for (u, v, k, _d), (u2, v2, k2, _d2) in zip(route, route[1:]):
        c_in = _edge_coords(G, u, v, k)
        c_out = _edge_coords(G, u2, v2, k2)
        if len(c_in) < 2 or len(c_out) < 2:
            continue
        b_in = _span_bearing(c_in, from_end=True)
        b_out = _span_bearing(c_out, from_end=False)
        if b_in is None or b_out is None:
            continue
        delta = ((b_out - b_in + 180.0) % 360.0) - 180.0
        phrase, sym = _maneuver(delta)
        in_name = _first_name(_edge_data(G, u, v, k))
        out_name = _first_name(_edge_data(G, u2, v2, k2))
        is_uturn = abs(delta) > 170.0
        turning = sym != "Straight" or is_uturn
        if not turning and out_name == in_name:
            continue  # straight through on the same street
        onto = f" onto {out_name}" if out_name else ""
        if is_uturn:
            text = f"Turn around on {out_name}" if out_name else "Turn around"
        elif turning:
            text = f"{phrase}{onto}"
        else:
            text = f"Continue onto {out_name}" if out_name else "Continue"
        raw.append((G.nodes[v]["y"], G.nodes[v]["x"], text, out_name,
                    "Straight" if is_uturn else sym, abs(delta)))
    uL, vL, kL, _ = route[-1]
    raw.append((G.nodes[vL]["y"], G.nodes[vL]["x"],
                "Arrive at destination", _first_name(_edge_data(G, uL, vL, kL)),
                "End", 999.0))

    # Debounce only SLIGHT wiggle (<50°) that's close together. NEVER drop a
    # genuine turn (>=50°) or a U-turn — that was the wrong-way bug. Start/End
    # always kept. Returned cues are 5-tuples (the magnitude is dropped).
    cues = [raw[0][:5]]
    for c in raw[1:-1]:
        if c[5] >= 50.0 or _cue_dist_m(cues[-1], c) >= min_gap_m:
            cues.append(c[:5])
    if len(raw) > 1:
        cues.append(raw[-1][:5])
    return cues


def _edge_data(G, u, v, key):
    """Edge data dict, tolerant of synthetic deadhead keys."""
    if v in G[u] and key in G[u][v]:
        return G[u][v][key]
    return next(iter(G[u][v].values()))


def _edge_coords(G, u, v, key):
    """
    Return [(lon, lat), ...] for edge (u,v,key), oriented from u -> v.
    Uses the OSM geometry if the street is curved, else a straight segment
    between the two intersection nodes.
    """
    data = _edge_data(G, u, v, key)
    geom = data.get("geometry")
    if geom is not None:
        coords = list(geom.coords)  # (lon, lat) pairs
    else:
        coords = [
            (G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"]),
        ]
    # orient toward v
    ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
    if coords and (abs(coords[0][0] - ux) + abs(coords[0][1] - uy)) > \
       (abs(coords[-1][0] - ux) + abs(coords[-1][1] - uy)):
        coords = list(reversed(coords))
    return coords


def _interp_elevations(G, u, v, coords):
    """
    Elevation (m) for each (lon,lat) in ``coords`` (oriented u->v), linearly
    interpolated between the two intersections' node elevations by distance along
    the block. Returns a list of None if either endpoint lacks 'elevation'.
    """
    zu = G.nodes[u].get("elevation")
    zv = G.nodes[v].get("elevation")
    if zu is None or zv is None or not coords:
        return [None] * len(coords)
    if len(coords) == 1:
        return [zu]
    cum = [0.0]
    for (lo1, la1), (lo2, la2) in zip(coords, coords[1:]):
        dx = (lo2 - lo1) * math.cos(math.radians((la1 + la2) / 2.0))
        dy = (la2 - la1)
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1] or 1.0
    return [zu + (zv - zu) * (d / total) for d in cum]


def route_to_gpx(G, route, path, name="Amble route", cues=True):
    """
    Write the ordered route as a GPX track, plus turn-by-turn cues as <wpt>
    waypoints (the format WorkOutDoors and other apps read for spoken/on-screen
    directions; pass cues=False to omit them). If the graph has node elevations,
    each trackpoint gets an <ele> so the watch can show a climb profile. Returns
    (path, n_trackpoints).
    """
    pts = []
    last = None
    for (u, v, key, _dead) in route:
        coords = _edge_coords(G, u, v, key)
        eles = _interp_elevations(G, u, v, coords)
        for (lon, lat), ele in zip(coords, eles):
            if (lon, lat) != last:
                pts.append((lat, lon, ele))
                last = (lon, lat)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Amble" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
    ]
    if cues:
        for lat, lon, text, street, sym in route_to_cues(G, route):
            lines += [
                f'  <wpt lat="{lat:.7f}" lon="{lon:.7f}">',
                f"    <name>{sx.escape(text)}</name>",
                f"    <desc>{sx.escape(street)}</desc>",
                f"    <sym>{sx.escape(sym)}</sym>",
                f"    <type>{sx.escape(sym)}</type>",
                "  </wpt>",
            ]
    lines.append(f"  <trk><name>{sx.escape(name)}</name><trkseg>")
    for lat, lon, ele in pts:
        if ele is None:
            lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}"></trkpt>')
        else:
            lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
                         f"<ele>{ele:.1f}</ele></trkpt>")
    lines += ["  </trkseg></trk>", "</gpx>"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path, len(pts)


def route_to_geojson(G, route, path):
    """Each traversed edge as a LineString feature; deadheads flagged."""
    feats = []
    for order, (u, v, key, dead) in enumerate(route):
        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": _edge_coords(G, u, v, key),
            },
            "properties": {
                "order": order,
                "deadhead": bool(dead),
                "name": _edge_data(G, u, v, key).get("name", ""),
                "length_m": round(_edge_data(G, u, v, key).get("length", 0.0), 1),
            },
        })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(path, "w") as f:
        json.dump(fc, f)
    return path


def progress_to_geojson(G, walked_ids, edge_id_fn, path):
    """
    Map of the whole network coloured by status. ``walked_ids`` is a set of
    edge identifiers; ``edge_id_fn(u, v, key)`` returns the same identifier
    scheme used by the progress store.
    """
    feats = []
    for u, v, key, data in G.edges(keys=True, data=True):
        eid = edge_id_fn(u, v, key)
        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": _edge_coords(G, u, v, key),
            },
            "properties": {
                "walked": eid in walked_ids,
                "name": data.get("name", ""),
            },
        })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(path, "w") as f:
        json.dump(fc, f)
    return path
