"""
trace.py — map-match a RECORDED GPS track onto the graph and mark it walked.

`plan` produces a route you follow; `done` records that planned route. But a walk
you actually recorded on your phone (a GPX track log) is a noisy list of fixes,
not a graph route — so it needs MATCHING before it can count toward coverage.

The match is an edge-state Hidden-Markov-Model (Newson & Krumm 2009), tuned to
be conservative — it would rather under-credit than fabricate a street:

  1. CANDIDATES: each fix proposes the nearby STREET edges (sidewalks excluded,
     so a fix on the sidewalk credits the street it parallels).
  2. VITERBI: pick the edge sequence maximising emission (fix->edge distance) +
     transition (graph route between projection points vs. the straight gap). A
     transition whose route blows past the cap is a GPS DROPOUT — the run breaks
     and the skip is counted; the detour is never walked.
  3. CREDIT an edge when the GPS swept >= _MARK_FRAC of it. Three crediting
     paths: the Viterbi sweep; connectors actually traversed (only when the route
     ~= the straight gap AND both fixes sit on-street, so a long detour is never
     fabricated — the same gate covers the connector's endpoint edges); and a
     direct "near" pass over the single nearest candidate per fix (so a
     fully-walked sub-block is recovered without crediting a parallel street).
     Traversal is never inferred from endpoints alone; unobserved middles of a
     block stay partial until actually observed.

Identity (which edge is which) is the same collision-free `progress.edge_id`
used everywhere else, so a matched track merges cleanly into the existing store.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import networkx as nx

from . import progress as prog


def _local(tag: str) -> str:
    """Local name of a namespaced XML tag ('{ns}trkpt' -> 'trkpt')."""
    return tag.rsplit("}", 1)[-1]


def parse_gpx(path: str):
    """Ordered list of (lat, lon) track points. Namespace-agnostic, so it reads
    GPX 1.0, 1.1, or vendor files without caring about the exact xmlns."""
    root = ET.parse(path).getroot()
    pts = []
    for el in root.iter():
        if _local(el.tag) == "trkpt":
            try:
                pts.append((float(el.get("lat")), float(el.get("lon"))))
            except (TypeError, ValueError):
                continue
    return pts


def parse_gpx_segments(path: str):
    """Return GPX track segments without inventing links across pauses/gaps."""
    root = ET.parse(path).getroot()
    segments = []
    for seg in root.iter():
        if _local(seg.tag) != "trkseg":
            continue
        pts = []
        for el in seg.iter():
            if _local(el.tag) != "trkpt":
                continue
            try:
                pts.append((float(el.get("lat")), float(el.get("lon"))))
            except (TypeError, ValueError):
                continue
        if pts:
            segments.append(pts)
    return segments or ([parse_gpx(path)] if parse_gpx(path) else [])


def _parse_gpx_segments_timed(path: str):
    """Track segments as lists of (lat, lon, epoch_seconds_or_None)."""
    import datetime
    root = ET.parse(path).getroot()
    segments = []
    for seg in root.iter():
        if _local(seg.tag) != "trkseg":
            continue
        pts = []
        for el in seg.iter():
            if _local(el.tag) == "trkpt":
                try:
                    pts.append([float(el.get("lat")), float(el.get("lon")),
                                None])
                except (TypeError, ValueError):
                    continue
            elif _local(el.tag) == "time" and pts and pts[-1][2] is None:
                try:
                    pts[-1][2] = datetime.datetime.fromisoformat(
                        (el.text or "").strip().replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
        if pts:
            segments.append([tuple(p) for p in pts])
    return segments


# A recording accidentally left running on BART/Muni/a car poisons coverage:
# the surface stretch of a train line can hug a street closely enough that
# riding it credits blocks nobody walked. Sustained speed is the honest
# discriminator — brisk walking is ~6 km/h, running ~12; transit holds >13
# for hundreds of metres.
_VEHICLE_KMH = 13.0
_VEHICLE_MIN_RUN_M = 150.0
_VEHICLE_MIN_STEPS = 3


def strip_vehicle_runs(path: str):
    """GPX segments with sustained vehicle-speed runs REMOVED (and the segment
    split there, so nothing bridges across the ride). Untimed points can't be
    speed-checked and pass through. Returns ``(segments, stats)`` where stats
    has ``vehicle_m`` / ``n_vehicle_fixes`` for reporting."""
    timed = _parse_gpx_segments_timed(path)
    if not timed:
        return parse_gpx_segments(path), {"vehicle_m": 0.0,
                                          "n_vehicle_fixes": 0}
    out, veh_m, veh_n = [], 0.0, 0
    for seg in timed:
        vehicular = [False] * max(0, len(seg) - 1)
        for i, (a, b) in enumerate(zip(seg, seg[1:])):
            if a[2] is not None and b[2] is not None and b[2] > a[2]:
                d = _geo_m(a[0], a[1], b[0], b[1])
                if d / (b[2] - a[2]) * 3.6 > _VEHICLE_KMH and d < 2000.0:
                    vehicular[i] = True
        # maximal runs of consecutive vehicular steps, long enough to be a ride
        drop = set()
        i = 0
        while i < len(vehicular):
            if not vehicular[i]:
                i += 1
                continue
            j = i
            run_m = 0.0
            while j < len(vehicular) and vehicular[j]:
                run_m += _geo_m(seg[j][0], seg[j][1], seg[j+1][0], seg[j+1][1])
                j += 1
            if j - i >= _VEHICLE_MIN_STEPS and run_m >= _VEHICLE_MIN_RUN_M:
                drop.update(range(i + 1, j))     # interior fixes of the ride
                veh_m += run_m
                veh_n += j - i - 1
            i = j
        cur = []
        for idx, p in enumerate(seg):
            if idx in drop:
                if len(cur) > 1:
                    out.append([(la, lo) for la, lo, _t in cur])
                cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            out.append([(la, lo) for la, lo, _t in cur])
    return out, {"vehicle_m": veh_m, "n_vehicle_fixes": veh_n}


def track_name(path: str):
    """The <trk><name> of a GPX file, or None — used as a default walk note."""
    root = ET.parse(path).getroot()
    for trk in root.iter():
        if _local(trk.tag) != "trk":
            continue
        for ch in trk:
            if _local(ch.tag) == "name" and ch.text and ch.text.strip():
                return ch.text.strip()
    return None


def _geo_m(lat1, lon1, lat2, lon2) -> float:
    """Equirectangular metres — matches network.nearest_node's approximation,
    plenty accurate at the block scale this matcher works at."""
    dlat = (lat2 - lat1) * 110540.0
    dlon = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dlat, dlon)


# Unnamed walk-surface footways that SHADOW a named street: sidewalks, marked
# crossings, desire-line connectors. A GPS fix taken on the sidewalk should credit
# the street it parallels, not the sidewalk — so these are excluded as match
# CANDIDATES (the matcher snaps to streets). Named footways (park paths, named
# stairways) and unnamed ROADS stay eligible, because those you genuinely walk.
_FOOT_HIGHWAYS = {
    "footway", "path", "pedestrian", "steps", "cycleway", "corridor",
    "crossing", "construction",
}
# Dedicated transit guideways you can't walk: never a candidate, never routed on.
# CAREFUL: highway=busway is ALSO tagged on the transit-lane segments of real,
# very-walkable NAMED streets (Market, Judah, Church, The Embarcadero). So a
# busway is non-walkable only when it's UNNAMED (a standalone guideway) or carries
# a dedicated-guideway name (progress.NON_WALKABLE_NAMES) — mirroring is_required.
# (Excluding busway wholesale dropped every busway-tagged block of Market et al.
# from matching, leaving walked corridors dashed.)
_NON_WALKABLE_HW = {"bus_guideway"}

# HMM map-matcher parameters (Newson & Krumm 2009). SIGMA = GPS noise scale for
# the emission term; BETA = tolerance for how far a route distance may stray from
# the straight-line gap before it's treated as an implausible detour.
_SIGMA_M = 20.0
_BETA_M = 25.0
_RADIUS_M = 65.0      # canonical centerlines may sit well inside wide rights-of-way
_MAX_CAND = 8         # enough alternatives for intersections without exploding HMM cost


def _hw_set(data):
    h = data.get("highway")
    return set(h) if isinstance(h, list) else {h}


def _length(data, default=0.0):
    try:
        return float(data.get("length", default))
    except (TypeError, ValueError):
        return float(default)


def _is_unnamed_foot(data) -> bool:
    return bool(_hw_set(data) & _FOOT_HIGHWAYS) and not prog._edge_name(data)


def _non_walkable(data) -> bool:
    h = _hw_set(data)
    if h & _NON_WALKABLE_HW:
        return True
    nm = prog._edge_name(data)
    if nm in prog.NON_WALKABLE_NAMES:
        return True
    return "busway" in h and not nm        # an UNNAMED standalone busway only


def _route_weight(u, v, kd):
    """Shortest-path weight over a MultiGraph edge: prefer a named street over
    an unnamed footway when both connect the same endpoints, otherwise use the
    shortest walkable length. Returns None if every parallel edge is a transit
    guideway."""
    best, best_pref = None, None
    for _k, d in kd.items():
        if _non_walkable(d):
            continue
        pref = _edge_pref(d)
        L = _length(d, 0.0)
        if best is None or pref < best_pref or (pref == best_pref and L < best):
            best, best_pref = L, pref
    return best


def _is_road_like(data) -> bool:
    """Street/road-like edges that should dominate over sidewalk shadows."""
    h = _hw_set(data)
    return not bool(h & _FOOT_HIGHWAYS) and not _non_walkable(data)


def _edge_pref(d):
    """Preference for a parallel edge. Roads/streets beat sidewalk shadows;
    footpaths only win when they are genuinely non-road paths."""
    if prog._edge_name(d) and _is_road_like(d):
        return (0, _length(d, 0.0))
    if _is_unnamed_foot(d):
        return (2, _length(d, 0.0))
    return (1, _length(d, 0.0))


def _min_walkable_key(G, x, y):
    best, bk = None, None
    for k, d in G[x][y].items():
        if _non_walkable(d):
            continue
        if _is_unnamed_foot(d) and any(
            prog._edge_name(other) and _is_road_like(other)
            for _ok, other in G[x][y].items()
        ):
            continue
        pref = _edge_pref(d)
        L = _length(d, float("inf"))
        if best is None or pref < best or (pref == best and L < _length(G[x][y][bk], float("inf"))):
            best, bk = pref, k
    return bk


def _edge_poly(G, u, v, d):
    """Edge geometry as a list of (lon, lat); falls back to the straight node-to-
    node segment when no geometry attribute is present."""
    g = d.get("geometry")
    if g is not None and hasattr(g, "coords"):
        try:
            return [(float(x), float(y)) for x, y in g.coords]
        except (TypeError, ValueError):
            pass
    try:
        return [(float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))]
    except (KeyError, ValueError, TypeError):
        return None


def _candidate_index(G):
    """A grid spatial index of surfaces that can evidence a coverage target.

    A prepared graph includes both canonical representatives and tightly aligned
    sidewalk observation aliases.  The latter never become completion targets;
    they merely let a phone trace recorded on the sidewalk credit its street.
    """
    cached = G.graph.get("_amble_cand_index")
    if cached is not None:
        return cached
    cell = _RADIUS_M / 111320.0 * 1.5
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if _non_walkable(d):
            continue
        # Match observations to canonical coverage targets, not arbitrary OSM
        # surfaces.  On an unprepared synthetic graph this falls back to the
        # historical named-way rule used by tests.
        if G.graph.get("amble_model") == "canonical-passages-v1":
            if not d.get("coverage_id"):
                continue
        elif _is_unnamed_foot(d):
            continue
        poly = _edge_poly(G, u, v, d)
        if poly is None:
            continue
        try:
            ula, ulo = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
            vla, vlo = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])
        except (KeyError, ValueError, TypeError):
            continue
        L = _length(d, 0.0)
        edges.append((u, v, _edge_pref(d), L, ula, ulo, vla, vlo, poly, k))
    grid = {}
    for idx, (u, v, pref, L, ula, ulo, vla, vlo, poly, k) in enumerate(edges):
        cells = set()
        pts = poly if len(poly) > 1 else poly * 2
        for (lo1, la1), (lo2, la2) in zip(pts, pts[1:]):
            steps = int(max(abs(lo2 - lo1), abs(la2 - la1)) / cell) + 1
            for s in range(steps + 1):
                t = s / steps
                cells.add((int((lo1 + (lo2 - lo1) * t) / cell),
                           int((la1 + (la2 - la1) * t) / cell)))
        for c in cells:
            grid.setdefault(c, []).append(idx)

    # Map every surface's fraction axis onto its REPRESENTATIVE's polyline, in
    # the rep's canonical geographic orientation. Evidence measured on an alias
    # then lands scale/offset-correct on the rep's own [0,1] frame: a 60 m
    # sidewalk inset in a 100 m block maps to [0.2, 0.8] (never [0, 1]), and
    # orientation comes from geometry — a lexicographic endpoint sort could
    # FLIP a near-axis-aligned alias whose endpoints jitter by centimetres,
    # silently inverting its evidence.
    rep_geo = {}                       # coverage key -> rep geometry
    for u, v, _pref, L, ula, ulo, vla, vlo, poly, k in edges:
        d = G[u][v][k]
        if d.get("coverage_id") and not d.get("coverage_required"):
            continue
        key = prog.edge_id(G, u, v, k)
        if key not in rep_geo:
            flip = ((ula, ulo) > (vla, vlo))
            rep_geo[key] = (poly, ulo, ula, flip, L)
    rep_map = []                       # per edge idx: (a, b, rep_length)
    for u, v, _pref, L, ula, ulo, vla, vlo, poly, k in edges:
        key = prog.edge_id(G, u, v, k)
        rep = rep_geo.get(key)
        if rep is None:
            flip = ((ula, ulo) > (vla, vlo))
            rep_map.append(((1.0, 0.0, L) if flip else (0.0, 1.0, L)))
            continue
        rpoly, rulon, rulat, rflip, rL = rep
        ref_cos = math.cos(math.radians(ula))
        _da, a = _poly_projection(ula, ulo, rpoly, rulon, rulat, ref_cos)
        _db, b = _poly_projection(vla, vlo, rpoly, rulon, rulat, ref_cos)
        if rflip:
            a, b = 1.0 - a, 1.0 - b
        if abs(b - a) < 1e-9:
            b = a + 1e-9
        rep_map.append((a, b, rL))
    index = (edges, grid, cell, rep_map)
    G.graph["_amble_cand_index"] = index
    return index


def _rep_frac(rep_map, idx, frac):
    """Fraction along surface ``idx`` mapped onto its representative's axis."""
    a, b, _rL = rep_map[idx]
    return min(1.0, max(0.0, a + frac * (b - a)))


def _canon_frac(G, u, v, frac):
    """Re-orient a fraction along u->v to the geographic endpoint order used by
    progress.canonical_intervals. Parallel surfaces of ONE coverage target
    (roadway + sidewalk aliases) then share a single fraction axis, so evidence
    that hops between surfaces accumulates instead of fragmenting."""
    try:
        pu = (float(G.nodes[u]["y"]), float(G.nodes[u]["x"]))
        pv = (float(G.nodes[v]["y"]), float(G.nodes[v]["x"]))
    except (KeyError, TypeError, ValueError):
        pu, pv = str(u), str(v)
    return frac if pu <= pv else 1.0 - frac


def _poly_projection(plat, plon, poly, u_lon, u_lat, ref_cos):
    """Return (distance_m, fraction) along the actual edge polyline.

    Fractions are oriented from graph node ``u``.  Unlike endpoint-chord
    projection this remains correct on curved streets, trails, and switchbacks.
    """
    if len(poly) < 2:
        return float("inf"), 0.0
    # Orient geometry toward u.
    if _geo_m(u_lat, u_lon, poly[-1][1], poly[-1][0]) < \
       _geo_m(u_lat, u_lon, poly[0][1], poly[0][0]):
        poly = list(reversed(poly))
    mx = 111320.0 * ref_cos
    pts = [(lon * mx, lat * 110540.0) for lon, lat in poly]
    px, py = plon * mx, plat * 110540.0
    seglens = [math.hypot(bx - ax, by - ay)
               for (ax, ay), (bx, by) in zip(pts, pts[1:])]
    total = sum(seglens) or 1.0
    best_d, best_along, before = float("inf"), 0.0, 0.0
    for ((ax, ay), (bx, by)), seglen in zip(zip(pts, pts[1:]), seglens):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0,
            ((px - ax) * dx + (py - ay) * dy) / L2))
        qx, qy = ax + t * dx, ay + t * dy
        dist = math.hypot(px - qx, py - qy)
        if dist < best_d:
            best_d = dist
            best_along = before + t * seglen
        before += seglen
    return best_d, max(0.0, min(1.0, best_along / total))


def _empty_result(points):
    return {"edge_ids": set(), "edge_meta": {}, "n_points": len(points),
            "n_snapped": 0, "n_unmatched": len(points), "n_skipped": 0,
            "raw_m": 0.0, "raw_gap_m": 0.0, "n_raw_gaps": 0,
            "near_target_trace_m": 0.0,
            "matched_m": 0.0, "named_m": 0.0,
            "edge_spans": {}}


# An edge is credited once the GPS swept most of the edge, so a brief pass-by
# that merely touches an endpoint or part of a block does not count as a fully
# walked segment.
_MARK_FRAC = 0.6

# Direct on-street crediting, independent of the Viterbi path (which can smooth a
# weaving walk onto too few edges and undercount). A fix within _NEAR_MARK_M of a
# street is "on" it; once the track sweeps most of the edge it's credited.
# Capped at _NEAR_MARK_M (< the 30 m fabrication bar), so it can only credit a
# street the GPS actually sat next to — never a fabricated detour.
_NEAR_MARK_M = 30.0
_NEAR_MOVE_FRAC = 0.6
def match_trace(G, points, gap_base_m: float = 150.0, gap_factor: float = 8.0):
    """
    Map-match a recorded GPS trace (list of (lat, lon)) onto graph edges with an
    edge-state Hidden-Markov-Model matcher (Newson & Krumm 2009).

    Each fix proposes the nearby STREET edges as candidate states (sidewalks
    excluded, so a fix on the sidewalk credits the street it parallels). Viterbi
    picks the edge sequence maximising an EMISSION term (perpendicular distance
    fix->edge, scale ``_SIGMA_M``) plus a TRANSITION term (how well the graph
    route between the projected points on consecutive edges matches the
    straight-line gap, scale ``_BETA_M``). Because the route is measured between
    PROJECTION POINTS — not intersections — it stays ~= the GPS gap for normal
    walking, so long blocks are never mistaken for dropouts. A transition whose
    route exceeds ``max(gap_base_m, gap_factor * gap)`` is rejected as a real GPS
    dropout: the detour is never fabricated. Routing keeps the full graph, so a
    street reachable only via a stairway is never stranded.

    An edge is marked walked when the projected points sweep >= ``_MARK_FRAC`` of
    it, or when it lies on a connector path actually traversed between two fixes.
    Returns: edge_ids, edge_meta, n_points, n_snapped, n_skipped, matched_m, named_m
    """
    edges, grid, cell, rep_map = _candidate_index(G)

    fixes = []  # drop consecutive near-duplicate fixes (a parked GPS)
    for lat, lon in points:
        if not fixes or _geo_m(fixes[-1][0], fixes[-1][1], lat, lon) > 1.0:
            fixes.append((lat, lon))
    if not fixes:
        return _empty_result(points)
    ref_cos = math.cos(math.radians(sum(f[0] for f in fixes) / len(fixes)))

    def candidates(lat, lon):
        cx, cy = int(lon / cell), int(lat / cell)
        idxs = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                idxs.update(grid.get((cx + dx, cy + dy), ()))
        scored = []
        for i in idxs:
            u, v, _pref, L, ula, ulo, vla, vlo, poly, _key = edges[i]
            d, frac = _poly_projection(lat, lon, poly, ulo, ula, ref_cos)
            if d <= _RADIUS_M:
                scored.append((d, i, u, v, L, frac))
        scored.sort(key=lambda x: x[0])
        # A road centerline and either sidewalk can all represent one passage.
        # Retain only the closest physical surface for that passage at each fix,
        # or interval distance would be counted two or three times.
        unique = []
        seen = set()
        for item in scored:
            _d, idx, u, v, _L, _frac = item
            k = edges[idx][-1]
            identity = prog.edge_id(G, u, v, k)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
            if len(unique) >= _MAX_CAND:
                break
        return unique                       # (perp, idx, u, v, L, frac)

    cand = [candidates(lat, lon) for lat, lon in fixes]

    # Evidence intervals are keyed by COVERAGE identity (edge_id), in canonical
    # geographic orientation. GPS wobbling between the parallel surfaces of one
    # block (roadway, either sidewalk — same coverage_id) must extend one
    # interval, not shatter it into sub-0.04 fragments that get dropped: that
    # fragmentation recorded fully-walked blocks as 15-50% partial.
    swept = {}            # coverage key -> disjoint contiguous [lo, hi] intervals
    span_rep = {}         # coverage key -> a candidate idx (for edge metadata)
    mark_set = set()      # (x, y) graph edges confirmed traversed between fixes
    skipped = 0

    def state_key(c):
        idx, u, v = c[1], c[2], c[3]
        return prog.edge_id(G, u, v, edges[idx][-1])

    def emis_lp(c):
        return -0.5 * (c[0] / _SIGMA_M) ** 2

    # Viterbi over edge states, broken into contiguous runs at every dropout. Each
    # run column carries its own state map, so backtracking never indexes by fix.
    run = []   # list of (col, bp, st, fix_j): col key->logprob;
    #            bp key->(prev_key, mark_payloads);
    #            st key->(idx, u, v, L, frac, perpendicular_distance)

    def finalize():
        if not run:
            return
        col = run[-1][0]
        if col:
            key = max(col, key=col.get)
            chosen = []
            for ci in range(len(run) - 1, -1, -1):
                _c, bp, st, fj = run[ci]
                prev_key, mark_pairs = bp[key]
                idx, u, v, L, frac, _perp = st[key]
                chosen.append((idx, frac, _perp, fj))
                mark_set.update(mark_pairs)
                key = prev_key
                if prev_key is None:
                    break
            chosen.reverse()
            # Preserve disjoint visits.  Two endpoint observations separated by
            # another street must not imply that the unobserved middle was walked.
            cur_key = None
            lo = hi = None
            prev_cf = prev_fj = None
            for idx, frac, perp, fj in chosen:
                # A far candidate can keep the HMM topologically stable, but it
                # is not observation evidence.  Treat it as an interval break.
                if perp > _NEAR_MARK_M:
                    if cur_key is not None:
                        swept.setdefault(cur_key, []).append([lo, hi])
                    cur_key = None; lo = hi = None
                    prev_cf = prev_fj = None
                    continue
                x, y, *_rest, bk = edges[idx]
                key = prog.edge_id(G, x, y, bk)
                cf = _rep_frac(rep_map, idx, frac)
                # CONTINUITY: extending an interval asserts the ground between
                # the previous and current fraction was walked. That assertion
                # is only honest when the along-block distance is explainable
                # by the physical step between the two fixes — otherwise a
                # hairpin's two ends (both near the walker) or an in-block GPS
                # dropout would fill a middle nobody observed.
                fresh_start = key != cur_key
                if not fresh_start and prev_fj is not None:
                    step = _geo_m(*fixes[prev_fj], *fixes[fj])
                    if step > gap_base_m or \
                            abs(cf - prev_cf) * rep_map[idx][2] > \
                            step * 1.8 + 15.0:
                        fresh_start = True
                if fresh_start:
                    if cur_key is not None:
                        swept.setdefault(cur_key, []).append([lo, hi])
                    cur_key = key; lo = hi = cf
                    span_rep.setdefault(key, idx)
                else:
                    lo, hi = min(lo, cf), max(hi, cf)
                prev_cf, prev_fj = cf, fj
            if cur_key is not None:
                swept.setdefault(cur_key, []).append([lo, hi])
        run.clear()

    def fresh(cj, j):
        col = {state_key(c): emis_lp(c) for c in cj}
        bp = {k: (None, set()) for k in col}
        st = {state_key(c): (c[1], c[2], c[3], c[4], c[5], c[0]) for c in cj}
        return col, bp, st, j

    for j, cj in enumerate(cand):
        if not cj:                        # nothing to snap to — end the run
            finalize()
            continue
        if not run:
            run.append(fresh(cj, j))
            continue
        prev_col, _pbp, prev_st, _pj = run[-1]
        gap = _geo_m(fixes[j - 1][0], fixes[j - 1][1], fixes[j][0], fixes[j][1])
        cap = max(gap_base_m, gap_factor * gap)
        srcs = set()
        for (idx, u, v, L, frac, _perp) in prev_st.values():
            srcs.add(u); srcs.add(v)
        reach = {s: nx.single_source_dijkstra(G, s, cutoff=cap, weight=_route_weight)
                 for s in srcs}
        col, bp = {}, {}
        for c in cj:
            bkey = state_key(c)
            _bidx, bu, bv, bL, bfrac = c[1], c[2], c[3], c[4], c[5]
            emis = emis_lp(c)
            best = None
            for akey, lpa in prev_col.items():
                _aidx, au, av, aL, afrac, aperp = prev_st[akey]
                if _aidx == _bidx:            # stayed on the same physical surface
                    route, mp = abs(afrac - bfrac) * aL, set()
                else:
                    route, pack = None, None
                    for ex, ealong in ((au, afrac * aL), (av, (1 - afrac) * aL)):
                        dist_map, path_map = reach[ex]
                        for en, ialong in ((bu, bfrac * bL), (bv, (1 - bfrac) * bL)):
                            if en not in dist_map:
                                continue
                            r = ealong + dist_map[en] + ialong
                            if route is None or r < route:
                                route, pack = r, (path_map[en], ealong, ialong)
                    if route is None:
                        continue
                    path, ealong, ialong = pack
                    # Credit the streets ALONG the connector only when the graph
                    # route is close to the straight-line gap — i.e. the GPS
                    # actually walked that path. A route much longer than the gap
                    # is a detour the matcher took to keep the states plausible,
                    # NOT walked ground; crediting its edges fabricates streets
                    # the GPS never touched (phantom coverage, sometimes in a
                    # whole other neighbourhood). The endpoint edges are still
                    # credited below when their own fix actually swept them.
                    # Sparse-but-continuous GPS fixes may skip a whole block.
                    # Credit the shortest connector only when its length closely
                    # agrees with the observed displacement; a generous fixed
                    # 80 m allowance fabricated alternate grid streets.
                    # A connector is only claimable across an OBSERVED-scale
                    # step. A fix-free jump beyond gap_base_m is missing
                    # observation (raw_m already refuses it) — and a subway
                    # under a street passes the route~=gap test by
                    # construction (stations ON the street, tunnel straight
                    # beneath it), so without this bound a BART ride credits
                    # every block it passes under.
                    plausible = gap <= gap_base_m and \
                        (route - gap) <= max(20.0, 0.20 * gap) and \
                        aperp <= _NEAR_MARK_M and c[0] <= _NEAR_MARK_M
                    mp = set()
                    if plausible:
                        mp = {(x, y) for x, y in zip(path, path[1:])}
                        # Endpoint edges obey the SAME plausibility gate as the
                        # connector, and record only the fraction range ACTUALLY
                        # traversed (projection point -> exit/entry node) as a
                        # span payload. A full-edge mark here would surface as a
                        # [[0,1]] span, recording a block complete from an
                        # in-and-out visit that never reached its far end.
                        exit_node, entry_node = path[0], path[-1]
                        sa = (0.0, afrac) if exit_node == au else (afrac, 1.0)
                        sb = (0.0, bfrac) if entry_node == bu else (bfrac, 1.0)
                        fa = sorted(_rep_frac(rep_map, _aidx, f) for f in sa)
                        fb = sorted(_rep_frac(rep_map, _bidx, f) for f in sb)
                        mp.add(("span", akey, fa[0], fa[1]))
                        mp.add(("span", bkey, fb[0], fb[1]))
                if route > cap:
                    continue
                lp = lpa + emis - abs(route - gap) / _BETA_M
                if best is None or lp > best[0]:
                    best = (lp, akey, mp)
            if best is not None:
                col[bkey] = best[0]
                bp[bkey] = (best[1], best[2])
        if col:
            run.append((col, bp, {
                state_key(c): (c[1], c[2], c[3], c[4], c[5], c[0]) for c in cj
            }, j))
        else:                             # no plausible link — a GPS dropout
            skipped += 1
            finalize()
            run.append(fresh(cj, j))
    finalize()

    # turn swept fractions + traversed connectors into the marked-edge set
    edge_meta = {}

    # Coverage distance belongs to the representative block, never to the
    # slightly longer/shorter sidewalk geometry used to observe it.
    target_meta = {}
    for x, y, k, d in G.edges(keys=True, data=True):
        if G.graph.get("amble_model") == "canonical-passages-v1" and not \
                d.get("coverage_required"):
            continue
        eid = prog.edge_id(G, x, y, k)
        target_meta.setdefault(eid, (_length(d, 0.0), prog.is_required(d)))

    def mark_candidate(idx):
        x, y, *_rest, bk = edges[idx]
        d = G[x][y][bk]
        eid = prog.edge_id(G, x, y, bk)
        edge_meta[eid] = target_meta.get(
            eid, (_length(d, 0.0), prog.is_required(d)))

    def _interval_fraction(intervals):
        return sum(hi - lo for lo, hi in prog._merge_intervals(intervals))

    for key, intervals in swept.items():
        if _interval_fraction(intervals) >= _MARK_FRAC:
            mark_candidate(span_rep[key])
    endpoint_spans = []       # ("span", key, lo, hi): partially-traversed
    for item in mark_set:     # entry/exit edges of plausibility-gated routes
        if len(item) == 4:
            endpoint_spans.append(item)
            continue
        x, y = item
        bk = _min_walkable_key(G, x, y)
        if bk is None:
            continue
        d = G[x][y][bk]
        if d.get("coverage_id") or prog.is_required(d):
            eid = prog.edge_id(G, x, y, bk)
            edge_meta[eid] = target_meta.get(
                eid, (_length(d, 0.0), bool(prog.is_required(d))))

    # direct on-street credit: any street the GPS sat on (<= _NEAR_MARK_M) and
    # swept along, regardless of which edge the Viterbi smoothed the path onto.
    # Accumulated per COVERAGE identity in canonical orientation, so the nearest
    # surface flipping between a block's roadway and its sidewalk aliases keeps
    # extending one interval instead of breaking it.
    near = {}
    nearest_at = set()        # coverage keys that were the NEAREST at >= 1 fix
    active = {}               # key -> [lo, hi, last_fix_j, last_cf]
    for j, cj in enumerate(cand):
        if not cj:
            for akey, st4 in active.items():
                near.setdefault(akey, []).append(st4[:2])
            active = {}
            continue
        # Only the nearest candidate contributes direct interval evidence.
        # Near-tied parallel passages can be physically indistinguishable from
        # GPS alone; crediting every tie produced impossible >100% distances.
        _d, idx, _u, _v, _L, frac = cj[0]
        key = prog.edge_id(G, _u, _v, edges[idx][-1])
        nearest_at.add(key)
        current = {}
        if _d <= _NEAR_MARK_M:
            current[key] = (idx, _rep_frac(rep_map, idx, frac))
            span_rep.setdefault(key, idx)
        for akey in list(active):
            if akey not in current:
                near.setdefault(akey, []).append(active.pop(akey)[:2])
        for akey, (cidx, cf) in current.items():
            st4 = active.get(akey)
            if st4 is None:
                active[akey] = [cf, cf, j, cf]
                continue
            lo, hi, lj, lcf = st4
            # Same continuity rule as the Viterbi sweep: only extend the
            # interval when the along-block distance is explainable by the
            # physical step — otherwise a hairpin whose two ends both sit
            # within 30 m, or an in-block GPS dropout, would fill an
            # unobserved middle.
            step = _geo_m(*fixes[lj], *fixes[j])
            if step > gap_base_m or \
                    abs(cf - lcf) * rep_map[cidx][2] > step * 1.8 + 15.0:
                near.setdefault(akey, []).append([lo, hi])
                active[akey] = [cf, cf, j, cf]
            else:
                active[akey] = [min(lo, cf), max(hi, cf), j, cf]
    for akey, st4 in active.items():
        near.setdefault(akey, []).append(st4[:2])
    # Credit an edge the GPS swept >= _NEAR_MOVE_FRAC of AND that was the NEAREST
    # candidate at some fix — so you were genuinely ON it, not walking alongside a
    # parallel street/carriageway (those are never the nearest). This also credits
    # the overlapping same-street blocks of a wide/diagonal street (Market), each
    # nearest over its own stretch, that a distance-tie gate split below the bar.
    for key, intervals in near.items():
        if key in nearest_at and _interval_fraction(intervals) >= _NEAR_MOVE_FRAC:
            mark_candidate(span_rep[key])

    # Per-edge swept FRACTION spans, for interval (partial) coverage tracking:
    # the 0->1 range of each block the GPS was actually on — including blocks it
    # only partly walked (below the single-walk completion bar). The store unions
    # these across walks, so walking half a block now and the rest later finishes
    # it. Sourced from the tie-gated/Viterbi sweeps plus any bridge-credited edge,
    # so a parallel street is never recorded (same guards as crediting).
    edge_spans = {}

    def _add_spans(eid, intervals):
        # Intervals live on the representative's axis (_rep_frac); a fragment
        # narrower than 0.04 is a single-fix touch, not a sweep.
        meaningful = [[lo, hi] for lo, hi in intervals if hi - lo >= 0.04]
        if meaningful:
            edge_spans[eid] = prog._merge_intervals(
                edge_spans.get(eid, []) + meaningful)

    for _key, intervals in near.items():
        _add_spans(_key, intervals)
    for _key, intervals in swept.items():
        _add_spans(_key, intervals)
    for _s, _key, _lo, _hi in endpoint_spans:
        meta = target_meta.get(_key)
        if meta and meta[1]:      # goal passages only, same gate as mark_set
            _add_spans(_key, [[_lo, _hi]])
    # A goal passage whose accumulated evidence (sweep + near + gated
    # entry/exit ranges) reaches the mark bar is credited like any other: a
    # block traversed end-to-end as an entry leg shows up here as a full span.
    for eid, intervals in edge_spans.items():
        if eid not in edge_meta and _interval_fraction(intervals) >= _MARK_FRAC:
            meta = target_meta.get(eid)
            if meta and meta[1]:
                edge_meta[eid] = meta
    # Any credited block with no recorded fraction is a plausibility-gated
    # connector (walked THROUGH between two on-street fixes) — record it full.
    # Traversal is never inferred merely because both endpoints or neighbouring
    # blocks were observed: dense city grids offer many alternate paths, and
    # that inference silently fabricates completion. Uncertain gaps stay partial.
    for eid in edge_meta:
        edge_spans.setdefault(eid, [[0.0, 1.0]])

    # Report interval distance, not only whole completed edges.  This makes the
    # import accounting explain partial blocks instead of making distance vanish.
    matched_m = named_m = 0.0
    for eid, intervals in edge_spans.items():
        length_m, req = target_meta.get(eid, edge_meta.get(eid, (0.0, False)))
        frac = _interval_fraction(intervals)
        matched_m += length_m * frac
        if req:
            named_m += length_m * frac
    raw_steps = [_geo_m(*a, *b) for a, b in zip(fixes, fixes[1:])]
    # A multi-kilometre jump between phone fixes is missing observation, not
    # distance walked.  Do not inflate the physical baseline with it.  GPX
    # <trkseg> boundaries are already handled by match_trace_segments; this
    # catches exporters that leave dropouts inside one segment.
    raw_m = sum(d for d in raw_steps if d <= gap_base_m)
    raw_gap_m = sum(d for d in raw_steps if d > gap_base_m)
    near_target_trace_m = sum(
        distance for distance, left, right in zip(raw_steps, cand, cand[1:])
        if distance <= gap_base_m and left and right and
        left[0][0] <= _NEAR_MARK_M and right[0][0] <= _NEAR_MARK_M)
    return {
        "edge_ids": set(edge_meta),
        "edge_meta": edge_meta,
        "edge_spans": edge_spans,   # eid -> [lo, hi] swept fraction, for partial coverage
        "n_points": len(points),
        "n_snapped": sum(1 for c in cand if c),
        "n_unmatched": sum(1 for c in cand if not c),
        "n_skipped": skipped,
        "raw_m": raw_m,
        "raw_gap_m": raw_gap_m,
        "n_raw_gaps": sum(d > gap_base_m for d in raw_steps),
        "near_target_trace_m": near_target_trace_m,
        "matched_m": matched_m,
        "named_m": named_m,
    }


def match_trace_segments(G, segments, gap_base_m: float = 150.0,
                         gap_factor: float = 8.0):
    """Match independent GPX segments and merge evidence without bridging them."""
    results = [match_trace(G, pts, gap_base_m, gap_factor)
               for pts in segments if pts]
    if not results:
        return _empty_result([])
    spans = {}
    edge_meta = {}
    for result in results:
        edge_meta.update(result.get("edge_meta", {}))
        for eid, intervals in result.get("edge_spans", {}).items():
            spans[eid] = prog._merge_intervals(spans.get(eid, []) + intervals)
    return {
        "edge_ids": set().union(*(r.get("edge_ids", set()) for r in results)),
        "edge_meta": edge_meta,
        "edge_spans": spans,
        "n_points": sum(r["n_points"] for r in results),
        "n_snapped": sum(r["n_snapped"] for r in results),
        "n_unmatched": sum(r.get("n_unmatched", 0) for r in results),
        "n_skipped": sum(r["n_skipped"] for r in results),
        "raw_m": sum(r.get("raw_m", 0.0) for r in results),
        "raw_gap_m": sum(r.get("raw_gap_m", 0.0) for r in results),
        "n_raw_gaps": sum(r.get("n_raw_gaps", 0) for r in results),
        "near_target_trace_m": sum(
            r.get("near_target_trace_m", 0.0) for r in results),
        # These are physical assigned distances and therefore include repeated
        # traversal across independent GPX segments. Store intervals remain a union.
        "matched_m": sum(r["matched_m"] for r in results),
        "named_m": sum(r["named_m"] for r in results),
    }
