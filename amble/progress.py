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
    try:
        cid = G[u][v][key].get("coverage_id")
    except (KeyError, TypeError):
        cid = None
    if cid:
        return str(cid)
    return legacy_edge_id(u, v, key)


def legacy_edge_id(u, v, key) -> str:
    """Pre-canonical edge identity, retained for reading existing stores."""
    a, b = sorted((u, v), key=str)
    return f"{a}-{b}-{key}"


def canonical_intervals(G, u, v, intervals):
    """Orient fractional evidence by geographic endpoint, not graph direction."""
    try:
        pu = (float(G.nodes[u]["y"]), float(G.nodes[u]["x"]))
        pv = (float(G.nodes[v]["y"]), float(G.nodes[v]["x"]))
    except (KeyError, TypeError, ValueError):
        pu, pv = str(u), str(v)
    if pu <= pv:
        return [[lo, hi] for lo, hi in intervals]
    return [[1.0 - hi, 1.0 - lo] for lo, hi in intervals]


# bus_stop is a point feature OSM sometimes imports as a named "way"
# (e.g. "7th Street & Brannan Street") — never a walkable street.
NON_WALKABLE_HIGHWAYS = {"bus_stop"}

# Dedicated bus-rapid-transit guideways you can't walk. Matched by NAME, not by
# highway=busway: OSM tags transit-lane segments of REAL streets (Market, Judah,
# Church, The Embarcadero…) as busway too, and those streets are very much walked.
# Only the standalone guideways carry these names. Confirmed absent from SF's
# official Street Names + TIGER road lists.
NON_WALKABLE_NAMES = {"Van Ness Bus Rapid Transit", "Transbay Bus Ramp"}


def is_required(data) -> bool:
    """
    The project's must-walk set is every NAMED way (street, alley, stairway,
    named park path). Unnamed ways (sidewalks, crossings, desire lines, minor
    connectors) are walked only as connectors and don't count toward 100%.

    Two carve-outs:
      * things you can't walk never count, even when named — bus_stop labels
        (NON_WALKABLE_HIGHWAYS) and dedicated BRT guideways (NON_WALKABLE_NAMES).
      * a declared CORRIDOR (e.g. the Great Highway's overlapping names — see
        equivalents.py) counts ONCE. Only its canonical name is required; the
        redundant aliases are optional connectors, so you walk the strip once
        instead of two-to-four times. Everything else keeps the plain "named" rule.
    """
    if "coverage_required" in data:
        return bool(data.get("coverage_required"))
    from .equivalents import canonical_name
    h = data.get("highway")
    hset = set(h) if isinstance(h, list) else {h}
    if hset & NON_WALKABLE_HIGHWAYS:
        return False
    name = _edge_name(data)
    if not name or name in NON_WALKABLE_NAMES:
        return False
    return canonical_name(name) == name


def _records_for_edge(G, u, v, key, store):
    """Records supporting one canonical passage, including legacy aliases."""
    walked = store.get("walked", {})
    data = G[u][v][key]
    ids = [edge_id(G, u, v, key)]
    aliases = data.get("coverage_alias_ids") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    ids.extend(str(x) for x in aliases)
    # The representative's own old edge ID may predate alias annotation.
    ids.append(legacy_edge_id(u, v, key))
    out, seen = [], set()
    for eid in ids:
        if eid not in seen and eid in walked:
            out.append(walked[eid]); seen.add(eid)
    return out


def _combined_record(records):
    """Union evidence records for a canonical target without losing legacy data."""
    if not records:
        return None
    intervals = []
    for rec in records:
        if not isinstance(rec, dict) or rec.get("intervals") is None:
            intervals.append([0.0, 1.0])
        else:
            intervals.extend(rec.get("intervals") or [])
    first = min((r for r in records if isinstance(r, dict)),
                key=lambda r: r.get("date", ""), default={})
    out = dict(first)
    out["intervals"] = _merge_intervals(intervals)
    return out


# ── Interval (fractional) coverage ──────────────────────────────────────────
# A street block (edge) is NOT all-or-nothing. We record WHICH fraction of its
# 0->1 length the GPS has swept, accumulated across walks: walk half today and
# the rest next month and the intervals union to a finished block. A block is
# DONE once its covered length reaches within COMPLETE_END_TOL_M of full (GPS
# never quite reaches the two corners) — length-aware so a short stairway isn't
# held to the same fraction as a long boulevard — but never from less than
# COMPLETE_MIN_FRAC, so a stray perpendicular crossing can't complete a block.
COMPLETE_END_TOL_M = 15.0
COMPLETE_MIN_FRAC = 0.5


def _merge_intervals(intervals):
    """Union [lo, hi] fraction ranges into minimal disjoint ranges."""
    out = []
    for lo, hi in sorted([min(a, b), max(a, b)] for a, b in intervals):
        if out and lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def coverage_frac(rec) -> float:
    """Fraction (0..1) of an edge walked across all recorded walks. A legacy
    record (no ``intervals`` key — the old binary schema) means fully walked."""
    if isinstance(rec, dict):
        intervals = rec.get("intervals")
        if intervals is None:
            return 1.0
    else:
        intervals = rec
    return min(1.0, sum(hi - lo for lo, hi in _merge_intervals(intervals))) if intervals else 0.0


def is_complete(rec, length_m: float) -> bool:
    need = max(COMPLETE_MIN_FRAC, 1.0 - COMPLETE_END_TOL_M / max(float(length_m), 1.0))
    return coverage_frac(rec) >= need


def completed_id_set(G, store: dict) -> set:
    """edge_ids that are fully walked (DONE). Needs G for each edge's length."""
    walked = store.get("walked", {})
    out = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if not is_required(data):
            continue
        eid = edge_id(G, u, v, key)
        rec = _combined_record(_records_for_edge(G, u, v, key, store))
        if rec is not None and is_complete(rec, data.get("length", 0.0)):
            out.add(eid)
    return out


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
        if G.graph.get("amble_model") == "canonical-passages-v1" and not is_required(d):
            continue
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
    old_geo = {}
    for u, v, k, d in Gold.edges(keys=True, data=True):
        info = (_edge_name(d), _midpoint(Gold, u, v), d.get("length", 0.0))
        old_geo[edge_id(Gold, u, v, k)] = info
        old_geo[legacy_edge_id(u, v, k)] = info
    walked, migrated, not_found, merged = {}, 0, 0, 0
    unmatched = {}
    for key, rec in store.get("walked", {}).items():
        info = old_geo.get(key)
        match = _best_match(*info, new_idx) if info else None
        if match is None:
            not_found += 1
            unmatched[key] = rec
        elif match in walked:
            merged += 1
            walked[match] = _combined_record([walked[match], rec])
        else:
            walked[match] = rec
            migrated += 1
    out = dict(store)
    out["walked"] = walked
    if unmatched:
        out["migration_unmatched"] = unmatched
    out["schema_version"] = 2
    return out, migrated, not_found, merged


def load_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            store = json.load(f)
    else:
        store = {"walked": {}}
    # Migrate the old binary schema ({eid: {date, note}}) to interval coverage:
    # an edge that was marked walked under the old rule is treated as fully done.
    for rec in store.get("walked", {}).values():
        if isinstance(rec, dict) and "intervals" not in rec:
            rec["intervals"] = [[0.0, 1.0]]
    return store


def save_store(store: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record_spans(store: dict, edge_spans, when: str | None = None, note="") -> int:
    """Union each ``{edge_id: (lo, hi)}`` swept fraction-span into the store. Walk
    part of a block now and the rest later and the intervals accumulate toward a
    completed block. Returns the number of edges that gained NEW coverage."""
    when = when or _dt.date.today().isoformat()
    walked = store.setdefault("walked", {})
    gained = 0
    for eid, span in edge_spans.items():
        # Accept either one [lo, hi] pair or a list of disjoint pairs.
        spans = span
        if (isinstance(span, (list, tuple)) and len(span) == 2
                and all(isinstance(x, (int, float)) for x in span)):
            spans = [span]
        clean = [[max(0.0, min(1.0, float(lo))),
                  max(0.0, min(1.0, float(hi)))] for lo, hi in spans]
        rec = walked.get(eid)
        if rec is None:
            walked[eid] = {"intervals": _merge_intervals(clean),
                           "date": when, "note": note}
            gained += 1
        else:
            before = coverage_frac(rec)
            existing = rec.get("intervals")
            if existing is None:                  # legacy fully-walked record
                existing = [[0.0, 1.0]]
            rec["intervals"] = _merge_intervals(existing + clean)
            rec["last_date"] = when
            if coverage_frac(rec) > before + 1e-6:
                gained += 1
    return gained


def mark_route_walked(store: dict, G, route, when: str | None = None, note=""):
    """Mark every canonical passage physically traversed by a confirmed route.

    In a prepared graph, ``deadhead`` means repeat/connector overhead, not "was
    not walked".  An incomplete named passage used as a connector still earns
    coverage.  Unannotated legacy/test graphs preserve the former behavior.
    """
    spans = {}
    for u, v, key, dead in route:
        try:
            data = G[u][v][key]
        except (KeyError, TypeError):
            continue
        if data.get("coverage_id"):
            spans[edge_id(G, u, v, key)] = (0.0, 1.0)
        elif not dead:
            spans[edge_id(G, u, v, key)] = (0.0, 1.0)
    return record_spans(store, spans, when, note)


def mark_edges_walked(store: dict, edge_ids, when: str | None = None, note=""):
    """Mark whole edges fully walked (0->1). For when the walked extent is the
    entire edge; partial GPS coverage should use ``record_spans`` instead."""
    return record_spans(store, {eid: (0.0, 1.0) for eid in edge_ids}, when, note)


def walked_id_set(store: dict) -> set:
    """Every edge with ANY recorded coverage (complete or partial). For 'done'
    only, use completed_id_set(G, store)."""
    return set(store.get("walked", {}).keys())


def walk_summary(G: nx.MultiGraph, store: dict) -> dict:
    """
    Per-day rollup of covered street + grand totals, for "how far have I come"
    reporting (a day's distance, the running total, days out).

    Each edge is counted once, on the date it was FIRST recorded — so a day's
    distance is the NEW street it covered (re-walking a block adds nothing). Both
    all recorded edges (named ways + the connectors between them, ~ what you
    actually walked) and the NAMED subset (what counts toward 100%) are reported.

    Returns: ``days`` (list, oldest first) of
    ``{date, km, named_km, edges, notes: {note: km}}``, plus ``total_km``,
    ``total_named_km``, ``total_edges`` and ``n_days``.
    """
    walked = store.get("walked", {})
    # Canonical target -> representative length, requirement, combined evidence.
    meta = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if not is_required(data):
            continue
        eid = edge_id(G, u, v, key)
        rec = _combined_record(_records_for_edge(G, u, v, key, store))
        if rec is not None:
            meta[eid] = (data.get("length", 0.0), True, rec)

    # Preserve historical diary entries for explicitly recorded unnamed
    # connectors on unprepared/legacy graphs.  Canonical coverage percentages do
    # not use these; actual GPX distance is reported separately by import.
    for u, v, key, data in G.edges(keys=True, data=True):
        if is_required(data) or data.get("coverage_id"):
            continue
        eid = legacy_edge_id(u, v, key)
        rec = walked.get(eid)
        if rec is not None:
            meta[eid] = (data.get("length", 0.0), False, rec)

    days = {}
    for eid, (length, req, rec) in meta.items():
        length *= coverage_frac(rec)          # credit only the fraction walked
        d = days.setdefault(rec.get("date", "?"),
                            {"date": rec.get("date", "?"), "m": 0.0,
                             "named_m": 0.0, "edges": 0, "notes_m": {}})
        d["m"] += length
        d["named_m"] += length if req else 0.0
        d["edges"] += 1
        note = rec.get("note", "")
        if note:
            d["notes_m"][note] = d["notes_m"].get(note, 0.0) + length

    day_list = []
    for d in (days[k] for k in sorted(days)):
        day_list.append({
            "date": d["date"],
            "km": d["m"] / 1000.0,
            "named_km": d["named_m"] / 1000.0,
            "edges": d["edges"],
            "notes": {n: m / 1000.0 for n, m in d["notes_m"].items()},
        })
    return {
        "days": day_list,
        "total_km": sum(d["km"] for d in day_list),
        "total_named_km": sum(d["named_km"] for d in day_list),
        "total_edges": sum(d["edges"] for d in day_list),
        "n_days": len(day_list),
    }


def remaining_subgraph(G: nx.MultiGraph, store: dict,
                       required_only: bool = False) -> nx.MultiGraph:
    """
    A MultiGraph of edges you have NOT yet walked. With ``required_only`` it
    keeps just the unwalked NAMED ways — the coverage target — which is what the
    routers chew on (they use the full graph for connectors separately).
    """
    done = completed_id_set(G, store)         # partially-walked blocks still remain
    R = nx.MultiGraph()
    for u, v, key, data in G.edges(keys=True, data=True):
        if required_only and not is_required(data):
            continue
        if edge_id(G, u, v, key) not in done:
            R.add_node(u, **G.nodes[u])
            R.add_node(v, **G.nodes[v])
            R.add_edge(u, v, key=key, **data)
    return R


def stats(G: nx.MultiGraph, store: dict) -> dict:
    """Progress over the must-walk set (named ways only); 100% = every named
    way walked. Unnamed connectors are ignored here."""
    total_m = covered_m = complete_m = 0.0
    total_e = complete_e = partial_e = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        if not is_required(data):
            continue
        L = data.get("length", 0.0)
        total_m += L
        total_e += 1
        rec = _combined_record(_records_for_edge(G, u, v, key, store))
        if rec is None:
            continue
        f = coverage_frac(rec)
        covered_m += L * f
        if is_complete(rec, L):
            complete_m += L
            complete_e += 1
        elif f > 0:
            partial_e += 1
    return {
        "total_km": total_m / 1000.0,
        # Headline completeness is block-complete distance.  Observed partial
        # evidence remains visible separately but cannot produce a misleading
        # sub-100% "nothing left" state.
        "walked_km": complete_m / 1000.0,
        "covered_km": covered_m / 1000.0,
        "complete_km": complete_m / 1000.0,
        "remaining_km": (total_m - complete_m) / 1000.0,
        "total_edges": total_e,
        "walked_edges": complete_e,        # "segments" = fully-done blocks
        "complete_edges": complete_e,
        "partial_edges": partial_e,
        "pct_done": (complete_m / total_m * 100.0) if total_m else 0.0,
        "observed_pct": (covered_m / total_m * 100.0) if total_m else 0.0,
    }
