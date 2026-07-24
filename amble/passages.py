"""Canonical public-passage inventory.

OSM's walking graph describes *surfaces*: a roadway, both sidewalks, a cycle
track, and two divided carriageways may all describe the same human goal: walk
one named block.  This module annotates the detailed routing graph with a
separate coverage inventory.

Exactly one edge in each equivalence cluster is marked ``coverage_required``.
Every equivalent edge receives the same ``coverage_id`` so observations on a
parallel representation credit the same passage.  Distinct passage classes
(notably stairs, alleys, named trails, and piers) are never merged into roads
merely because they are close.

The clustering is deliberately conservative.  It merges only same-name,
same-block near-parallel representations with aligned endpoints.  Ambiguous
cases remain separate rather than creating a false 100% result.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

from .equivalents import canonical_name


EXCLUDED_SERVICES = {
    "driveway", "parking_aisle", "drive-through", "drive_through",
}
EXCLUDED_ACCESS = {"private", "no", "customers", "emergency", "permit"}
FOOT_ALLOWED = {"yes", "designated", "permissive", "official"}
EXCLUDED_HIGHWAYS = {"bus_stop", "motorway", "motorway_link", "raceway",
                     "busway", "bus_guideway"}
# A roadway a pedestrian cannot legally walk ON; the walk happens on its
# parallel sidewalk/sidepath. The NAME is still owed (it stays a coverage
# target) — these tags redirect *evidence* to aligned parallel surfaces.
FOOT_RESTRICTED = {"no", "use_sidepath"}
PARKING_NAME_WORDS = (
    "parking entrance", "parking ramp", "parking garage", "parking structure",
    "loading dock", "garage entrance",
)


def _values(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    return {str(value).strip().lower()} if str(value).strip() else set()


def display_name(data) -> str:
    name = data.get("name")
    if isinstance(name, list):
        name = name[0] if name else ""
    return str(name or "").strip()


def normalized_name(data) -> str:
    """Stable comparison name, including configured corridor equivalences."""
    name = canonical_name(display_name(data))
    name = re.sub(r"\s+", " ", name).strip().casefold()
    return name


def passage_class(data) -> str:
    """Human passage class used to prevent unsafe geometric merging."""
    highway = _values(data.get("highway"))
    service = _values(data.get("service"))
    man_made = _values(data.get("man_made"))
    if "pier" in man_made:
        return "pier"
    if "steps" in highway:
        return "staircase"
    if "alley" in service:
        return "alley"
    if highway & {"path", "footway", "pedestrian", "bridleway", "track",
                  "cycleway"}:
        return "trail"
    return "street"


def foot_restricted(data) -> bool:
    """The roadway itself is not legally walkable (foot=no/use_sidepath or
    sidewalks mapped as separate ways) — evidence must come from a parallel
    surface, but the named passage remains a coverage target."""
    foot = _values(data.get("foot"))
    sidewalk = _values(data.get("sidewalk"))
    return bool(foot & FOOT_RESTRICTED) or "separate" in sidewalk


def exclusion_reason(data) -> str | None:
    """Why an edge is outside the public-passage/routing model, if known.

    ``foot=yes/designated/permissive`` overrides a generic access restriction.
    This matters for ways tagged ``access=no, foot=yes``.  Older caches that did
    not retain ``foot`` are conservatively excluded and surfaced by the audit.
    """
    highway = _values(data.get("highway"))
    service = _values(data.get("service"))
    access = _values(data.get("access"))
    foot = _values(data.get("foot"))
    amenity = _values(data.get("amenity"))
    parking = _values(data.get("parking"))
    indoor = _values(data.get("indoor"))
    tunnel = _values(data.get("tunnel"))
    railway = _values(data.get("railway"))
    public_transport = _values(data.get("public_transport"))
    area = _values(data.get("area"))
    man_made = _values(data.get("man_made"))
    name = display_name(data).casefold()

    if service & EXCLUDED_SERVICES:
        return "parking_or_driveway"
    if highway & EXCLUDED_HIGHWAYS:
        return "non_pedestrian_highway"
    if "elevator" in highway:
        return "mechanical_vertical_transport"
    if "platform" in railway or "platform" in public_transport or "platform" in highway:
        return "transit_platform"
    if (access & EXCLUDED_ACCESS) and not (foot & FOOT_ALLOWED):
        return "restricted_access"
    if "parking" in amenity or parking:
        return "parking_facility"
    if indoor & {"yes", "room", "corridor"}:
        return "indoor"
    if "building_passage" in tunnel:
        return "building_passage"
    if area & {"yes", "1", "true"} and "pier" not in man_made:
        return "non_linear_area"
    if any(word in name for word in PARKING_NAME_WORDS):
        return "parking_or_loading"
    return None


def is_public_routing_edge(data) -> bool:
    return exclusion_reason(data) is None


def is_eligible_passage(data) -> bool:
    return bool(display_name(data)) and exclusion_reason(data) is None


def _coords(G, u, v, data):
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        try:
            return [(float(x), float(y)) for x, y in geom.coords]
        except (TypeError, ValueError):
            pass
    try:
        return [(float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))]
    except (KeyError, TypeError, ValueError):
        return []


def _xy(lon, lat, lon0, lat0):
    return ((lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
            (lat - lat0) * 110540.0)


def _features(G, edge):
    u, v, _k, data = edge
    coords = _coords(G, u, v, data)
    if len(coords) < 2:
        return None
    lon0 = sum(p[0] for p in coords) / len(coords)
    lat0 = sum(p[1] for p in coords) / len(coords)
    ax, ay = _xy(coords[0][0], coords[0][1], lon0, lat0)
    bx, by = _xy(coords[-1][0], coords[-1][1], lon0, lat0)
    dx, dy = bx - ax, by - ay
    chord = math.hypot(dx, dy)
    if chord < 0.5:
        return None
    length = float(data.get("length", chord) or chord)
    return {
        "mid": ((ax + bx) / 2.0, (ay + by) / 2.0),
        "ends": ((ax, ay), (bx, by)),
        "dir": (dx / chord, dy / chord),
        "length": max(length, 0.1),
        "origin": (lon0, lat0),
    }


def _to_common(feat, origin):
    """Reproject a local feature's endpoints into ``origin`` local metres."""
    lon0, lat0 = origin
    flon, flat = feat["origin"]
    ox, oy = _xy(flon, flat, lon0, lat0)
    ends = tuple((x + ox, y + oy) for x, y in feat["ends"])
    return {
        **feat,
        "ends": ends,
        "mid": ((ends[0][0] + ends[1][0]) / 2.0,
                (ends[0][1] + ends[1][1]) / 2.0),
    }


def _same_block(a, b, class_a, class_b) -> bool:
    # Stairs, alleys, and piers are distinct passage types even beside a road.
    if class_a != class_b:
        # A named sidewalk/footway carrying the *same street name* can be a
        # surface representation of that street, but merge it only under tight
        # block-level alignment.
        if {class_a, class_b} != {"street", "trail"}:
            return False
        tight = True
    else:
        tight = False

    b = _to_common(b, a["origin"])
    dot = abs(a["dir"][0] * b["dir"][0] + a["dir"][1] * b["dir"][1])
    if dot < math.cos(math.radians(22.0)):
        return False
    ratio = a["length"] / b["length"]
    if not 0.60 <= ratio <= 1.67:
        return False

    # Align endpoint direction before comparing both ends.  Requiring *both*
    # endpoint pairs to be close avoids merging adjacent collinear blocks.
    ae = a["ends"]
    be0, be1 = b["ends"]
    direct = math.hypot(ae[0][0] - be0[0], ae[0][1] - be0[1]) + \
             math.hypot(ae[1][0] - be1[0], ae[1][1] - be1[1])
    reverse = math.hypot(ae[0][0] - be1[0], ae[0][1] - be1[1]) + \
              math.hypot(ae[1][0] - be0[0], ae[1][1] - be0[1])
    be = (be0, be1) if direct <= reverse else (be1, be0)
    end_dist = [math.hypot(ae[i][0] - be[i][0], ae[i][1] - be[i][1])
                for i in (0, 1)]
    # Same-name street/street pairs get a wider bar: divided boulevards
    # (Sunset, Park Presidio, Geary's expressway blocks) put 35-45 m of median
    # between carriageways that are ONE obligation. Adjacent collinear blocks
    # can't sneak in — their endpoint distance is a full block length.
    if tight:
        limit = 24.0
    elif class_a == "street":
        limit = 45.0
    else:
        limit = 32.0
    if max(end_dist) > limit:
        return False

    rx = b["mid"][0] - a["mid"][0]
    ry = b["mid"][1] - a["mid"][1]
    along = abs(rx * a["dir"][0] + ry * a["dir"][1])
    perp = abs(rx * -a["dir"][1] + ry * a["dir"][0])
    # The along floor must stay BELOW short-block lengths: two collinear
    # same-name pieces of ~10 m each have along = (L1+L2)/2 ~ 10 m, and a
    # 12 m floor merged them SERIALLY into one target (walking one half then
    # completed the never-walked other half).
    return along <= max(6.0, 0.20 * min(a["length"], b["length"])) and perp <= limit


def _representative_score(edge):
    _u, _v, _k, data = edge
    cls = passage_class(data)
    highway = _values(data.get("highway"))
    # Prefer a street centerline to its named sidewalk shadow.  For genuinely
    # distinct trails/stairs/classes, clustering never compares them here.
    road = cls == "street" and not highway & {"footway", "path", "pedestrian"}
    return (0 if road else 1,
            0 if not _values(data.get("service")) else 1,
            float(data.get("length", 0.0) or 0.0))


def _passage_id(G, edge, cls):
    u, v, _k, data = edge
    name = normalized_name(data)
    # Rounded endpoint geometry is independent of OSM node IDs and edge keys,
    # while remaining specific enough to distinguish adjacent blocks.
    pts = []
    for n in (u, v):
        try:
            pts.append((round(float(G.nodes[n]["y"]), 5),
                        round(float(G.nodes[n]["x"]), 5)))
        except (KeyError, TypeError, ValueError):
            pts.append((str(n), ""))
    pts.sort(key=str)
    raw = f"v1|{name}|{cls}|{pts[0]}|{pts[1]}"
    return "passage:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _collision_suffix(G, edge) -> str:
    """Disambiguate distinct same-name routes sharing the same endpoints.

    Curving streets can form two different arcs between the same intersections.
    Endpoint identity alone intentionally groups normal parallel surfaces, but
    those arcs fail the geometric equivalence test and must remain two targets.
    """
    u, v, _k, data = edge
    coords = _coords(G, u, v, data)
    forward = [(round(x, 6), round(y, 6)) for x, y in coords]
    reverse = list(reversed(forward))
    geom = min(forward, reverse) if forward else []
    raw = repr((geom, data.get("osmid"), round(float(data.get("length", 0) or 0), 1)))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def legacy_edge_id(u, v, key) -> str:
    a, b = sorted((u, v), key=str)
    return f"{a}-{b}-{key}"


def annotate_passages(G):
    """Annotate a detailed routing graph with canonical coverage targets.

    Returns ``(G, audit)``.  The graph is modified in place.  ``audit`` contains
    aggregate counts and exclusion reasons suitable for CLI diagnostics.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        data.pop("coverage_id", None)
        data.pop("coverage_required", None)
        data.pop("coverage_alias_ids", None)
        data.pop("coverage_name", None)
        data.pop("coverage_class", None)
        data["coverage_exclusion"] = exclusion_reason(data) or ""

    candidates = []
    exclusions = defaultdict(lambda: [0, 0.0])
    pier_area_n, pier_area_m = 0, 0.0
    for u, v, k, data in G.edges(keys=True, data=True):
        reason = exclusion_reason(data)
        if reason:
            exclusions[reason][0] += 1
            exclusions[reason][1] += float(data.get("length", 0.0) or 0.0)
        if is_eligible_passage(data):
            # A named pier mapped as a POLYGON (Pier 39's shed footprint)
            # contributes ring edges, not a walking obligation: keeping them
            # required manufactured kilometres of largely unwalkable
            # perimeter. Rings stay routable but are never targets. Linear
            # named piers remain full obligations.
            if passage_class(data) == "pier" and \
                    _values(data.get("area")) & {"yes", "1", "true"}:
                pier_area_n += 1
                pier_area_m += float(data.get("length", 0.0) or 0.0)
                data["coverage_required"] = False
                data["coverage_exclusion"] = "pier_area_ring"
                continue
            candidates.append((u, v, k, data))

    # Same canonical name is a necessary condition.  Spatial comparison within
    # each name keeps city-scale clustering tractable and avoids cross-name loss.
    groups = defaultdict(list)
    for edge in candidates:
        groups[normalized_name(edge[3])].append(edge)

    cluster_count = alias_count = id_collisions = 0
    canonical_m = 0.0
    fr_targets, fr_m = 0, 0.0
    near_miss_pairs, near_miss_m = 0, 0.0
    used_ids = set()
    for _name, edges in groups.items():
        parent = list(range(len(edges)))
        feats = [_features(G, e) for e in edges]

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a

        for i in range(len(edges)):
            if feats[i] is None:
                continue
            for j in range(i + 1, len(edges)):
                if feats[j] is None:
                    continue
                if _same_block(feats[i], feats[j],
                               passage_class(edges[i][3]), passage_class(edges[j][3])):
                    union(i, j)

        clusters = defaultdict(list)
        for i in range(len(edges)):
            clusters[find(i)].append(i)
        street_rep_feats = []
        for idxs in clusters.values():
            members = [edges[i] for i in idxs]
            rep_i = min(idxs, key=lambda i: _representative_score(edges[i]))
            rep = edges[rep_i]
            cls = passage_class(rep[3])
            if cls == "street" and feats[rep_i] is not None:
                street_rep_feats.append(feats[rep_i])
            cid = _passage_id(G, rep, cls)
            if cid in used_ids:
                id_collisions += 1
                cid = f"{cid}:{_collision_suffix(G, rep)}"
                # Defensive fallback for exact duplicate geometries that were
                # conservatively left in separate clusters.
                salt = 2
                base = cid
                while cid in used_ids:
                    cid = f"{base}-{salt}"
                    salt += 1
            used_ids.add(cid)
            alias_ids = [legacy_edge_id(u, v, k) for u, v, k, _d in members]
            for edge in members:
                data = edge[3]
                data["coverage_id"] = cid
                data["coverage_required"] = edge is rep
                data["coverage_alias_ids"] = alias_ids
                data["coverage_name"] = canonical_name(display_name(rep[3]))
                data["coverage_class"] = cls
            cluster_count += 1
            alias_count += len(members) - 1
            canonical_m += float(rep[3].get("length", 0.0) or 0.0)
            if foot_restricted(rep[3]):
                fr_targets += 1
                fr_m += float(rep[3].get("length", 0.0) or 0.0)

        # Audit-only: same-name street reps that look like an UNMERGED divided
        # pair (just past the 45 m endpoint bar). Reported, never auto-merged —
        # ambiguity stays visible instead of silently manufacturing coverage.
        for i in range(len(street_rep_feats)):
            for j in range(i + 1, len(street_rep_feats)):
                a = street_rep_feats[i]
                b = _to_common(street_rep_feats[j], a["origin"])
                dot = abs(a["dir"][0] * b["dir"][0] + a["dir"][1] * b["dir"][1])
                if dot < math.cos(math.radians(22.0)):
                    continue
                ratio = a["length"] / b["length"]
                if not 0.5 <= ratio <= 2.0:
                    continue
                rx = b["mid"][0] - a["mid"][0]
                ry = b["mid"][1] - a["mid"][1]
                along = abs(rx * a["dir"][0] + ry * a["dir"][1])
                perp = abs(rx * -a["dir"][1] + ry * a["dir"][0])
                if 45.0 < perp <= 90.0 and \
                        along <= 0.35 * min(a["length"], b["length"]):
                    near_miss_pairs += 1
                    near_miss_m += min(a["length"], b["length"])

    # Demote intersection SHARDS: name-boundary simplification leaves sub-20 m
    # named street fragments inside intersection boxes (the stub of a side
    # street between the cross street's centerline and the corner). A walker
    # owes the BLOCK, not the paint inside the junction — and these shards sit
    # exactly where GPS is most ambiguous, so they pollute both the denominator
    # and corner evidence. A short street piece adjacent to a LONGER same-name
    # street block is absorbed into intersection ground: excluded entirely,
    # never a target, never an evidence surface. Standalone short passages
    # (real alleys, stair flights, paths) have no such neighbour and are kept.
    # Demotion works over ALL cluster members (not just elected reps): a shard
    # hanging off the non-rep carriageway of a merged divided block must still
    # see its long neighbour, a cluster is "short" only if its LONGEST member
    # is (rep election prefers the shortest member), and the shard must be
    # near-COLLINEAR with that neighbour — a perpendicular terminal leg of the
    # same street extends beyond the corner and is real ground, not paint.
    cid_members = defaultdict(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("coverage_id") and d.get("coverage_class") == "street":
            cid_members[d["coverage_id"]].append((u, v, k, d))
    cid_maxlen = {cid: max(float(d.get("length", 0.0) or 0.0)
                           for *_e, d in mem)
                  for cid, mem in cid_members.items()}
    node_idx = defaultdict(list)       # node -> (cid, name, member direction)
    member_feats = {}
    for cid, mem in cid_members.items():
        for u, v, k, d in mem:
            feat = _features(G, (u, v, k, d))
            if feat is None:
                continue
            member_feats[(u, v, k)] = feat
            info = (cid, normalized_name(d), feat["dir"])
            node_idx[u].append(info)
            node_idx[v].append(info)

    def _has_collinear_long_neighbour(cid, mem):
        for u, v, k, d in mem:
            feat = member_feats.get((u, v, k))
            if feat is None:
                continue
            nm = normalized_name(d)
            for node in (u, v):
                for ocid, onm, odir in node_idx[node]:
                    if ocid == cid or onm != nm:
                        continue
                    if cid_maxlen.get(ocid, 0.0) < 20.0:
                        continue
                    dot = abs(feat["dir"][0] * odir[0] +
                              feat["dir"][1] * odir[1])
                    if dot >= 0.7:
                        return True
        return False

    shard_targets, shard_m = 0, 0.0
    demoted = set()
    for cid, mem in cid_members.items():
        if cid_maxlen[cid] >= 20.0:
            continue
        if _has_collinear_long_neighbour(cid, mem):
            demoted.add(cid)
            shard_targets += 1
            shard_m += cid_maxlen[cid]        # physical extent removed
    if demoted:
        for _u, _v, _k, d in G.edges(keys=True, data=True):
            if d.get("coverage_id") in demoted:
                if d.get("coverage_required"):
                    cluster_count -= 1
                    canonical_m -= float(d.get("length", 0.0) or 0.0)
                    if foot_restricted(d):
                        fr_targets -= 1
                        fr_m -= float(d.get("length", 0.0) or 0.0)
                else:
                    alias_count -= 1
                d.pop("coverage_id", None)
                d.pop("coverage_alias_ids", None)
                d.pop("coverage_name", None)
                d.pop("coverage_class", None)
                # Keep the key, explicitly False: is_required treats presence
                # as authoritative; popping it would resurrect the shard via
                # the legacy named-way fallback.
                d["coverage_required"] = False
                d["coverage_exclusion"] = "intersection_shard"

    # Associate unnamed sidewalk surfaces with an aligned canonical street
    # block.  They remain non-required routing/observation edges, but GPS on
    # either sidewalk credits the same block.  Unnamed generic park paths are
    # intentionally not associated: they are connector-only and may cross near
    # unrelated roads.
    reps = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
            if d.get("coverage_required") and d.get("coverage_class") == "street"]
    cell = 0.00035
    grid = defaultdict(list)

    def geo_mid(edge):
        u, v, _k, _d = edge
        try:
            return ((float(G.nodes[u]["x"]) + float(G.nodes[v]["x"])) / 2.0,
                    (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"])) / 2.0)
        except (KeyError, TypeError, ValueError):
            return None

    rep_feats = {}
    for rep in reps:
        mid = geo_mid(rep)
        feat = _features(G, rep)
        if mid is not None and feat is not None:
            idx = len(rep_feats)
            rep_feats[idx] = (rep, feat)
            grid[(int(mid[0] // cell), int(mid[1] // cell))].append(idx)

    observation_aliases = 0
    for edge in list(G.edges(keys=True, data=True)):
        u, v, k, data = edge
        if data.get("coverage_id") or display_name(data):
            continue
        highway = _values(data.get("highway"))
        footway = _values(data.get("footway"))
        # Do not infer that every unnamed highway=footway is a sidewalk: an
        # unnamed park path can run parallel to a road and is not proof that the
        # road block was walked.  OSM's explicit footway=sidewalk is the safe
        # street-side observation tag.  An aligned cycleway is also another
        # carriage surface of the same corridor, not a separate named goal.
        # Exception: when the roadway itself is foot-PROHIBITED (foot=no /
        # use_sidepath), its aligned parallel path IS the walking surface of
        # that street, so generic unnamed footways/paths may evidence THAT rep
        # only. sidewalk=separate does NOT qualify: those streets have real
        # mapped sidewalks (which alias via footway=sidewalk) — accepting any
        # coincidental park path would credit the street from the park side.
        sidewalkish = "sidewalk" in footway or "cycleway" in highway
        pathish = bool(highway & {"footway", "path", "pedestrian"})
        if not (sidewalkish or pathish):
            continue
        mid = geo_mid(edge)
        feat = _features(G, edge)
        if mid is None or feat is None:
            continue
        ci, cj = int(mid[0] // cell), int(mid[1] // cell)
        matches = []
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for idx in grid.get((ci + di, cj + dj), ()):
                    rep, rfeat = rep_feats[idx]
                    if not sidewalkish and not (
                            _values(rep[3].get("foot")) & FOOT_RESTRICTED):
                        continue
                    if _same_block(feat, rfeat, "trail", "street"):
                        # Prefer the closest block midpoint if a corner offers
                        # more than one geometrically plausible association.
                        rf = _to_common(rfeat, feat["origin"])
                        dist = math.hypot(feat["mid"][0] - rf["mid"][0],
                                          feat["mid"][1] - rf["mid"][1])
                        matches.append((dist, rep))
        if not matches:
            continue
        _dist, rep = min(matches, key=lambda x: x[0])
        rd = rep[3]
        data["coverage_id"] = rd["coverage_id"]
        data["coverage_required"] = False
        data["coverage_name"] = rd["coverage_name"]
        data["coverage_class"] = rd["coverage_class"]
        alias = legacy_edge_id(u, v, k)
        aliases = rd.get("coverage_alias_ids") or []
        if alias not in aliases:
            aliases.append(alias)
        data["coverage_alias_ids"] = aliases
        observation_aliases += 1

    inventory_rows = sorted(
        f"{d.get('coverage_id')}:{round(float(d.get('length', 0.0) or 0.0), 1)}"
        for *_e, d in G.edges(data=True) if d.get("coverage_required"))
    fingerprint = hashlib.sha256("\n".join(inventory_rows).encode("utf-8")).hexdigest()
    if len({d.get("coverage_id") for *_e, d in G.edges(data=True)
            if d.get("coverage_required")}) != cluster_count:
        raise RuntimeError("canonical passage IDs are not unique")
    G.graph["amble_inventory_fingerprint"] = fingerprint
    return G, {
        "targets": cluster_count,
        "aliases": alias_count,
        "observation_aliases": observation_aliases,
        "id_collisions_resolved": id_collisions,
        "canonical_m": canonical_m,
        "foot_restricted_targets": {"targets": fr_targets, "m": fr_m},
        "divided_near_miss": {"pairs": near_miss_pairs, "m": near_miss_m},
        "intersection_shards": {"targets": shard_targets, "m": shard_m},
        "pier_area_rings_not_required": {"edges": pier_area_n,
                                         "m": pier_area_m},
        "excluded": {reason: {"edges": n, "m": metres}
                     for reason, (n, metres) in exclusions.items()},
        "fingerprint": fingerprint,
    }
