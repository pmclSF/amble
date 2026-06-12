"""
trace.py — map-match a RECORDED GPS track onto the graph and mark it walked.

`plan` produces a route you follow; `done` records that planned route. But a walk
you actually recorded on your phone (a GPX track log) is a noisy list of fixes,
not a graph route — so it needs MATCHING before it can count toward coverage.

The match is deliberately simple and conservative:

  1. SNAP each GPS fix to its nearest graph node.
  2. CONNECT consecutive (distinct) snaps along the shortest path by length.
  3. CAP each connector: if the graph path between two snaps is far longer than
     the straight-line hop between the fixes, it's a GPS DROPOUT, not a walk —
     skip it rather than invent a detour. Coverage is never fabricated across a
     gap; the skipped count is reported so silence never reads as success.

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


def _make_snap(G):
    """A fast nearest-node closure. Uses numpy if available (one vectorised
    pass over all nodes per fix); otherwise falls back to the pure-Python
    network.nearest_node so the module still works with networkx alone."""
    try:
        import numpy as np
    except ImportError:
        from . import network as net
        return lambda lat, lon: net.nearest_node(G, lat, lon)

    ids, ys, xs = [], [], []
    for n, d in G.nodes(data=True):
        try:
            y, x = float(d["y"]), float(d["x"])
        except (KeyError, ValueError, TypeError):
            continue
        ids.append(n)
        ys.append(y)
        xs.append(x)
    # a 1-D object array: np.array(list_of_tuples) would build a 2-D (N,2) array
    # and break node-id indexing when nodes are coordinate tuples.
    ids_arr = np.empty(len(ids), dtype=object)
    ids_arr[:] = ids
    ys_arr = np.array(ys)
    xs_arr = np.array(xs)

    def snap(lat, lon):
        cos_lat = math.cos(math.radians(lat))
        d2 = ((xs_arr - lon) * cos_lat) ** 2 + (ys_arr - lat) ** 2
        return ids_arr[int(d2.argmin())]

    return snap


def match_trace(G, points, gap_base_m: float = 150.0, gap_factor: float = 8.0):
    """
    Map-match a recorded GPS trace (list of (lat, lon)) to graph edges.

    A connector between two consecutive snapped nodes is kept only if its graph
    path length is within ``max(gap_base_m, gap_factor * straight_line_gap)`` —
    otherwise it's treated as a GPS dropout and skipped (never fabricates the
    streets in between). Returns a diagnostics dict:

        edge_ids   : set of progress.edge_id for every matched edge
        edge_meta  : edge_id -> (length_m, is_required)   (each edge once)
        n_points   : raw GPS fixes
        n_snapped  : distinct snapped nodes after de-duping consecutive repeats
        n_skipped  : connectors dropped as dropouts (path > cap, or no path)
        matched_m  : total length of matched edges
        named_m    : matched length over NAMED (coverage-counting) ways
    """
    snap = _make_snap(G)

    # snap every fix, collapse consecutive fixes that hit the same node
    snapped = []  # (node, lat, lon)
    for lat, lon in points:
        n = snap(lat, lon)
        if not snapped or snapped[-1][0] != n:
            snapped.append((n, lat, lon))

    edge_meta = {}   # eid -> (length, is_required), counted once even if re-walked
    skipped = 0
    for (a, la, lo), (b, lb, lob) in zip(snapped, snapped[1:]):
        cap = max(gap_base_m, gap_factor * _geo_m(la, lo, lb, lob))
        try:
            plen, nodes = nx.single_source_dijkstra(G, a, target=b, weight="length")
        except nx.NetworkXNoPath:
            skipped += 1
            continue
        if plen > cap:
            skipped += 1  # GPS dropout — don't claim a detour we didn't walk
            continue
        for x, y in zip(nodes, nodes[1:]):
            key = min(G[x][y], key=lambda k: G[x][y][k].get("length", float("inf")))
            d = G[x][y][key]
            edge_meta[prog.edge_id(G, x, y, key)] = (
                d.get("length", 0.0), prog.is_required(d))

    matched_m = sum(L for L, _ in edge_meta.values())
    named_m = sum(L for L, req in edge_meta.values() if req)
    return {
        "edge_ids": set(edge_meta),
        "edge_meta": edge_meta,
        "n_points": len(points),
        "n_snapped": len(snapped),
        "n_skipped": skipped,
        "matched_m": matched_m,
        "named_m": named_m,
    }
