"""
render.py — draw the walked-vs-remaining network as a nice dark JPEG.

By default it shows the WHOLE city: every street faint, all walked street in one
highlight colour (the running total), over the full graph extent.

Privacy: ``redact`` zones (lat, lon, radius_m) demote any walked segment within
the radius back to a plain unwalked-looking street. We demote rather than erase
on purpose — a blank hole would advertise the spot, whereas a redacted block is
indistinguishable from any street you simply haven't walked yet.

`build_layers` is pure geometry (no plotting) and is unit-tested directly;
matplotlib/Pillow are imported lazily inside `render_coverage`, so importing this
module — and the `amble` package — never needs them. Install them only to render:
``pip install matplotlib pillow``.
"""
from __future__ import annotations

import math
import os

from . import progress as prog

BG = "#0b0f14"
STREET = "#46526a"          # brighter than the background so the grid reads clearly
INK = "#e6edf3"
MUTED = "#8b98a9"
WALKED = "#ef4444"          # single highlight colour for the running total
PARTIAL = "#f25c3c"         # in-progress: walked part-way (a warm red-orange, reads
                            # close to done so a corridor still looks continuous)

# a calm, high-contrast cycle, used only in per-walk mode
PALETTE = [
    "#38bdf8", "#fbbf24", "#f472b6", "#34d399", "#a78bfa",
    "#fb7185", "#facc15", "#22d3ee", "#4ade80", "#e879f9",
]


def _edge_xy(G, u, v, data):
    geom = data.get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        return list(zip(xs, ys))                 # (lon, lat) pairs
    return [(float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
            (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))]


def _pt_seg_dist_m(px, py, ax, ay, bx, by):
    """Distance from point P to segment AB (all in local metres)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _seg_in_zone(seg, zones):
    """True if any part of ``seg`` (a list of (lon, lat)) lies within a redact
    zone — exact point-to-segment distance, so an edge that merely PASSES near a
    zone (endpoints outside) is still caught."""
    for zlat, zlon, r in zones:
        cl = math.cos(math.radians(zlat))
        proj = [((lon - zlon) * 111320.0 * cl, (lat - zlat) * 110540.0)
                for lon, lat in seg]
        if len(proj) == 1:
            if math.hypot(*proj[0]) <= r:
                return True
            continue
        for (ax, ay), (bx, by) in zip(proj, proj[1:]):
            if _pt_seg_dist_m(0.0, 0.0, ax, ay, bx, by) <= r:
                return True
    return False


def build_layers(G, store, redact=None):
    """
    Pure geometry split for the map. Returns ``(base_segs, partial_segs, layers, bbox)``:

      base_segs   : coordinate paths for every UNwalked edge (plus any walked edge
                    inside a ``redact`` zone, demoted to look unwalked)
      partial_segs: blocks walked PART-WAY (recorded but not yet complete) — drawn
                    in the in-progress colour, between base and done
      layers      : per-walk ``{note, color, label, km, segs}`` of COMPLETED blocks,
                    in walk-date order
      bbox        : ``(minx, maxx, miny, maxy)`` over the SHOWN walked edges, or None

    ``redact`` is a list of ``(lat, lon, radius_m)`` zones to hide. Colour is
    assigned by walk order so adding a walk never recolours the earlier ones.
    """
    walked = store.get("walked", {})
    zones = redact or []
    order = _note_order(store)
    color_of = {note: PALETTE[i % len(PALETTE)] for i, (note, _d) in enumerate(order)}
    layers = {note: {"note": note, "color": color_of[note],
                     "label": (f"{note}  ({date})" if date else note),
                     "m": 0.0, "segs": []}
              for note, date in order}

    base_segs = []
    partial_segs = []                    # in-progress blocks (walked, not yet done)
    bb = [180.0, -180.0, 90.0, -90.0]
    have_walked = False
    for u, v, k, d in G.edges(keys=True, data=True):
        if G.graph.get("amble_model") == "canonical-passages-v1" and not d.get("coverage_required"):
            continue
        seg = _edge_xy(G, u, v, d)
        rec = prog._combined_record(prog._records_for_edge(G, u, v, k, store))
        if rec is None or _seg_in_zone(seg, zones):
            base_segs.append(seg)        # never walked, or redacted near home
            continue
        if not prog.is_complete(rec, d.get("length", 0.0)):
            if prog.coverage_frac(rec) >= 0.05:     # meaningfully started
                partial_segs.append(seg)
                have_walked = True
                for x, y in seg:
                    bb[0], bb[1] = min(bb[0], x), max(bb[1], x)
                    bb[2], bb[3] = min(bb[2], y), max(bb[3], y)
            else:
                base_segs.append(seg)    # negligible touch — leave it unwalked-looking
            continue
        have_walked = True
        note = rec.get("note", "") or "(unlabelled)"
        lyr = layers.setdefault(note, {"note": note,
                                       "color": PALETTE[len(layers) % len(PALETTE)],
                                       "label": note, "m": 0.0, "segs": []})
        lyr["segs"].append(seg)
        lyr["m"] += d.get("length", 0.0)
        for x, y in seg:
            bb[0], bb[1] = min(bb[0], x), max(bb[1], x)
            bb[2], bb[3] = min(bb[2], y), max(bb[3], y)

    layer_list = []
    for note, _date in order:
        lyr = layers[note]
        if not lyr["segs"]:
            continue                              # fully redacted walk: drop it
        layer_list.append({"note": note, "color": lyr["color"], "label": lyr["label"],
                           "km": lyr["m"] / 1000.0, "segs": lyr["segs"]})
    bbox = tuple(bb) if have_walked else None
    return base_segs, partial_segs, layer_list, bbox


def _note_order(store):
    """Notes ordered by first walk date (then name), each paired with its date."""
    first = {}
    for rec in store.get("walked", {}).values():
        note = rec.get("note", "") or "(unlabelled)"
        date = rec.get("date", "")
        if note not in first or date < first[note]:
            first[note] = date
    return sorted(first.items(), key=lambda kv: (kv[1], kv[0]))


def _extent(base_segs, layers, trim=0.0015):
    """
    Robust (minx, maxx, miny, maxy) over every drawn segment, for sizing the
    frame. Uses a small percentile TRIM instead of raw min/max so a few outlier
    edges — a lone bridge approach or pier shooting off to the edge — don't
    inflate the bounding box and leave a big blank band on one side.
    """
    xs, ys = [], []
    for seg in base_segs:
        for x, y in seg:
            xs.append(x)
            ys.append(y)
    for lyr in layers:
        for seg in lyr["segs"]:
            for x, y in seg:
                xs.append(x)
                ys.append(y)
    if not xs:
        return -122.5, -122.4, 37.7, 37.8
    xs.sort()
    ys.sort()
    n = len(xs)
    lo = int(n * trim)
    hi = min(n - 1, int(n * (1.0 - trim)))
    return xs[lo], xs[hi], ys[lo], ys[hi]


def render_coverage(G, store, out_path, title="", date_label="",
                    focus=False, mode="total", redact=None):
    """
    Render the coverage map to ``out_path`` (JPEG). Requires matplotlib + Pillow.

    mode="total"  : all walked street in one colour (the running total) — default.
    mode="by_walk": colour each walk separately, with a per-walk legend.
    focus=False   : show the whole city (default); True crops to the walked area.
    redact        : list of (lat, lon, radius_m) zones to hide (e.g. home).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    base_segs, partial_segs, layers, bbox = build_layers(G, store, redact=redact)

    # extent of what we'll draw, with one small EVEN border (no letterboxing:
    # the figure is sized to the map's aspect, so the only blank is this border)
    if focus and bbox:
        minx, maxx, miny, maxy = bbox
    else:
        minx, maxx, miny, maxy = _extent(base_segs, layers)
    lat0 = (miny + maxy) / 2.0
    spanx, spany = (maxx - minx) or 1e-6, (maxy - miny) or 1e-6
    pad = 0.035
    pad_bottom = pad + (0.045 if date_label else 0.0)   # clear band for the date
    minx, maxx = minx - spanx * pad, maxx + spanx * pad
    miny, maxy = miny - spany * pad_bottom, maxy + spany * pad

    cos_lat = math.cos(math.radians(lat0))
    disp_w, disp_h = (maxx - minx) * cos_lat, (maxy - miny)
    base_in = 16.0
    figw, figh = ((base_in, base_in * disp_h / disp_w) if disp_w >= disp_h
                  else (base_in * disp_w / disp_h, base_in))

    fig, ax = plt.subplots(figsize=(figw, figh), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.add_collection(LineCollection(base_segs, colors=STREET, linewidths=0.55, zorder=1))
    # in-progress blocks (walked part-way, not yet done) sit between base and done
    if partial_segs:
        ax.add_collection(LineCollection(partial_segs, colors=PARTIAL, linewidths=1.6,
                                         capstyle="round", zorder=2))

    handles = []
    if mode == "total":
        segs = [s for lyr in layers for s in lyr["segs"]]
        ax.add_collection(LineCollection(segs, colors=WALKED, linewidths=1.8,
                                         capstyle="round", zorder=3))
    else:
        for lyr in layers:
            ax.add_collection(LineCollection(lyr["segs"], colors=lyr["color"],
                                             linewidths=2.2, capstyle="round",
                                             zorder=3, alpha=0.97))
            handles.append(Line2D([0], [0], color=lyr["color"], lw=3.2,
                                  label=f"{lyr['label']} — {lyr['km']:.1f} km"))

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1.0 / cos_lat)
    ax.axis("off")

    # title and date overlaid INSIDE the frame, so they don't add any margin
    if title:
        ax.text(0.5, 0.982, title, transform=ax.transAxes, ha="center", va="top",
                color=INK, fontsize=30, fontweight="bold")
    if date_label:
        ax.text(0.988, 0.02, date_label, transform=ax.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=18)
    if handles:
        leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                        fontsize=15, labelcolor=INK)
        leg.get_frame().set_facecolor("#11161d")
        leg.get_frame().set_edgecolor(STREET)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=BG, pil_kwargs={"quality": 92})
    plt.close(fig)
    return out_path
