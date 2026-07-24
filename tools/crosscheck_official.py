"""Cross-check the canonical inventory against an official street-centerline
dataset, per name. The faithfulness check behind the amble-coverage-v2 filter.

For San Francisco the reference is DataSF's CNN centerlines ("Streets - Active
and Retired", dataset 3psu-pn9h):

    curl -o cnn.csv "https://data.sfgov.org/api/views/3psu-pn9h/rows.csv?accessType=DOWNLOAD"
    .venv/bin/python tools/crosscheck_official.py \
        --cache data/sf.grid.graphml --cnn cnn.csv --out report.json

Any city works with a CSV of (street, st_type, layer, active, line-WKT) rows —
adapt the column mapping below. Interpretation notes:
  * official centerline sets often digitize BOTH roadbeds of a divided street,
    so official per-name km can run ~2x; the >=45% match bar absorbs that;
  * a "missing" name may be absent from OSM, excluded by policy (private,
    ramps), or a spelling variant — grep the raw graphml for the name to
    distinguish an OSM gap from a pipeline loss before acting on it;
  * the matched percentage is a NAME-LEVEL measure over the layers you pass,
    not a guarantee that every block of a matched name is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ST_TYPE = {
    "ST": "street", "AVE": "avenue", "BLVD": "boulevard", "DR": "drive",
    "CT": "court", "LN": "lane", "PL": "place", "RD": "road",
    "TER": "terrace", "WAY": "way", "ALY": "alley", "CIR": "circle",
    "HWY": "highway", "PLZ": "plaza", "SQ": "square", "STPS": "steps",
    "STWY": "stairway", "WALK": "walk", "PATH": "path", "ROW": "row",
    "EXPY": "expressway", "TUNL": "tunnel", "PARK": "park", "LOOP": "loop",
    "PSGE": "passage", "XING": "crossing", "PKWY": "parkway",
}
WORD_FIX = {"saint": "st", "mount": "mt"}


def norm(s: str) -> str:
    s = s.casefold().replace("’", "'")
    s = re.sub(r"[.,']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b0(\d)(st|nd|rd|th)\b", r"\1\2", s)   # 01ST -> 1st
    return " ".join(WORD_FIX.get(w, w) for w in s.split())


def wkt_length_m(wkt: str) -> float:
    pts = [(float(a), float(b)) for a, b in
           re.findall(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", wkt or "")]
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        cl = math.cos(math.radians((y1 + y2) / 2))
        total += math.hypot((x2 - x1) * 111320.0 * cl, (y2 - y1) * 110540.0)
    return total


def official_names(cnn_csv: str, layers: set[str]):
    """name -> official metres over the requested centerline layers."""
    csv.field_size_limit(10_000_000)
    out = defaultdict(float)
    with open(cnn_csv, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("active", "")).strip().lower() != "true":
                continue
            if (row.get("layer") or "").strip().upper() not in layers:
                continue
            stt = (row.get("st_type") or "").strip().upper()
            name = norm(f"{row.get('street', '').strip()} "
                        f"{ST_TYPE.get(stt, stt.lower())}")
            # vehicular-only structures/ramps and placeholders are not walking
            # obligations; the street name itself carries those blocks
            if not name or stt == "TUNL" or \
                    re.match(r"^(unnamed \d+|i-\d+\b.*)$", name) or \
                    name.endswith((" on ramp", " off ramp")):
                continue
            out[name] += wkt_length_m(row.get("line", ""))
    return out


def inventory_names(cache: str):
    """name -> canonical required metres from a prepared amble graph."""
    from amble import network, passages
    G = network.load_or_download("", cache)
    H, _ = network.prepare_graph(G)
    inv = defaultdict(float)
    for *_uvk, d in H.edges(keys=True, data=True):
        if d.get("coverage_required"):
            inv[norm(passages.display_name(d))] += \
                float(d.get("length", 0.0) or 0.0)
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cnn", required=True)
    ap.add_argument("--out", default="crosscheck_report.json")
    ap.add_argument("--layers", default="STREETS",
                    help="comma-separated official centerline layers to check")
    ap.add_argument("--match-frac", type=float, default=0.45,
                    help="inventory/official ratio counted as matched "
                         "(<1 absorbs official double-digitized roadbeds)")
    a = ap.parse_args()

    official = official_names(a.cnn, {s.strip().upper()
                                      for s in a.layers.split(",")})
    inv = inventory_names(a.cache)
    squash = defaultdict(float)
    for n, m in inv.items():
        squash[n.replace(" ", "")] += m

    missing, undercount = [], []
    total = matched = 0.0
    for name, off_m in official.items():
        if off_m < 30:
            continue
        total += off_m
        inv_m = inv.get(name, 0.0) or squash.get(name.replace(" ", ""), 0.0) \
            or inv.get(name + " street", 0.0)
        if inv_m >= a.match_frac * off_m:
            matched += off_m
            continue
        row = {"name": name, "official_km": round(off_m / 1000, 2),
               "inventory_km": round(inv_m / 1000, 2)}
        (missing if inv_m < 30 else undercount).append(row)
    missing.sort(key=lambda r: -r["official_km"])
    undercount.sort(key=lambda r: -(r["official_km"] - r["inventory_km"]))

    report = {
        "official_km": round(total / 1000, 1),
        "matched_km": round(matched / 1000, 1),
        "matched_pct": round(100 * matched / total, 1) if total else 0.0,
        "missing": missing, "undercount": undercount,
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"official {report['official_km']} km; matched "
          f"{report['matched_km']} km ({report['matched_pct']}%); "
          f"{len(missing)} names missing, {len(undercount)} undercounted; "
          f"details -> {a.out}")


if __name__ == "__main__":
    main()
