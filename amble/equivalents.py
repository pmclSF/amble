"""
equivalents.py — declare that several named ways are really ONE place.

The scope rule is "walk every NAMED way". But map data sometimes tags a single
physical corridor with several overlapping names, so "walk each named way" would
march you down the same strip two, three, four times. List those redundant names
together here: only the group's canonical name counts toward the goal, and the
rest become optional connectors — so the corridor is walked, and counted, once.

This is deliberately a hand-curated list, not automatic detection: geometry alone
can't tell a redundant twin from two genuinely distinct parallel streets (the two
carriageways of a real divided boulevard), and guessing wrong silently drops a
must-walk street. So you opt in, name by name.

Each group is ``{canonical, names}``. Edit the list for your own city. The
example below is San Francisco's former Great Highway — now mapped as a road, a
promenade, a trail, and a park path: four names, one walk.
"""

from __future__ import annotations

EQUIVALENT_WAYS = [
    {
        "canonical": "Great Highway",          # the name the merged corridor keeps
        "names": {
            "Great Highway",
            "Great Highway Promenade",
            "Lower Great Highway Trail",
            "Sunset Dunes",
        },
    },
]


def _canon_map(groups=None) -> dict:
    """name -> canonical name, for every name in any equivalence group."""
    groups = EQUIVALENT_WAYS if groups is None else groups
    m = {}
    for g in groups:
        for nm in g["names"]:
            m[nm] = g["canonical"]
    return m


def canonical_name(name, groups=None):
    """The corridor name for ``name`` if it's in a declared group, else name."""
    return _canon_map(groups).get(name, name)
