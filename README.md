# Amble

Amble tracks and plans a multi-year attempt to walk every named public passage
in a geographic zone while minimizing repeated distance.

The unit of completion is a **human passage**, not an OSM routing surface. A
named block counts once whether the GPS trace follows its roadway centerline,
left sidewalk, or right sidewalk. Separate carriageways, cycle tracks, and
sidewalk geometries must not turn one block into several obligations.

## Coverage contract

OpenStreetMap is the source of truth for names, geometry, access, and passage
type. Amble derives a versioned canonical inventory from its own OSM download
(`amble-coverage-v2`), NOT from a stock "walking network":

- The download keeps every non-motorway road and path **regardless of `foot`
  and `sidewalk` tags**, plus `man_made=pier` ways. A stock walk filter
  silently deletes roadway centerlines tagged `sidewalk=separate` (exactly the
  micromapped downtown corridors) and `foot=no` divided carriageways — named
  streets a walker still owes.
- A named roadway that is itself foot-restricted (`foot=no`,
  `foot=use_sidepath`, `sidewalk=separate`) **remains a target**; its aligned
  parallel sidewalks and sidepaths become observation aliases, so walking
  beside it is what completes it. The audit reports these separately.
- Simplification preserves name boundaries, so one edge never blends two
  differently named streets and distance is never credited to the wrong name.
- Components unreachable on foot from the mainland (e.g. Treasure Island) are
  kept: their named streets are part of the zone.

Included when named and publicly walkable:

- street blocks and divided-road blocks;
- public alleys;
- staircases;
- named trails and paths, including trails in parks;
- piers when OSM identifies them as named walkable ways.

Excluded from 100%:

- unnamed sidewalks, crossings, and connector paths;
- intersection shards: sub-20 m named street stubs adjacent to a longer
  same-name block (junction ground the block already owns — name-boundary
  splitting leaves these inside intersection boxes, exactly where GPS is
  most ambiguous);
- driveways and parking aisles;
- parking lots, garages, structures, and their ramps;
- private, customer-only, or gated apartment/building walks;
- indoor corridors, loading areas, and non-pedestrian highways.

Scope is also configurable: `OUT_OF_SCOPE_ZONES` in
[`amble/equivalents.py`](amble/equivalents.py) removes declared areas from the
network entirely (San Francisco ships with Alcatraz — real paths, ferry-only
access, not part of the goal). Removals are reported by `audit` as
`out_of_scope`.

Excluded ways are not completion targets. Public unnamed connectors can remain
in the routing graph when needed to reach a target. An unnamed sidewalk that is
tightly aligned with a named block is an **observation alias**: GPS fixes on it
credit the named block but never create another target.

Clustering is deliberately conservative. Amble merges only same-name,
same-block, near-parallel representations with aligned endpoints (up to 45 m
apart for same-name street pairs, wide enough for a divided boulevard's
median). Alleys, stairs, named trails, and piers remain distinct passage
classes. Same-name street pairs 45–90 m apart are surfaced by `audit` for
manual review. Ambiguity is left as separate work instead of silently
manufacturing 100%.

## Install

```bash
pip install -r requirements.txt
```

## Establish the inventory

Fetch a place resolved by OSM/Nominatim:

```bash
python amblecli.py fetch "San Francisco, California, USA" \
  --cache data/sf.graphml
```

For an exact project boundary, use a GeoJSON Polygon or Feature. This is the
preferred way to define what “100% of the zone” means:

```bash
python amblecli.py fetch --polygon data/zone.geojson \
  --cache data/zone.graphml
```

Audit the resulting denominator before trusting it:

```bash
python amblecli.py audit --cache data/zone.graphml
```

The audit reports canonical targets, merged duplicate surfaces, sidewalk
observation aliases, resolved identity collisions, passage classes, exclusions,
and important tags absent from an older cache. Newly fetched GraphML records the
requested tag schema explicitly, so a tag that simply has no current features
is not mistaken for a deficient cache. Re-fetch old GraphML files: a cache that did not retain
`foot`, `access`, `parking`, `indoor`, `man_made`, and related tags cannot
reliably distinguish every public pier from every private or parking access.

## Track actual walks

A recorded phone trace is evidence; a generated route is only a plan. Import
the GPX actually walked:

```bash
python amblecli.py import \
  --cache data/zone.graphml \
  --store data/progress.json \
  --gpx walks/recorded.gpx \
  --date 2026-07-13 \
  --note "Richmond loop"
```

Recordings accidentally left running on transit or in a car are handled at
import: sustained movement above 13 km/h (from GPX timestamps) is stripped
before matching, the track is split at the ride so nothing bridges across it,
and the removed distance is reported and persisted (`vehicle_m`, flagged by
`review`). Blocks can only be credited at walking speed.

The matcher reports four intentionally different distances:

- **physical GPX**: distance between contiguous recorded fixes; jumps over
  150 m are reported separately and are not presumed walked;
- **assigned**: observed portions of canonical passages, including partial
  blocks but never multiple parallel surfaces for one block;
- **near targets**: physical trace whose adjacent fixes both lie within 30 m of
  a canonical passage; the difference from assigned helps expose repeats versus
  inventory/matching loss;
- **new complete**: target distance newly brought past the completion threshold.

These need not be equal. Repeated walking, unnamed connectors, GPS gaps, and
partial end blocks make assigned/new-complete distance smaller than physical
distance. Assigned distance greater than physical distance is a warning sign and
should be investigated. Evidence is stored as disjoint fractional intervals so
two separate visits do not fill an unobserved middle, while complementary
partial visits can eventually complete a block. Intervals accumulate per
coverage target on one canonical axis: GPS wobbling between the parallel
surfaces of a block (roadway, either sidewalk) extends one contiguous interval
rather than fragmenting the evidence, so a street walked end-to-end never
records as a chain of disconnected slivers.

Exact GPX re-imports are idempotent. To reprocess a corpus after inventory or
matcher changes, build a new store rather than mutating the old one:

```bash
python amblecli.py rebuild \
  --cache data/zone.graphml \
  --store data/progress.canonical.json \
  --gpx walks/*.gpx   # NOTE: shell glob = alphabetical; list files in WALK
                      # ORDER instead — the diary credits each block to the
                      # first file that recorded it
```

`rebuild` records per-file physical, assigned, unmatched, skipped, and newly
completed metrics in the store's `imports` section. It refuses to overwrite an
existing store unless `--force` is supplied.

Review those results for impossible or suspicious cases:

```bash
python amblecli.py review --store data/progress.canonical.json
```

The review flags assigned distance greater than physical distance, embedded GPX
teleports, high unmatched rates, low proximity to the inventory, and traces
whose near-target distance collapses heavily when reduced to unique evidence.

## Plan the next walk

```bash
python amblecli.py status \
  --cache data/zone.graphml --store data/progress.canonical.json

python amblecli.py plan \
  --cache data/zone.graphml \
  --store data/progress.canonical.json \
  --target-km 8 \
  --style efficient \
  --out walks/walk_001
```

The output is a GPX, GeoJSON preview, and exact route JSON. The default is an
open route because forcing a return to the start usually adds repetition; add
`--loop` when required. `--start lat,lon` anchors a walk near a transit stop or
chosen corner.

Two planning styles are available:

- `efficient` chooses remaining canonical targets, connects them through the
  public routing graph, and solves the resulting undirected route-inspection
  problem to reduce deadheading. It is the general-purpose choice.
- `snake` groups long streets of one orientation into easy-to-follow sweeps on
  grids. Use `--axis ns` and `--axis ew` on separate outings. It prioritizes
  legibility rather than a global distance optimum.

The undirected Chinese Postman step is exact for the selected connected target
subgraph: odd intersections are paired by minimum-weight matching, duplicated
as necessary, and traversed in an Euler trail. The overall multi-walk problem
is still heuristic—selecting an approximately target-sized subset and choosing
future chunks is not a globally optimal solution across every outing. A final
route can exceed `--target-km` because a target block is indivisible and because
connecting/deadheading distance is added after target selection.

After a planned walk, prefer importing the recorded GPX. If it was followed
exactly and no recording exists, mark the route explicitly:

```bash
python amblecli.py done \
  --cache data/zone.graphml \
  --store data/progress.canonical.json \
  --route walks/walk_001.route.json
```

`done` validates every physical edge against the current cache before changing
progress. Traversed target passages count even when the planner labeled them as
connectors, because physically walking a block is what matters.

## Inspect progress

```bash
python amblecli.py remaining --cache data/zone.graphml --store data/progress.canonical.json
python amblecli.py log       --cache data/zone.graphml --store data/progress.canonical.json
python amblecli.py map       --cache data/zone.graphml --store data/progress.canonical.json \
  --out walks/progress.geojson
python amblecli.py render    --cache data/zone.graphml --store data/progress.canonical.json
```

Headline completion counts only fully completed canonical targets. Partial
observations are reported separately; they do not produce a contradictory
“100% observed, blocks still remaining” result. Inventory fingerprints warn
when a store is paired with a different cache.

`migrate` can geometrically carry evidence from one cache to a re-fetch and
backs up the original store, but a rebuild from original GPXs is stronger
evidence whenever those files are available.

## Known limits that matter at 100%

- OSM omissions and incorrect tags remain omissions and errors in Amble. The
  audit cannot prove the map is complete on the ground.
- Relation-only names, complex multi-level structures, and poorly tagged piers
  may need future OSM enrichment rather than guessing from geometry.
- A conservative alias heuristic can leave a duplicate target; a permissive
  heuristic could erase a real parallel passage. The former failure is safer
  but must be reviewed before declaring completion.
- Distinct curved routes can share the same graph endpoints. Amble detects and
  suffixes those rare identity collisions instead of sharing their progress;
  the inventory build fails if target IDs are not unique.
- Phone GPS cannot always distinguish very close, genuinely separate passages.
  Amble credits only the nearest canonical identity and preserves uncertainty;
  manual ground-truth evidence is not yet modeled.
- Disconnected public passage components are retained in status and maps.
  Planning selects a component per outing, so every island must eventually be
  planned rather than disappearing behind “largest component” cleanup.
- There is no mathematical proof that a sequence of bounded outings minimizes
  total repeats globally. The planner minimizes the route-inspection portion of
  each chosen chunk and uses heuristics for chunk selection.

## Tests

```bash
pytest
```

The offline suite covers canonical passage filtering and aliases, partial and
disjoint GPS evidence, dropout resistance, parallel-street false positives,
progress migration, route-key fidelity, planning, rendering, and exports.
