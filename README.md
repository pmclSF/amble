# Amble

A toolkit for a multi-year project: **walk every street, alley, and staircase
in San Francisco** while backtracking as little as mathematically possible.

## What problem this actually is

"Cover every street with minimal repetition" is the **Route Inspection
Problem**, better known as the **Chinese Postman Problem (CPP)**. For an
undirected network (correct for walking — you can go either way on any
sidewalk or stair) it is solvable *exactly* in polynomial time.

**Important honesty up front:** walking every street with *zero* repetition is
impossible for San Francisco. A repeat-free closed route (an *Eulerian
circuit*) exists only if every intersection has an even number of streets
meeting it. Real cities are full of odd intersections — dead ends (1 street),
T-junctions (3), the tops of staircases. So *some* re-walking is forced. What
this tool does is compute the **provably minimum** amount and route around it.

The solver:

1. Find every odd-degree intersection.
2. Compute shortest paths between them.
3. Find a **minimum-weight perfect matching** on those odd intersections — the
   cheapest set of connections that makes every degree even.
4. Duplicate the streets along those connections, then extract an Eulerian
   tour. The duplicated length is your unavoidable "deadhead" distance.

It also supports **open routes** (start and end at different places — e.g. two
different transit stops), which is almost always cheaper than forcing a loop,
and it automatically picks the endpoints that save the most walking.

## Install

```bash
pip install -r requirements.txt
```

`networkx` alone runs the solver and the test suite. `osmnx` (which pulls in
geopandas/shapely) is only needed to download real OSM data.

## The multi-year workflow

Do it **one neighborhood at a time** — both because the whole-city graph is
huge (see scaling notes) and because a neighborhood is a natural unit of
progress.

```bash
# 1. Pull a neighborhood's walkable network (includes alleys + highway=steps)
python amblecli.py fetch "Bernal Heights, San Francisco, California, USA" \
      --cache data/bernal.graphml

# 2. Check size + progress
python amblecli.py status --cache data/bernal.graphml --store data/progress.json

# 3. Plan the next ~8 km walk over streets you HAVEN'T done yet
python amblecli.py plan --cache data/bernal.graphml --store data/progress.json \
      --target-km 8 --out walks/walk_001
#    -> walk_001.gpx (load on your phone), .geojson (preview), .route.json

# 4. After walking it, record it
python amblecli.py done --cache data/bernal.graphml --store data/progress.json \
      --route walks/walk_001.route.json --note "foggy, great"

# 5. Export a walked-vs-remaining map anytime (open at geojson.io)
python amblecli.py map --cache data/bernal.graphml --store data/progress.json \
      --out walks/progress_map.geojson
```

Repeat steps 3–4. The planner always works over the *remaining* network, so
coverage marches to 100%.

## Configuring for your city

Nothing here is San Francisco-specific except the examples — the algorithms work
from the graph's own coordinates, so just point `fetch` at any place
OpenStreetMap can geocode (a neighborhood, a city, or a custom boundary):

```bash
python amblecli.py fetch "Capitol Hill, Seattle, Washington, USA" \
      --cache data/caphill.graphml
```

Everything downstream (`status`/`plan`/`done`/`map`) takes that cache and works
the same way.

The one thing worth tailoring per city is **corridor equivalences**. When OSM
tags a single physical strip with several overlapping names — SF's Great Highway
is mapped as a road, a promenade, a trail, *and* a park path — you don't want to
walk it four times. List those names together in
[`amble/equivalents.py`](amble/equivalents.py) and only the
group's canonical name counts toward 100%. It ships with the Great Highway as a
worked example; swap in your city's corridors, or empty the list if you have none.

## Progress, re-fetching, and city-wide totals

Progress is keyed on a **unique** id (OSM node ids + key) — collision-free, so
walking one street never falsely marks another done. It's stable for the same
`.graphml`; **keep your caches.**

- Carrying progress across a **re-fetch** (new node ids): match old→new by
  geometry, once:
  `python amblecli.py migrate --from-cache old.graphml --to-cache new.graphml --store area.json`
- Finishing a neighborhood: `python amblecli.py remaining …` prints the punch-list
  of named ways left.
- City-wide rollup (dedupes boundary streets by fuzzy geometry):
  `python amblecli.py total --area sunset.graphml sunset.json --area bernal.graphml bernal.json …`

If a store and cache don't match (wrong pair, or a re-fetch you haven't
migrated), `status`/`plan` warn instead of silently showing 0%.

## What counts as "done": named ways

The must-walk set is every **named** way — streets, alleys, named stairways,
named park paths. Unnamed ways (sidewalks, crossings, desire-line park trails,
minor connectors) are still *walked through* as connectors but don't count
toward 100%. This is the one rule that makes the project **finishable**:
a park like Buena Vista is ~50 km of total path but only ~7 km named.

At load time the network is also auto-cleaned (`_load_cached`):
- **driveways / parking aisles dropped** (`drop_driveways`) — alleys kept.
- **divided ways collapsed** (`collapse_divided_ways`) — the two "Upper Great
  Hwy" carriageways, or two paths around a median, become one. Never collapses
  two *differently*-named streets or a road + a parallel stairway, and never
  disconnects the network.

## Routing styles (`plan --style ...`)

- **`snake`** (default, grids like the Sunset/Richmond): walk whole streets of
  **one orientation** end to end, snaking one block over to the next parallel
  street. Pick `--axis ns` or `--axis ew` and do them on separate days. Easy to
  follow without a map; ~15 turns instead of ~50.
- **`efficient`** (parks / irregular areas): **Rural-Postman** coverage of the
  named ways over the *full* connected graph — named "islands" (a stray footpath
  or stairway) are stitched together with connector paths into one continuous
  walk, never stranded. `--target-km` bounds total walking (named + connectors).
  Its straight-preferring trail rides long "swirls" (a hill's ring roads).
- **Hills** (Twin Peaks, Buena Vista): `--style efficient --start-at-top`
  (needs `--elev`). Starts at the highest unwalked point and works **down**,
  climbing up via named **stairways** (Vulcan-class) when handy — roads are
  fine. Deep exploration showed contour-bands/spirals all lose to this.

## Elevation

`python amblecli.py elevation --cache X --elev e.json` fetches node heights (free,
no API key). Then `plan --elev e.json` embeds a climb profile in the GPX and
unlocks `--start-at-top`.

## Scaling notes (the honest limits)

- The matching step is the expensive part: with *k* odd intersections it costs
  ~*k* shortest-path runs plus an O(*k³*) matching. A dense pedestrian network
  has many odd nodes, so **don't run the whole city as one graph.** Per
  neighborhood (hundreds–low thousands of edges) it's fast; the included tests
  solve grids instantly.
- For a citywide single solve you'd want a faster matching (e.g. Blossom V /
  LEMON bindings) and/or contracting degree-2 chains first. The per-neighborhood
  approach sidesteps this and is also just more practical to walk.
- Dead-end stubs are always walked twice (in and back out) — that's correct and
  unavoidable, and the open-route option spends them as cheaply as possible.

## Files

```
amble/postman.py      CPP solver + straight-preferring trail   [tested]
amble/network.py      OSM fetch, caching, filters, chunking, elevation
amble/straightline.py snake (boustrophedon) router
amble/coverage.py     Rural-Postman named-way coverage (parks/hills)
amble/contour.py      hill helpers (start-at-top, stair ascent)
amble/export.py       GPX (route + turn cues + elevation) + GeoJSON
amble/progress.py     durable named-way coverage tracking
amblecli.py                     CLI: fetch/status/plan/elevation/done/map/remaining
test_postman.py + tests/     pytest suite (91 tests)
```

## Run the tests

```bash
pytest          # 91 tests, offline (elevation HTTP is mocked)
```
