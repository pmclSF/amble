# Amble

Plans walks that cover every street, alley, and stairway in San Francisco with
as little backtracking as possible. Meant for doing it over several years, a
neighborhood at a time.

## The problem

Covering every street with minimal repetition is the Route Inspection Problem,
usually called the Chinese Postman Problem. Walking is undirected — you can go
either way on any sidewalk or stair — and the undirected case is solvable
exactly in polynomial time.

Zero repetition isn't possible, though. A repeat-free closed route (an Eulerian
circuit) only exists if every intersection has an even number of streets meeting
it, and cities are full of odd ones: dead ends, T-junctions, the tops of
stairways. Some re-walking is forced; Amble finds the least of it and routes
around the rest.

The solver:

1. Find every odd-degree intersection.
2. Compute shortest paths between them.
3. Find a minimum-weight perfect matching on the odd intersections — the
   cheapest set of connections that makes every degree even.
4. Duplicate the streets along those connections and extract an Eulerian tour.
   The duplicated length is the unavoidable "deadhead" distance.

It can also return open routes (start and end in different places, e.g. two
transit stops), which usually beats looping back, and it picks the endpoints
that save the most walking.

## Install

```bash
pip install -r requirements.txt
```

networkx alone runs the solver and the tests. osmnx (which pulls in
geopandas/shapely) is only needed to download OSM data.

## Workflow

Work one neighborhood at a time. The whole-city graph is large (see scaling
notes), and a neighborhood is a natural chunk of progress.

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

Repeat steps 3–4. The planner only routes over what you haven't walked, so you
converge to full coverage.

## Configuring for your city

Only the examples are SF-specific; the math works off the graph's own
coordinates. Point `fetch` at any place OpenStreetMap can geocode — a
neighborhood, a city, or a custom boundary:

```bash
python amblecli.py fetch "Capitol Hill, Seattle, Washington, USA" \
      --cache data/caphill.graphml
```

Everything downstream (`status`/`plan`/`done`/`map`) takes that cache.

The one thing worth setting per city is corridor equivalences. When OSM tags one
physical strip with several overlapping names — SF's Great Highway is a road, a
promenade, a trail, and a park path — you don't want to walk it four times. Group
those names in [`amble/equivalents.py`](amble/equivalents.py) and only the
canonical one counts. It ships with the Great Highway as an example; replace it
with your city's, or empty the list.

## Progress, re-fetching, totals

Progress is keyed on a unique id (OSM node ids + key), so walking one street
never marks another done. The id holds as long as you reuse the same `.graphml`,
so keep your caches.

- Carrying progress across a re-fetch (which renumbers nodes): match old→new by
  geometry, once:
  `python amblecli.py migrate --from-cache old.graphml --to-cache new.graphml --store area.json`
- Finishing a neighborhood: `python amblecli.py remaining …` lists the named ways left.
- City-wide total (dedupes boundary streets by geometry):
  `python amblecli.py total --area sunset.graphml sunset.json --area bernal.graphml bernal.json …`

If a store and cache don't match (wrong pair, or a re-fetch you haven't
migrated), `status`/`plan` warn instead of showing a misleading 0%.

## What counts as done

The must-walk set is every named way: streets, alleys, named stairways, named
park paths. Unnamed ways (sidewalks, crossings, desire-line trails, connectors)
get walked through but don't count toward 100%. That's what makes the project
finishable — Buena Vista is ~50 km of path but only ~7 km of it is named.

At load time the network is cleaned up (`_load_cached`):

- driveways and parking aisles dropped (`drop_driveways`); alleys kept.
- divided ways collapsed (`collapse_divided_ways`): the two carriageways of a
  divided road, or two paths around a median, become one. It won't collapse two
  differently-named streets or a road and a parallel stairway, and won't
  disconnect the network.

## Routing styles (`plan --style ...`)

- `snake` (default; grids like the Sunset/Richmond): walk whole streets of one
  orientation end to end, jogging one block over to the next. Pick `--axis ns`
  or `--axis ew` and do them on separate days. Easy to follow without a map,
  about 15 turns instead of 50.
- `efficient` (parks, irregular areas): Rural-Postman coverage over the full
  connected graph. Named islands — a stray footpath or stairway — get stitched in
  with connectors instead of stranded. `--target-km` bounds total distance. The
  straight-preferring trail tends to ride long swirls, like a hill's ring roads.
- Hills (Twin Peaks, Buena Vista): `--style efficient --start-at-top` (needs
  `--elev`). Starts at the highest unwalked point and works down, taking named
  stairways up where it's handy; roads are fine. Contour-bands and spirals both
  did worse in testing.

## Elevation

`python amblecli.py elevation --cache X --elev e.json` fetches node heights
(free, no key). Then `plan --elev e.json` embeds a climb profile in the GPX and
enables `--start-at-top`.

## Scaling notes

- The matching is the expensive step: ~k shortest-path runs plus an O(k³)
  matching for k odd intersections. A dense pedestrian network has a lot of odd
  nodes, so don't run the whole city as one graph. Per neighborhood (hundreds to
  low thousands of edges) it's fast.
- A citywide single solve would want a faster matching (Blossom V / LEMON) and/or
  contracting degree-2 chains first. Going neighborhood by neighborhood sidesteps
  that, and it's more practical to walk anyway.
- Dead-end stubs always get walked twice, in and back out. That's unavoidable;
  open routes just spend them as cheaply as possible.

## Files

```
amble/postman.py        CPP solver + straight-preferring trail  [tested]
amble/network.py        OSM fetch, caching, filters, chunking, elevation
amble/straightline.py   snake (boustrophedon) router
amble/coverage.py       Rural-Postman named-way coverage (parks/hills)
amble/contour.py        hill helpers (start-at-top, stair ascent)
amble/export.py         GPX (route + turn cues + elevation) + GeoJSON
amble/progress.py       durable named-way coverage tracking
amblecli.py             CLI: fetch/status/plan/elevation/done/map/remaining
test_postman.py, tests/ pytest suite (91 tests)
```

## Tests

```bash
pytest          # 91 tests, offline (elevation HTTP is mocked)
```
