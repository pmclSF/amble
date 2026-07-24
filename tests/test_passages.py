import networkx as nx

from amble import network, passages, progress, trace


def _node(G, n, x, y):
    G.add_node(n, x=x, y=y)


def test_parallel_surfaces_of_same_block_are_one_target():
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "s0", 0.0, 0.00008); _node(G, "s1", 0.001, 0.00008)
    G.add_edge("r0", "r1", length=111, name="Kirkham Street", highway="residential")
    G.add_edge("s0", "s1", length=111, name="Kirkham Street", highway="footway")

    passages.annotate_passages(G)
    required = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
                if progress.is_required(d)]
    assert len(required) == 1
    ids = {d["coverage_id"] for *_e, d in G.edges(keys=True, data=True)}
    assert len(ids) == 1


def test_adjacent_blocks_do_not_merge():
    G = nx.MultiGraph()
    for i, x in enumerate((0.0, 0.001, 0.002)):
        _node(G, i, x, 0.0)
    G.add_edge(0, 1, length=111, name="Kirkham Street", highway="residential")
    G.add_edge(1, 2, length=111, name="Kirkham Street", highway="residential")
    passages.annotate_passages(G)
    assert sum(progress.is_required(d) for *_e, d in G.edges(data=True)) == 2


def test_distinct_routes_with_same_nodes_never_share_coverage_id():
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.001, 0.0)
    # Length mismatch deliberately prevents geometric clustering, as with two
    # different arcs of a curving street between the same intersections.
    G.add_edge("a", "b", length=40, name="Loop Road", highway="residential",
               osmid=1)
    G.add_edge("a", "b", length=100, name="Loop Road", highway="residential",
               osmid=2)
    _H, audit = network.prepare_graph(G)
    ids = [d["coverage_id"] for *_e, d in G.edges(data=True)
           if d.get("coverage_required")]
    assert len(ids) == len(set(ids)) == 2
    assert audit["passages"]["id_collisions_resolved"] == 1


def test_parallel_staircase_remains_distinct_from_street():
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.001, 0.0)
    G.add_edge("a", "b", key=0, length=111, name="Hill Street", highway="residential")
    G.add_edge("a", "b", key=1, length=111, name="Hill Steps", highway="steps")
    passages.annotate_passages(G)
    req = [d for *_e, d in G.edges(data=True) if progress.is_required(d)]
    assert len(req) == 2
    assert {d["coverage_class"] for d in req} == {"street", "staircase"}


def test_parking_and_private_edges_are_not_targets_or_shortcuts():
    G = nx.MultiGraph()
    for i in range(4):
        _node(G, i, i * 0.001, 0.0)
    G.add_edge(0, 1, length=100, name="Public Street", highway="residential")
    G.add_edge(1, 2, length=100, name="Garage Ramp", highway="service",
               service="parking_aisle")
    G.add_edge(2, 3, length=100, name="Gated Walk", highway="footway",
               access="private")
    H, audit = network.prepare_graph(G)
    assert H.number_of_edges() == 1
    assert audit["removed"]["parking_or_driveway"]["edges"] == 1
    assert audit["removed"]["restricted_access"]["edges"] == 1


def test_out_of_scope_zone_is_removed_and_audited():
    # Alcatraz: real streets/paths, deliberately outside the project's goal.
    G = nx.MultiGraph()
    _node(G, "a", -122.4232, 37.8268); _node(G, "b", -122.4226, 37.8268)
    _node(G, "x", -122.41, 37.80); _node(G, "y", -122.409, 37.80)
    G.add_edge("a", "b", length=53, name="Main Road", highway="service")
    G.add_edge("x", "y", length=88, name="Mainland Street",
               highway="residential")
    H, audit = network.prepare_graph(G)
    assert audit["removed"]["out_of_scope"]["edges"] == 1
    names = {d.get("coverage_name") for *_e, d in H.edges(data=True)
             if d.get("coverage_required")}
    assert names == {"Mainland Street"}


def test_foot_permission_overrides_generic_access_no():
    assert passages.exclusion_reason({"access": "no", "foot": "yes",
                                      "name": "Public Walk"}) is None


def test_named_plaza_and_transit_platform_are_not_linear_targets():
    assert passages.exclusion_reason({"name": "Civic Plaza", "area": "yes",
                                      "highway": "pedestrian"}) == "non_linear_area"
    assert passages.exclusion_reason({"name": "Platform 1",
                                      "railway": "platform"}) == "transit_platform"
    assert passages.exclusion_reason({"name": "Harbor Pier", "area": "yes",
                                      "man_made": "pier"}) is None


def test_legacy_alias_record_completes_canonical_target():
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "s0", 0.0, 0.00008); _node(G, "s1", 0.001, 0.00008)
    G.add_edge("r0", "r1", length=111, name="Main Street", highway="residential")
    G.add_edge("s0", "s1", length=111, name="Main Street", highway="footway")
    passages.annotate_passages(G)
    sidewalk_id = progress.legacy_edge_id("s0", "s1", 0)
    store = {"walked": {sidewalk_id: {"date": "2026-01-01", "note": "old"}}}
    assert progress.stats(G, store)["pct_done"] == 100.0


def test_unnamed_sidewalk_is_observation_alias_not_second_target():
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "s0", 0.0, 0.00008); _node(G, "s1", 0.001, 0.00008)
    G.add_edge("r0", "r1", length=100, name="Main Street",
               highway="residential")
    G.add_edge("s0", "s1", length=112, highway="footway",
               footway="sidewalk")

    H, audit = network.prepare_graph(G)
    assert audit["passages"]["targets"] == 1
    assert audit["passages"]["observation_aliases"] == 1
    sidewalk = H["s0"]["s1"][0]
    assert sidewalk["coverage_required"] is False

    # The fixes lie exactly on the mapped sidewalk.  They must complete the one
    # named block using its 100 m canonical length, not create/measure a 112 m
    # sidewalk target.
    matched = trace.match_trace(H, [(0.00008, 0.0), (0.00008, 0.001)])
    target = next(d["coverage_id"] for *_e, d in H.edges(data=True)
                  if d.get("coverage_required"))
    assert matched["edge_ids"] == {target}
    assert matched["named_m"] == 100.0


def test_divided_carriageways_40m_apart_are_one_obligation():
    # Two same-name parallel carriageways across a ~40 m median (Sunset Blvd,
    # Park Presidio): ONE coverage target, not two.
    G = nx.MultiGraph()
    _node(G, "a0", 0.0, 0.0); _node(G, "a1", 0.001, 0.0)
    _node(G, "b0", 0.0, 0.00036); _node(G, "b1", 0.001, 0.00036)
    G.add_edge("a0", "a1", length=111, name="Sunset Boulevard",
               highway="secondary")
    G.add_edge("b0", "b1", length=111, name="Sunset Boulevard",
               highway="secondary")
    _G, audit = passages.annotate_passages(G)
    assert audit["targets"] == 1
    assert sum(progress.is_required(d) for *_e, d in G.edges(data=True)) == 1


def test_far_apart_same_name_pair_stays_separate_but_is_audited():
    # ~60 m apart: past the 45 m bar — kept as two targets, but surfaced in
    # the divided_near_miss audit for manual review.
    G = nx.MultiGraph()
    _node(G, "a0", 0.0, 0.0); _node(G, "a1", 0.001, 0.0)
    _node(G, "b0", 0.0, 0.00054); _node(G, "b1", 0.001, 0.00054)
    G.add_edge("a0", "a1", length=111, name="Alemany Boulevard",
               highway="secondary")
    G.add_edge("b0", "b1", length=111, name="Alemany Boulevard",
               highway="secondary")
    _G, audit = passages.annotate_passages(G)
    assert audit["targets"] == 2
    assert audit["divided_near_miss"]["pairs"] == 1


def test_foot_restricted_roadway_takes_generic_path_as_evidence():
    # foot=no roadway (walkers use the parallel unnamed path): the street
    # stays a TARGET, and the path becomes its observation alias.
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "p0", 0.0, 0.00012); _node(G, "p1", 0.001, 0.00012)
    G.add_edge("r0", "r1", length=111, name="Park Presidio Boulevard",
               highway="trunk", foot="no")
    G.add_edge("p0", "p1", length=111, highway="path")
    _G, audit = passages.annotate_passages(G)
    assert audit["targets"] == 1
    assert audit["foot_restricted_targets"]["targets"] == 1
    assert audit["observation_aliases"] == 1
    path = G["p0"]["p1"][0]
    road = G["r0"]["r1"][0]
    assert path["coverage_id"] == road["coverage_id"]
    assert path["coverage_required"] is False


def test_named_pier_without_highway_tag_is_a_target():
    G = nx.MultiGraph()
    _node(G, "p0", 0.0, 0.0); _node(G, "p1", 0.0005, 0.0)
    G.add_edge("p0", "p1", length=55, name="Pier 7", man_made="pier")
    _G, audit = passages.annotate_passages(G)
    req = [d for *_e, d in G.edges(data=True) if progress.is_required(d)]
    assert len(req) == 1
    assert req[0]["coverage_class"] == "pier"


def test_busway_is_excluded():
    assert passages.exclusion_reason(
        {"name": "Van Ness Bus Rapid Transit",
         "highway": "busway"}) == "non_pedestrian_highway"


def test_named_track_and_cycleway_are_trail_class_targets():
    assert passages.passage_class({"highway": "track"}) == "trail"
    assert passages.passage_class({"highway": "cycleway"}) == "trail"


def test_intersection_shard_is_absorbed_not_a_target():
    # A 7 m named stub inside the intersection box, adjacent to the street's
    # real 111 m block: junction ground, not a separate obligation. It must
    # vanish from targets AND from evidence surfaces (no coverage_id).
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.001, 0.0)
    _node(G, "c", 0.001063, 0.0)
    G.add_edge("a", "b", length=111, name="Alma Street", highway="residential")
    G.add_edge("b", "c", length=7, name="Alma Street", highway="residential")
    _G, audit = passages.annotate_passages(G)
    assert audit["intersection_shards"]["targets"] == 1
    assert audit["targets"] == 1
    shard = G["b"]["c"][0]
    assert "coverage_id" not in shard
    assert shard["coverage_exclusion"] == "intersection_shard"
    # explicit False: presence of the key is authoritative for is_required —
    # popping it would resurrect the shard via the legacy named-way fallback
    assert shard["coverage_required"] is False
    assert not progress.is_required(shard)


def test_collinear_same_name_halves_do_not_merge_serially():
    # Two ~10 m collinear same-name pieces sharing a node are SERIAL ground:
    # merging them made walking one half complete the never-walked other half.
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.00009, 0.0)
    _node(G, "c", 0.00018, 0.0)
    G.add_edge("a", "b", length=10, name="Tiny Terrace", highway="residential")
    G.add_edge("b", "c", length=10, name="Tiny Terrace", highway="residential")
    _G, audit = passages.annotate_passages(G)
    assert audit["targets"] == 2
    assert audit["intersection_shards"]["targets"] == 0


def test_perpendicular_short_leg_is_real_ground_not_a_shard():
    # A 15 m terminal leg at right angles to its same-name main street extends
    # BEYOND the corner — real obligation, not junction paint.
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.001, 0.0)
    _node(G, "t", 0.001, 0.000135)
    G.add_edge("a", "b", length=111, name="Elm Street", highway="residential")
    G.add_edge("b", "t", length=15, name="Elm Street", highway="residential")
    _G, audit = passages.annotate_passages(G)
    assert audit["intersection_shards"]["targets"] == 0
    assert audit["targets"] == 2


def test_shard_off_the_non_rep_carriageway_is_still_demoted():
    # Divided same-name pair merges to one target; a 7 m collinear stub hangs
    # off the NON-rep carriageway's endpoint. Neighbour detection must see
    # cluster MEMBERS, not only elected reps.
    G = nx.MultiGraph()
    _node(G, "a0", 0.0, 0.0); _node(G, "a1", 0.001, 0.0)
    _node(G, "b0", 0.0, 0.00036); _node(G, "b1", 0.001, 0.00036)
    _node(G, "s", 0.001063, 0.00036)
    G.add_edge("a0", "a1", length=111, name="Dual Street",
               highway="residential")
    G.add_edge("b0", "b1", length=111, name="Dual Street",
               highway="residential")
    G.add_edge("b1", "s", length=7, name="Dual Street", highway="residential")
    _G, audit = passages.annotate_passages(G)
    assert audit["targets"] == 1                      # merged pair only
    assert audit["intersection_shards"]["targets"] == 1


def test_short_divided_block_with_one_long_carriageway_is_kept():
    # Merged divided block with carriageways 19 m + 30 m: the cluster's
    # LONGEST member decides shard-ness, not the (short) elected rep.
    G = nx.MultiGraph()
    _node(G, "m0", 0.0, 0.0); _node(G, "m1", 0.001, 0.0)
    _node(G, "a0", 0.00105, 0.0); _node(G, "a1", 0.00122, 0.0)
    _node(G, "b0", 0.00104, 0.00027); _node(G, "b1", 0.00131, 0.00027)
    G.add_edge("m0", "m1", length=111, name="Cross Street",
               highway="residential")
    G.add_edge("a0", "a1", length=19, name="Median Street",
               highway="residential")
    G.add_edge("b0", "b1", length=30, name="Median Street",
               highway="residential")
    _G, audit = passages.annotate_passages(G)
    med = [d for *_e, d in G.edges(data=True)
           if d.get("coverage_name") == "Median Street"
           and d.get("coverage_required")]
    assert len(med) >= 1                       # never erased as a shard
    assert audit["intersection_shards"]["targets"] == 0


def test_sidewalk_separate_street_does_not_take_park_paths_as_evidence():
    # sidewalk=separate means the real sidewalk IS mapped; a coincidental
    # unnamed park path 20 m away must not become evidence for the street.
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "p0", 0.0, 0.00018); _node(G, "p1", 0.001, 0.00018)
    G.add_edge("r0", "r1", length=111, name="Park Edge Street",
               highway="residential", sidewalk="separate")
    G.add_edge("p0", "p1", length=111, highway="path")
    _G, audit = passages.annotate_passages(G)
    assert audit["observation_aliases"] == 0
    assert not G["p0"]["p1"][0].get("coverage_id")


def test_named_pier_polygon_ring_is_not_a_target():
    G = nx.MultiGraph()
    _node(G, "p0", 0.0, 0.0); _node(G, "p1", 0.0005, 0.0)
    G.add_edge("p0", "p1", length=55, name="Pier 39", man_made="pier",
               area="yes")
    _G, audit = passages.annotate_passages(G)
    d = G["p0"]["p1"][0]
    assert d["coverage_required"] is False
    assert d["coverage_exclusion"] == "pier_area_ring"
    assert audit["pier_area_rings_not_required"]["edges"] == 1
    assert audit["targets"] == 0


def test_standalone_short_named_passage_is_kept():
    # A real 15 m alley/stair with no longer same-name neighbour stays a target.
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.000135, 0.0)
    _node(G, "x", 0.0, 0.0002); _node(G, "y", 0.001, 0.0002)
    G.add_edge("a", "b", length=15, name="Burritt Street", highway="residential")
    G.add_edge("x", "y", length=111, name="Bush Street", highway="residential")
    _G, audit = passages.annotate_passages(G)
    assert audit["intersection_shards"]["targets"] == 0
    assert audit["targets"] == 2


def test_short_staircase_flight_next_to_same_name_flight_is_kept():
    # Chained stair flights share a name; short flights are real climbing,
    # never junction paint — the shard rule applies to streets only.
    G = nx.MultiGraph()
    _node(G, "a", 0.0, 0.0); _node(G, "b", 0.00012, 0.0)
    _node(G, "c", 0.0005, 0.0)
    G.add_edge("a", "b", length=13, name="Vulcan Stairway", highway="steps")
    G.add_edge("b", "c", length=42, name="Vulcan Stairway", highway="steps")
    _G, audit = passages.annotate_passages(G)
    assert audit["intersection_shards"]["targets"] == 0
    assert audit["targets"] == 2


def test_zigzag_between_surfaces_of_one_block_stays_contiguous():
    # GPS wobbling between a block's roadway and its sidewalk alias (same
    # coverage target) must accumulate ONE contiguous interval. Per-surface
    # accounting used to shatter it into single-fix fragments (< 0.04 width,
    # all dropped), recording a fully-walked block as a 15-50% partial —
    # the 'missing segments where walks were not contiguous' bug.
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "s0", 0.0, 0.00009); _node(G, "s1", 0.001, 0.00009)
    G.add_edge("r0", "r1", length=111, name="Main Street",
               highway="residential")
    G.add_edge("s0", "s1", length=111, highway="footway", footway="sidewalk")
    H, audit = network.prepare_graph(G)
    assert audit["passages"]["observation_aliases"] == 1
    cid = next(d["coverage_id"] for *_e, d in H.edges(data=True)
               if d.get("coverage_required"))

    fixes = [(0.000005 if i % 2 == 0 else 0.000085, i * 0.00005)
             for i in range(21)]          # nearest surface flips every fix
    m = trace.match_trace(H, fixes)
    assert cid in m["edge_ids"]
    frac = sum(hi - lo for lo, hi in m["edge_spans"][cid])
    assert frac >= 0.85, frac
    assert m["named_m"] >= 0.85 * 111


def test_inset_sidewalk_evidence_is_scaled_onto_the_streets_axis():
    # A 64 m sidewalk inset ~18 m from each corner of a 100 m block: walking
    # the sidewalk end-to-end is evidence for the MIDDLE of the block only.
    # Fractions measured on the alias polyline must map onto the rep's axis
    # ([~0.18, ~0.82]), never be applied unscaled as [0, 1].
    G = nx.MultiGraph()
    _node(G, "W", 0.0, 0.0); _node(G, "E", 0.0009, 0.0)
    _node(G, "S1", 0.00016, 0.00008); _node(G, "S2", 0.00074, 0.00008)
    G.add_edge("W", "E", length=100, name="Inset Street",
               highway="residential")
    G.add_edge("S1", "S2", length=64.5, highway="footway",
               footway="sidewalk")
    H, audit = network.prepare_graph(G)
    assert audit["passages"]["observation_aliases"] == 1
    cid = next(d["coverage_id"] for *_e, d in H.edges(data=True)
               if d.get("coverage_required"))
    fixes = [(0.00008, 0.00016 + i * 0.00003) for i in range(20)]
    m = trace.match_trace(H, fixes)
    spans = m["edge_spans"][cid]
    frac = sum(hi - lo for lo, hi in spans)
    assert 0.45 <= frac <= 0.75, spans
    assert not progress.is_complete({"intervals": spans}, 100.0)
    assert m["matched_m"] <= 80.0, m["matched_m"]


def test_alias_with_reversed_endpoint_order_does_not_invert_evidence():
    # The alias's endpoints sort lexicographically OPPOSITE to the street's
    # (tiny latitude jitter decides the order for near-E-W ways). Walking only
    # the WEST half, wobbling between roadway and sidewalk, must credit ~half
    # the block — an endpoint-sort orientation would invert the alias evidence
    # and union the two halves into a fabricated full block.
    G = nx.MultiGraph()
    _node(G, "W", 0.0, 0.0); _node(G, "E", 0.0009, 0.0)
    _node(G, "S2", 0.0009, 0.00008)          # east end, LOWER latitude...
    _node(G, "S1", 0.0, 0.000081)            # ...than the west end
    G.add_edge("W", "E", length=100, name="Even Street", highway="residential")
    G.add_edge("S2", "S1", length=100, highway="footway", footway="sidewalk")
    H, audit = network.prepare_graph(G)
    assert audit["passages"]["observation_aliases"] == 1
    cid = next(d["coverage_id"] for *_e, d in H.edges(data=True)
               if d.get("coverage_required"))
    out = [(0.00002 if i % 2 == 0 else 0.00006, i * 0.000028)
           for i in range(17)]               # west half only, surfaces flip
    m = trace.match_trace(H, out + list(reversed(out)))
    spans = m["edge_spans"][cid]
    frac = sum(hi - lo for lo, hi in spans)
    assert frac <= 0.72, spans               # ~half the block, never all of it
    assert not progress.is_complete({"intervals": spans}, 100.0)


def test_unnamed_generic_footway_is_not_assumed_to_be_a_sidewalk():
    G = nx.MultiGraph()
    _node(G, "r0", 0.0, 0.0); _node(G, "r1", 0.001, 0.0)
    _node(G, "p0", 0.0, 0.00008); _node(G, "p1", 0.001, 0.00008)
    G.add_edge("r0", "r1", length=100, name="Park Road",
               highway="residential")
    G.add_edge("p0", "p1", length=100, highway="footway")
    H, audit = network.prepare_graph(G)
    assert audit["passages"]["observation_aliases"] == 0
    assert not H["p0"]["p1"][0].get("coverage_id")
