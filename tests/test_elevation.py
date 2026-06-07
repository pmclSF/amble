"""
network.add_elevations / attach_elevations. These hit OpenTopoData over HTTP, so
the network call is mocked — tests must never touch the live API.
"""
import json

import networkx as nx
import pytest

from amble import network as net


class _FakeResp:
    def __init__(self, results):
        self._results = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._results}


def _small_graph():
    G = nx.MultiGraph()
    G.add_node(1, x=-122.50, y=37.75)
    G.add_node(2, x=-122.49, y=37.76)
    G.add_edge(1, 2, length=100.0)
    return G


def test_add_elevations_fetches_attaches_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        locs = params["locations"].split("|")
        return _FakeResp([{"elevation": 42.0} for _ in locs])

    monkeypatch.setattr("requests.get", fake_get)
    G = _small_graph()
    cache = str(tmp_path / "elev.json")
    fetched = net.add_elevations(G, cache)

    assert fetched == 2
    assert calls["n"] == 1                       # both nodes in one batch
    assert G.nodes[1]["elevation"] == 42.0
    assert json.load(open(cache)) == {"1": 42.0, "2": 42.0}


def test_add_elevations_uses_cache_without_network(tmp_path, monkeypatch):
    cache = str(tmp_path / "elev.json")
    json.dump({"1": 11.0, "2": 22.0}, open(cache, "w"))

    def boom(*a, **k):
        raise AssertionError("network must not be hit when cache is complete")

    monkeypatch.setattr("requests.get", boom)
    G = _small_graph()
    assert net.add_elevations(G, cache) == 0      # nothing to fetch
    assert G.nodes[2]["elevation"] == 22.0


def test_attach_elevations_sets_known_and_counts_missing(tmp_path):
    cache = str(tmp_path / "elev.json")
    json.dump({"1": 5.0}, open(cache, "w"))       # node 2 deliberately absent
    G = _small_graph()
    missing = net.attach_elevations(G, cache)
    assert G.nodes[1]["elevation"] == 5.0
    assert missing == 1
    assert net.has_elevations(G) is False


def test_null_elevation_stored_as_zero(tmp_path, monkeypatch):
    def fake_get(url, params=None, timeout=None):
        locs = params["locations"].split("|")
        return _FakeResp([{"elevation": None} for _ in locs])   # ocean / no data

    monkeypatch.setattr("requests.get", fake_get)
    G = _small_graph()
    net.add_elevations(G, str(tmp_path / "e.json"))
    assert G.nodes[1]["elevation"] == 0.0


def test_add_elevations_batches_over_100_node_limit(tmp_path, monkeypatch):
    # OpenTopoData allows 100 locations/request, so 150 nodes must split into
    # exactly 2 batched calls (100 + 50) — never one oversized request that the
    # API would reject. Earlier tests used 2 nodes/1 call and never exercised
    # the batch loop. (time.sleep is the inter-batch pacing; stub it out.)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    sizes = []

    def fake_get(url, params=None, timeout=None):
        locs = params["locations"].split("|")
        sizes.append(len(locs))
        return _FakeResp([{"elevation": float(i)} for i in range(len(locs))])

    monkeypatch.setattr("requests.get", fake_get)
    G = nx.MultiGraph()
    for i in range(150):
        G.add_node(i, x=-122.5 + i * 1e-4, y=37.75)
    cache = str(tmp_path / "big.json")
    fetched = net.add_elevations(G, cache)

    assert fetched == 150
    assert sizes == [100, 50]                 # two batched requests, none over 100
    assert len(json.load(open(cache))) == 150  # every node cached after batching
