"""Glasgow integration tests: every engine answer must agree with brute force."""

from __future__ import annotations

import networkx as nx
import pytest

subiso_glasgow = pytest.importorskip("subiso_glasgow")

from subiso.matcher import Query, UnsupportedOperation  # noqa: E402
from subiso.semantics import Semantics  # noqa: E402

PATTERNS = [
    nx.path_graph(3),
    nx.cycle_graph(4),
    nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (2, 4)]),  # house
    nx.star_graph(3),
]


def _brute(host: nx.Graph, pattern: nx.Graph, induced: bool) -> set[frozenset]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    iterator = matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()
    return {frozenset(mapping) for mapping in iterator}


def test_glasgow_agrees_with_brute_force_on_atlas():
    matcher = subiso_glasgow.GlasgowMatcher()
    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in PATTERNS:
            for induced in (True, False):
                semantics = Semantics.INDUCED if induced else Semantics.ORDINARY
                query = Query(host, pattern, semantics)
                expected = _brute(host, pattern, induced)

                assert matcher.contains(query) == bool(expected)
                mappings = matcher.mappings(query)
                assert {frozenset(mapping.values()) for mapping in mappings} == expected
                assert set(matcher.occurrences(query)) == expected
                assert matcher.count(query) == len(expected)
                assert matcher.count_embeddings(query) == len(mappings)

                first = matcher.first_mapping(query)
                assert (first is None) == (not expected)
                if first is not None:
                    assert frozenset(first.values()) in expected

                for cap in (1, 2):
                    assert matcher.count(query, cap=cap) == min(len(expected), cap)
                checked += 1
    assert checked > 0


def test_glasgow_labels_and_unsupported_requests():
    matcher = subiso_glasgow.GlasgowMatcher()
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(pattern, {0: "C", 1: "O", 2: "C"}, "label")

    query = Query(host, pattern, Semantics.INDUCED, node_attr="label")
    assert matcher.count(query) == 1  # only the C-O-C path centred on host vertex 1

    nx.set_node_attributes(pattern, {0: "C", 1: None, 2: "C"}, "label")
    wildcarded = Query(host, pattern, Semantics.INDUCED, node_attr="label")
    with pytest.raises(UnsupportedOperation, match="wildcards"):
        matcher.contains(wildcarded)

    with pytest.raises(UnsupportedOperation, match="domains"):
        matcher.contains(Query(host, pattern, Semantics.INDUCED), domains={0: {0}})
