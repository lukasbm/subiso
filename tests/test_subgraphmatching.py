"""Backend tests for the MIT SubgraphMatching suite driver.

The suite solves the ordinary (non-induced) problem and reports only embedding
counts, so parity is checked against brute-force monomorphism enumeration.
Skipped when the vendored binary has not been built.
"""

from __future__ import annotations

import networkx as nx
import pytest

from subiso.backends.subgraphmatching import ALGORITHMS, SubgraphMatchingMatcher
from subiso.matcher import Query, UnsupportedOperation
from subiso.semantics import Semantics

pytestmark = pytest.mark.skipif(
    not SubgraphMatchingMatcher("quicksi").available(),
    reason="vendor/subgraphmatching binary not built (see docs/missing.md)",
)

PATTERNS = [nx.path_graph(3), nx.star_graph(3), nx.cycle_graph(4)]
ALGORITHMS_UNDER_TEST = ("quicksi", "dpiso")


def _brute(host: nx.Graph, pattern: nx.Graph) -> set[frozenset]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    return {frozenset(mapping) for mapping in matcher.subgraph_monomorphisms_iter()}


def _brute_embeddings(host: nx.Graph, pattern: nx.Graph) -> int:
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    return sum(1 for _ in matcher.subgraph_monomorphisms_iter())


def test_contains_agrees_with_brute_force_on_small_atlas():
    matchers = [SubgraphMatchingMatcher(algorithm) for algorithm in ALGORITHMS_UNDER_TEST]
    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 5 or host.number_of_edges() == 0:
            continue  # the suite segfaults on edgeless inputs and is gated off
        for pattern in PATTERNS:
            expected = bool(_brute(host, pattern))
            query = Query(host, pattern, Semantics.ORDINARY)
            for matcher in matchers:
                assert matcher.contains(query) == expected, (matcher.name, host.edges(), pattern.edges())
                checked += 1
    assert checked > 0


def test_embedding_count_matches_brute_force():
    host = nx.cycle_graph(6)
    pattern = nx.path_graph(3)
    query = Query(host, pattern, Semantics.ORDINARY)
    expected = _brute_embeddings(host, pattern)
    for algorithm in ALGORITHMS:
        matcher = SubgraphMatchingMatcher(algorithm)
        assert matcher.count_embeddings(query) == expected, algorithm
        assert matcher.count_embeddings(query, num=1) == min(expected, 1), algorithm


def test_labels_are_respected_and_wildcards_rejected():
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(pattern, {0: "C", 1: "O", 2: "C"}, "label")
    query = Query(host, pattern, Semantics.ORDINARY, node_attr="label")

    matcher = SubgraphMatchingMatcher("dpiso")
    assert matcher.contains(query)

    nx.set_node_attributes(pattern, {0: "C", 1: "N", 2: "C"}, "label")
    assert not matcher.contains(Query(host, pattern, Semantics.ORDINARY, node_attr="label"))

    nx.set_node_attributes(pattern, {0: "C", 1: None, 2: "C"}, "label")
    with pytest.raises(UnsupportedOperation, match="concrete integer labels"):
        matcher.contains(Query(host, pattern, Semantics.ORDINARY, node_attr="label"))


def test_induced_is_unsupported():
    host = nx.cycle_graph(6)
    pattern = nx.path_graph(3)
    with pytest.raises(UnsupportedOperation, match="ordinary"):
        SubgraphMatchingMatcher("quicksi").count_embeddings(Query(host, pattern, Semantics.INDUCED))


def test_edgeless_patterns_are_gated_off():
    """Upstream crashes on edgeless and disconnected patterns; never call it."""
    host = nx.cycle_graph(6)
    matcher = SubgraphMatchingMatcher("graphql")
    with pytest.raises(UnsupportedOperation, match="edgeless or disconnected"):
        matcher.contains(Query(host, nx.empty_graph(3), Semantics.ORDINARY))

    disconnected = nx.path_graph(3)
    disconnected.add_node(3)
    with pytest.raises(UnsupportedOperation, match="edgeless or disconnected"):
        matcher.contains(Query(host, disconnected, Semantics.ORDINARY))


def test_nonconsecutive_node_ids_are_relabelled():
    """Regression: planted patterns carry host ids; the suite requires [0, N-1].

    Without relabelling, five of seven engines crashed and CECI answered
    False on instances every other engine found (caught by the sweep's
    cross-engine agreement assertion).
    """
    raw = nx.gnp_random_graph(30, 0.4, seed=7)
    host = nx.relabel_nodes(raw, {node: node * 7 + 3 for node in raw})
    tree = nx.bfs_tree(raw, 0)
    pattern_nodes = list(tree.nodes())[:4]
    assert len(pattern_nodes) == 4
    pattern = raw.subgraph(pattern_nodes).copy()
    assert nx.is_connected(pattern) and pattern.number_of_edges() > 0

    query = Query(host, pattern, Semantics.ORDINARY)
    expected = nx.algorithms.isomorphism.GraphMatcher(host, pattern).subgraph_is_monomorphic()
    assert expected
    for algorithm in ALGORITHMS:
        matcher = SubgraphMatchingMatcher(algorithm)
        assert matcher.contains(query) == expected, algorithm
