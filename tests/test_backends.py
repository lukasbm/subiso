"""Backend dispatch must be exact; these compare every engine against brute force."""

from __future__ import annotations

from collections.abc import Collection, Hashable

import networkx as nx
import pytest

from subiso import (
    are_isomorphic,
    automorphism_orbits,
    contains_subgraph,
    count_automorphisms,
    count_subgraph_positions,
    get_first_subgraph_mapping,
    get_subgraph_mappings,
)
from subiso.backends import fastiso, lad, vf2, vf2pp, vf3
from subiso.matcher import Query, UnsupportedOperation
from subiso.semantics import Semantics

PATTERNS = [
    nx.path_graph(3),
    nx.cycle_graph(4),
    nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (2, 4)]),  # house
    nx.star_graph(3),
]


def _brute(host: nx.Graph, pattern: nx.Graph, induced: bool) -> set[frozenset[int]]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    iterator = matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()
    return {frozenset(mapping) for mapping in iterator}


def test_contains_and_count_match_brute_force_on_atlas():
    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in PATTERNS:
            for induced in (True, False):
                expected = _brute(host, pattern, induced)
                assert contains_subgraph(host, pattern, induced=induced) == bool(expected), (
                    host.edges(),
                    pattern.edges(),
                    induced,
                )
                assert count_subgraph_positions(host, pattern, induced=induced) == len(expected), (
                    host.edges(),
                    pattern.edges(),
                    induced,
                )
                checked += 1
    assert checked > 0


def test_mappings_agree_with_occurrence_count():
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in PATTERNS:
            for induced in (True, False):
                expected = _brute(host, pattern, induced)
                mappings = get_subgraph_mappings(host, pattern, induced=induced)
                sets = {frozenset(mapping.values()) for mapping in mappings}
                assert sets == expected
                first = get_first_subgraph_mapping(host, pattern, induced=induced)
                assert (first is None) == (not expected)
                if first is not None:
                    assert frozenset(first.values()) in expected


def test_each_backend_agrees_with_brute_force_on_atlas():
    """Every backend is exact on its own, not only through the router."""
    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in PATTERNS:
            for induced in (True, False):
                expected = _brute(host, pattern, induced)

                assert vf2.contains(host, pattern, induced=induced) == bool(expected)
                assert vf2.count(host, pattern, induced=induced) == len(expected)
                assert set(vf2.occurrences(host, pattern, induced=induced, cap=None)) == expected
                assert {frozenset(m.values()) for m in vf2.mappings(host, pattern, induced=induced)} == expected

                assert vf2pp.contains(host, pattern, induced=induced) == bool(expected)
                assert vf2pp.count(host, pattern, induced=induced) == len(expected)
                assert set(vf2pp.occurrences(host, pattern, induced=induced)) == expected
                assert {frozenset(m.values()) for m in vf2pp.mappings(host, pattern, induced=induced)} == expected
                first = vf2pp.first_mapping(host, pattern, induced=induced)
                assert (first is None) == (not expected)
                if first is not None:
                    assert frozenset(first.values()) in expected

                assert lad.contains(host, pattern, induced=induced) == bool(expected)
                assert lad.count(host, pattern, induced=induced) == len(expected)
                assert set(lad.occurrences(host, pattern, induced=induced)) == expected
                assert {frozenset(m.values()) for m in lad.mappings(host, pattern, induced=induced)} == expected

                if vf3.available() and host.number_of_edges() and pattern.number_of_edges():
                    assert vf3.contains(host, pattern, induced=induced) == bool(expected)
                    assert {frozenset(m.values()) for m in vf3.mappings(host, pattern, induced=induced)} == expected

                if induced and fastiso.available() and host.number_of_edges() and pattern.number_of_edges():
                    embeddings = fastiso.count_induced_embeddings(host, pattern)
                    assert embeddings // count_automorphisms(pattern) == len(expected)
                checked += 1
    assert checked > 0


def _brute_automorphism_orbits(graph: nx.Graph) -> set[frozenset]:
    parent = {node: node for node in graph}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for mapping in nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter():
        for node, image in mapping.items():
            union(node, image)

    groups: dict[Hashable, set] = {}
    for node in graph:
        groups.setdefault(find(node), set()).add(node)
    return {frozenset(group) for group in groups.values()}


def _normalized_orbits(orbits: Collection[frozenset]) -> set[tuple]:
    return {tuple(sorted(orbit)) for orbit in orbits}


def test_automorphism_orbits_match_bruteforce():
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        expected = _brute_automorphism_orbits(host)
        actual = {frozenset(orbit) for orbit in automorphism_orbits(host)}
        assert _normalized_orbits(actual) == _normalized_orbits(expected), (
            host.number_of_nodes(),
            sorted(host.edges()),
        )


def _brute_anchored(host: nx.Graph, pattern: nx.Graph, induced: bool, domains: dict) -> set[frozenset]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    iterator = matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()
    found = set()
    for mapping in iterator:
        pattern_to_host = {pattern_node: host_node for host_node, pattern_node in mapping.items()}
        if all(pattern_to_host[node] in allowed for node, allowed in domains.items()):
            found.add(frozenset(pattern_to_host.values()))
    return found


def test_anchored_domains_restrict_matching():
    host: nx.Graph = nx.cycle_graph(6)
    pattern: nx.Graph = nx.path_graph(3)
    domains: dict[Hashable, Collection[Hashable]] = {1: {0}}
    expected = _brute_anchored(host, pattern, True, domains)
    assert expected  # C6 hosts an induced P3 centred at 0
    assert contains_subgraph(host, pattern, induced=True, domains=domains) == bool(expected)
    assert count_subgraph_positions(host, pattern, induced=True, domains=domains) == len(expected)
    assert {
        frozenset(m.values()) for m in get_subgraph_mappings(host, pattern, induced=True, domains=domains)
    } == expected
    first = get_first_subgraph_mapping(host, pattern, induced=True, domains=domains)
    assert first is not None and first[1] == 0 and frozenset(first.values()) in expected

    # An anchored induced count must deduplicate images instead of dividing by |Aut|.
    edge: nx.Graph = nx.Graph([(0, 1)])
    path: nx.Graph = nx.path_graph(2)
    edge_domains: dict[Hashable, Collection[Hashable]] = {0: {0}}
    assert count_subgraph_positions(edge, path, induced=True, domains=edge_domains) == 1

    empty_domains: dict[Hashable, Collection[Hashable]] = {1: set()}
    assert not contains_subgraph(host, pattern, induced=True, domains=empty_domains)
    with pytest.raises(ValueError, match="unlabelled-engine features"):
        contains_subgraph(host, pattern, induced=True, node_attr="label", domains=domains)
    assert get_first_subgraph_mapping(host, pattern, induced=True, time_limit=10.0) is not None


def test_capped_count_is_minimum_of_exact_and_cap():
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in PATTERNS:
            for induced in (True, False):
                exact = count_subgraph_positions(host, pattern, induced=induced)
                for cap in (1, 2, 3):
                    assert count_subgraph_positions(host, pattern, induced=induced, cap=cap) == min(exact, cap)

    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    labelled: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(labelled, {0: "C", 1: "O", 2: "C"}, "label")
    exact = count_subgraph_positions(host, labelled, induced=True, node_attr="label")
    assert count_subgraph_positions(host, labelled, induced=True, node_attr="label", cap=2) == min(exact, 2)

    with pytest.raises(ValueError, match="anchored counting"):
        count_subgraph_positions(host, labelled, induced=True, domains={0: {0}}, cap=1)


def test_are_isomorphic_matches_relabellings_and_labels():
    for left in nx.graph_atlas_g():
        if left.number_of_nodes() > 6:
            continue
        shuffled = nx.relabel_nodes(left, {node: node * 7 + 3 for node in left})
        assert are_isomorphic(left, shuffled)
    assert are_isomorphic(nx.Graph(), nx.Graph())
    assert not are_isomorphic(nx.path_graph(4), nx.cycle_graph(4))

    labelled: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(labelled, {0: "a", 1: "b", 2: "a"}, "label")
    same = nx.relabel_nodes(labelled, {0: 2, 1: 0, 2: 1})
    other: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(other, {0: "a", 1: "a", 2: "b"}, "label")
    assert are_isomorphic(labelled, same, node_attr="label")
    assert not are_isomorphic(labelled, other, node_attr="label")

    wildcarded: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(wildcarded, {0: "a", 1: None, 2: "a"}, "label")
    assert are_isomorphic(labelled, wildcarded, node_attr="label")
    assert not are_isomorphic(wildcarded, labelled, node_attr="label")


def test_empty_pattern_semantics():
    host = nx.path_graph(3)
    empty = nx.Graph()
    assert contains_subgraph(host, empty, induced=True)
    assert count_subgraph_positions(host, empty, induced=True) == 1
    assert get_first_subgraph_mapping(host, empty, induced=True) == {}
    oversize = nx.cycle_graph(5)
    assert not contains_subgraph(host, oversize)
    assert count_subgraph_positions(host, oversize) == 0
    assert get_first_subgraph_mapping(host, oversize) is None


def test_homomorphism_is_not_a_containment_relation():
    assert {semantics.value for semantics in Semantics} == {"induced", "ordinary"}


def test_engines_reject_inapplicable_inputs():
    vf3_matcher = vf3.Vf3Matcher()
    if not vf3_matcher.available():
        pytest.skip("vf3py is not installed")
    edgeless = nx.empty_graph(3)
    with pytest.raises(UnsupportedOperation, match="edge-less"):
        vf3_matcher.contains(Query(nx.path_graph(3), edgeless, Semantics.INDUCED))


def test_vf3_variants_agree_with_brute_force():
    if not vf3.available():
        pytest.skip("vf3py is not installed")
    from subiso.backends.vf3 import Vf3lMatcher, Vf3pMatcher

    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 4 or host.number_of_edges() == 0:
            continue
        for pattern in PATTERNS[:2]:
            if pattern.number_of_edges() == 0:
                continue
            for induced in (True, False):
                expected = _brute(host, pattern, induced)
                for matcher in (Vf3lMatcher(), Vf3pMatcher(num_threads=2)):
                    mappings = matcher.mappings(
                        Query(host, pattern, Semantics.INDUCED if induced else Semantics.ORDINARY)
                    )
                    assert {frozenset(m.values()) for m in mappings} == expected, (
                        matcher.name,
                        host.edges(),
                        pattern.edges(),
                    )
                    assert matcher.contains(
                        Query(host, pattern, Semantics.INDUCED if induced else Semantics.ORDINARY)
                    ) == bool(expected)
                    checked += 1
    assert checked > 0
