"""Label semantics: equality, wildcards, and the FastISO colour path."""

from __future__ import annotations

import networkx as nx
import pytest

from subiso import count_subgraph_positions, get_first_subgraph_mapping


def test_labelled_first_mapping_respects_labels_and_wildcards():
    host: nx.Graph = nx.Graph([(0, 1), (1, 2)])
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C"}, "label")
    pattern: nx.Graph = nx.Graph([(10, 11), (11, 12)])
    nx.set_node_attributes(pattern, {10: "C", 11: "O", 12: "C"}, "label")

    mapping = get_first_subgraph_mapping(host, pattern, induced=True, node_attr="label")
    assert mapping is not None and mapping[11] == 1  # the only oxygen

    nx.set_node_attributes(pattern, {10: "C", 11: "C", 12: "C"}, "label")
    assert get_first_subgraph_mapping(host, pattern, induced=True, node_attr="label") is None
    assert get_first_subgraph_mapping(host, pattern, induced=True) is not None

    nx.set_node_attributes(pattern, {10: "C", 11: None, 12: "C"}, "label")
    wildcard = get_first_subgraph_mapping(host, pattern, induced=True, node_attr="label")
    assert wildcard is not None and host.nodes[wildcard[11]]["label"] == "O"


def _brute_labelled(
    host: nx.Graph,
    pattern: nx.Graph,
    induced: bool,
    node_attr: str | None,
    edge_attr: str | None,
) -> set[frozenset]:
    """Labelled hosting sets under VF2's wildcard semantics (``None`` matches anything)."""

    def node_match(left: dict, right: dict) -> bool:
        required = right.get(node_attr) if node_attr is not None else None
        return required is None or left.get(node_attr) == required

    def edge_match(left: dict, right: dict) -> bool:
        required = right.get(edge_attr) if edge_attr is not None else None
        return required is None or left.get(edge_attr) == required

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        host,
        pattern,
        node_match=node_match if node_attr is not None else None,
        edge_match=edge_match if edge_attr is not None else None,
    )
    iterator = matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()
    return {frozenset(mapping) for mapping in iterator}


def test_labelled_count_matches_bruteforce_with_and_without_wildcards():
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(pattern, {0: "C", 1: "O", 2: "C"}, "label")

    assert count_subgraph_positions(host, pattern, induced=True, node_attr="label") == len(
        _brute_labelled(host, pattern, True, "label", None)
    )
    nx.set_node_attributes(pattern, {0: "C", 1: None, 2: "C"}, "label")
    assert count_subgraph_positions(host, pattern, induced=True, node_attr="label") == len(
        _brute_labelled(host, pattern, True, "label", None)
    )


def test_edge_labelled_count_matches_bruteforce():
    host: nx.Graph = nx.Graph([(0, 1), (1, 2)])
    nx.set_edge_attributes(host, {(0, 1): "a", (1, 2): "b"}, "bond")
    pattern: nx.Graph = nx.Graph([(10, 11), (11, 12)])
    nx.set_edge_attributes(pattern, {(10, 11): "a", (11, 12): "b"}, "bond")

    assert count_subgraph_positions(host, pattern, induced=True, edge_attr="bond") == 1
    assert count_subgraph_positions(host, pattern, induced=True, edge_attr="bond") == len(
        _brute_labelled(host, pattern, True, None, "bond")
    )


def test_labelled_count_with_isolated_pattern_vertex_stays_exact():
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.Graph([(10, 11)])
    pattern.add_nodes_from([12, 13])
    nx.set_node_attributes(pattern, {10: "C", 11: "O", 12: "C", 13: "C"}, "label")

    # FastISO drops the labels of isolated pattern vertices, so this must take
    # the VF2 path and match brute force (regression: it once returned 0).
    assert count_subgraph_positions(host, pattern, induced=True, node_attr="label") == len(
        _brute_labelled(host, pattern, True, "label", None)
    )


def test_vf2pp_labelled_parity_and_wildcard_rejection():
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.path_graph(3)
    nx.set_node_attributes(pattern, {0: "C", 1: "O", 2: "C"}, "label")

    # VF2++ goes through the portfolio for plain functions; check the engine directly.
    from subiso.backends.vf2pp import Vf2ppMatcher
    from subiso.matcher import Query, UnsupportedOperation
    from subiso.semantics import Semantics

    matcher = Vf2ppMatcher()
    query = Query(host, pattern, Semantics.INDUCED, node_attr="label")
    expected = len(_brute_labelled(host, pattern, True, "label", None))
    assert matcher.count(query) == expected
    assert count_subgraph_positions(host, pattern, induced=True, node_attr="label") == expected

    nx.set_node_attributes(pattern, {0: "C", 1: None, 2: "C"}, "label")
    with pytest.raises(UnsupportedOperation, match="wildcards"):
        matcher.contains(Query(host, pattern, Semantics.INDUCED, node_attr="label"))


def test_edgeless_labelled_pattern_count_stays_exact():
    """Regression: the portfolio once fell back to VF2 here; Glasgow (when installed)
    or VF2 must both be exact, and no dispatch may lose the labels."""
    host: nx.Graph = nx.cycle_graph(5)
    nx.set_node_attributes(host, {0: "C", 1: "O", 2: "C", 3: "N", 4: "O"}, "label")
    pattern: nx.Graph = nx.Graph()
    pattern.add_nodes_from([10, 11])
    nx.set_node_attributes(pattern, {10: "C", 11: "O"}, "label")

    expected = len(_brute_labelled(host, pattern, True, "label", None))
    assert expected == 1  # induced edgeless pair: only the non-adjacent C-O pair qualifies
    assert count_subgraph_positions(host, pattern, induced=True, node_attr="label") == expected
