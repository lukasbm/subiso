"""NetworkX VF2 backend.

VF2 is the only matcher here that reads node/edge attributes and the only
occurrence iterator that can be stopped at a cap without materialising every
embedding.  It accepts every graph, including empty and edge-less ones.

Wildcards: a pattern vertex whose label is ``None`` (or absent) matches *any*
target vertex.  The pattern side declares what it requires; the target side
simply has whatever it has.
"""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_query
from ..semantics import Semantics


def _label_matcher(attribute: str):
    """Comparator for VF2.  ``data2`` is the *pattern* side; ``None`` is a wildcard."""

    def match(data1: dict, data2: dict) -> bool:
        required = data2.get(attribute)
        return required is None or data1.get(attribute) == required

    return match


def _matcher(
    target: nx.Graph,
    pattern: nx.Graph,
    node_attr: str | None,
    edge_attr: str | None,
) -> nx.algorithms.isomorphism.GraphMatcher:
    return nx.algorithms.isomorphism.GraphMatcher(
        target,
        pattern,
        node_match=_label_matcher(node_attr) if node_attr else None,
        edge_match=_label_matcher(edge_attr) if edge_attr else None,
    )


def _iterator(matcher: nx.algorithms.isomorphism.GraphMatcher, induced: bool):
    return matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()


def _single_node_contains(target: nx.Graph, pattern: nx.Graph, node_attr: str | None) -> bool:
    if node_attr is None:
        return target.number_of_nodes() > 0
    pattern_node = next(iter(pattern.nodes()))
    required = pattern.nodes[pattern_node].get(node_attr)
    return any(required is None or data.get(node_attr) == required for _, data in target.nodes(data=True))


def contains(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> bool:
    if pattern.number_of_nodes() == 1:
        return _single_node_contains(target, pattern, node_attr)
    matcher = _matcher(target, pattern, node_attr, edge_attr)
    return matcher.subgraph_is_isomorphic() if induced else matcher.subgraph_is_monomorphic()


def count(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> int:
    matcher = _matcher(target, pattern, node_attr, edge_attr)
    return len({frozenset(mapping) for mapping in _iterator(matcher, induced)})


def occurrences(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    cap: int | None,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> list[frozenset[Hashable]]:
    matcher = _matcher(target, pattern, node_attr, edge_attr)
    seen: set[frozenset[Hashable]] = set()
    ordered: list[frozenset[Hashable]] = []
    for mapping in _iterator(matcher, induced):
        node_set = frozenset(mapping)  # VF2 yields target -> pattern; keys are target nodes.
        if node_set in seen:
            continue
        seen.add(node_set)
        ordered.append(node_set)
        if cap is not None and len(ordered) >= cap:
            break
    return ordered


def mappings(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> list[dict[Hashable, Hashable]]:
    matcher = _matcher(target, pattern, node_attr, edge_attr)
    # VF2 yields target -> pattern; every caller assumes pattern -> target.
    return [
        {pattern_node: target_node for target_node, pattern_node in mapping.items()}
        for mapping in _iterator(matcher, induced)
    ]


def first_mapping(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> dict[Hashable, Hashable] | None:
    matcher = _matcher(target, pattern, node_attr, edge_attr)
    for mapping in _iterator(matcher, induced):
        return {pattern_node: target_node for target_node, pattern_node in mapping.items()}
    return None


def automorphism_count(
    pattern: nx.Graph,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> int:
    """``|Aut(pattern)|`` under this backend's label semantics (wildcards included)."""
    return sum(1 for _ in _matcher(pattern, pattern, node_attr, edge_attr).isomorphisms_iter())


class Vf2Matcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of this module."""

    name = "vf2"
    capabilities = Capabilities(
        induced=True,
        ordinary=True,
        node_labels=True,
        edge_labels=True,
        wildcards=True,
        contains=True,
        first_mapping=True,
        mappings=True,
        occurrences=True,
        count=True,
    )

    def available(self) -> bool:
        return True

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """This engine converts inside its calls, so preparation is a no-op holder."""
        return PreparedTarget(self.name, target, target)

    def automorphism_count(self, pattern: nx.Graph, *, node_attr: str | None = None) -> int:
        """Count topological (or label-preserving) automorphisms of *pattern*."""
        return automorphism_count(pattern, node_attr=node_attr)

    def _check(self, query: Query, *, domains, time_limit) -> None:
        require_semantics(self.capabilities, query.semantics, self.name)
        if domains is not None or time_limit is not None:
            raise UnsupportedOperation("VF2 cannot express domains or time limits; those are LAD-only.")
        if not isinstance(query.target, nx.Graph) or not isinstance(query.pattern, nx.Graph):
            raise UnsupportedOperation("VF2 reads NetworkX graphs.")

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit)
        return contains(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            node_attr=query.node_attr,
            edge_attr=query.edge_attr,
        )

    def first_mapping(self, query, *, domains=None, time_limit=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit)
        return first_mapping(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            node_attr=query.node_attr,
            edge_attr=query.edge_attr,
        )

    def mappings(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        result = mappings(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            node_attr=query.node_attr,
            edge_attr=query.edge_attr,
        )
        return result if cap is None else result[:cap]

    def count(self, query, *, cap=None, domains=None) -> int:
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        if cap is not None:
            return len(
                occurrences(
                    query.target,
                    query.pattern,
                    induced=query.semantics is Semantics.INDUCED,
                    cap=cap,
                    node_attr=query.node_attr,
                    edge_attr=query.edge_attr,
                )
            )
        return count(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            node_attr=query.node_attr,
            edge_attr=query.edge_attr,
        )

    def occurrences(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        return occurrences(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            cap=cap,
            node_attr=query.node_attr,
            edge_attr=query.edge_attr,
        )
