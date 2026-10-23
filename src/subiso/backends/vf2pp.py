"""NetworkX VF2++ backend.

Pure Python like VF2, but with the VF2++ matching order and cutting rules.
Supports induced and ordinary matching and node labels as equality classes;
NetworkX yields host -> pattern mappings, inverted here to pattern -> host.
No wildcards: a pattern label of ``None`` is rejected rather than silently
compared against the default label.
"""

# Pyright's bundled networkx stubs predate the vf2pp subgraph/monomorphism
# entry points; the runtime module (networkx >= 3.4) provides them.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
from networkx.algorithms.isomorphism.vf2pp import (
    vf2pp_all_monomorphisms,
    vf2pp_all_subgraph_isomorphisms,
    vf2pp_is_monomorphic,
    vf2pp_monomorphism,
    vf2pp_subgraph_is_isomorphic,
    vf2pp_subgraph_isomorphism,
)

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_query
from ..semantics import Semantics


def _pattern_first(mapping: dict) -> dict[Hashable, Hashable]:
    # NetworkX yields host -> pattern, like VF2.
    return {pattern_node: target_node for target_node, pattern_node in mapping.items()}


def _has_wildcards(pattern: nx.Graph, node_attr: str | None) -> bool:
    return node_attr is not None and any(pattern.nodes[node].get(node_attr) is None for node in pattern)


def contains(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
) -> bool:
    if pattern.number_of_nodes() == 0:
        return True
    checker = vf2pp_subgraph_is_isomorphic if induced else vf2pp_is_monomorphic
    return bool(checker(target, pattern, node_label=node_attr))


def first_mapping(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
) -> dict[Hashable, Hashable] | None:
    if pattern.number_of_nodes() == 0:
        return {}
    finder = vf2pp_subgraph_isomorphism if induced else vf2pp_monomorphism
    found = finder(target, pattern, node_label=node_attr)
    return None if found is None else _pattern_first(found)


def mappings(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    node_attr: str | None = None,
) -> list[dict[Hashable, Hashable]]:
    if pattern.number_of_nodes() == 0:
        return [{}]
    enumerator = vf2pp_all_subgraph_isomorphisms if induced else vf2pp_all_monomorphisms
    return [_pattern_first(mapping) for mapping in enumerator(target, pattern, node_label=node_attr)]


def occurrences(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    cap: int | None = None,
    node_attr: str | None = None,
) -> list[frozenset[Hashable]]:
    seen: set[frozenset[Hashable]] = set()
    ordered: list[frozenset[Hashable]] = []
    for mapping in mappings(target, pattern, induced=induced, node_attr=node_attr):
        image = frozenset(mapping.values())
        if image in seen:
            continue
        seen.add(image)
        ordered.append(image)
        if cap is not None and len(ordered) >= cap:
            break
    return ordered


def count(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    induced: bool,
    cap: int | None = None,
    node_attr: str | None = None,
) -> int:
    if cap is not None:
        return len(occurrences(target, pattern, induced=induced, cap=cap, node_attr=node_attr))
    return len(occurrences(target, pattern, induced=induced, node_attr=node_attr))


class Vf2ppMatcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of NetworkX VF2++."""

    name = "vf2pp"
    capabilities = Capabilities(
        induced=True,
        ordinary=True,
        node_labels=True,
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

    def _check(self, query: Query, *, domains, time_limit) -> None:
        require_semantics(self.capabilities, query.semantics, self.name)
        if domains is not None or time_limit is not None:
            raise UnsupportedOperation("VF2++ cannot express domains or time limits.")
        if query.edge_attr is not None:
            raise UnsupportedOperation("VF2++ matches node labels only.")
        if not isinstance(query.target, nx.Graph) or not isinstance(query.pattern, nx.Graph):
            raise UnsupportedOperation("VF2++ reads NetworkX graphs.")
        if query.target.is_directed() or query.pattern.is_directed():
            raise UnsupportedOperation("VF2++ here handles simple undirected graphs only.")
        if query.target.is_multigraph() or query.pattern.is_multigraph():
            raise UnsupportedOperation("VF2++ here handles simple graphs only.")
        if _has_wildcards(query.pattern, query.node_attr):
            raise UnsupportedOperation("VF2++ labels are equality classes; None wildcards are not expressible.")

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit)
        return contains(
            query.target, query.pattern, induced=query.semantics is Semantics.INDUCED, node_attr=query.node_attr
        )

    def first_mapping(self, query, *, domains=None, time_limit=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit)
        return first_mapping(
            query.target, query.pattern, induced=query.semantics is Semantics.INDUCED, node_attr=query.node_attr
        )

    def mappings(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        found = mappings(
            query.target, query.pattern, induced=query.semantics is Semantics.INDUCED, node_attr=query.node_attr
        )
        return found if cap is None else found[:cap]

    def occurrences(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        return occurrences(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            cap=cap,
            node_attr=query.node_attr,
        )

    def count(self, query, *, cap=None, domains=None) -> int:
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None)
        return count(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            cap=cap,
            node_attr=query.node_attr,
        )
