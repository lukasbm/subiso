"""Function-level convenience API, mirroring the historical matcher names.

These build a default :class:`~subiso.portfolio.Portfolio` per call; the
portfolio itself is the primary interface.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping

from .graphs import node_count
from .matcher import GraphLike, Query, UnsupportedOperation
from .portfolio import Portfolio
from .semantics import Semantics


def _lad():
    """The igraph-backed LAD module, imported only when automorphisms are asked for."""
    try:
        from .backends import lad
    except ImportError as error:  # igraph is an optional extra (GPL)
        raise UnsupportedOperation("automorphism helpers need the LAD backend: install subiso[lad].") from error
    return lad


def _query(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool,
    node_attr: str | None,
    edge_attr: str | None,
) -> Query:
    return Query(
        target=target,
        pattern=pattern,
        semantics=Semantics.INDUCED if induced else Semantics.ORDINARY,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )


_DEFAULT_PORTFOLIO: Portfolio | None = None


def default_portfolio() -> Portfolio:
    """Lazily initialized, shared :class:`~subiso.portfolio.Portfolio` for function calls."""
    global _DEFAULT_PORTFOLIO
    if _DEFAULT_PORTFOLIO is None:
        _DEFAULT_PORTFOLIO = Portfolio()
    return _DEFAULT_PORTFOLIO


def contains_subgraph(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool = False,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    time_limit: float | None = None,
) -> bool:
    """Whether *pattern* appears as a subgraph of *target*."""
    return default_portfolio().contains(
        _query(target, pattern, induced, node_attr, edge_attr), domains=domains, time_limit=time_limit
    )


def count_subgraph_positions(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool = False,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    cap: int | None = None,
) -> int:
    """Number of distinct vertex *sets* of *target* that host *pattern*.

    With ``cap`` set the result is ``min(true count, cap)``.
    """
    return default_portfolio().count(_query(target, pattern, induced, node_attr, edge_attr), cap=cap, domains=domains)


def count_embeddings(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool = False,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    cap: int | None = None,
) -> int:
    """Number of injective embeddings of *pattern* into *target* (not hosting sets).

    Requires an engine that natively reports embedding counts (Glasgow or the MIT suite).
    """
    return default_portfolio().count_embeddings(_query(target, pattern, induced, node_attr, edge_attr), cap=cap)


def get_subgraph_mappings(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool = False,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    domains: Mapping[Hashable, Collection[Hashable]] | None = None,
) -> list[dict[Hashable, Hashable]]:
    """Node mappings from *pattern* into *target*, keyed by pattern node."""
    return default_portfolio().mappings(_query(target, pattern, induced, node_attr, edge_attr), domains=domains)


def get_first_subgraph_mapping(
    target: GraphLike,
    pattern: GraphLike,
    induced: bool = False,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    time_limit: float | None = None,
) -> dict[Hashable, Hashable] | None:
    """Return a single node mapping from *pattern* into *target*, or ``None``."""
    return default_portfolio().first_mapping(
        _query(target, pattern, induced, node_attr, edge_attr), domains=domains, time_limit=time_limit
    )


def find_subgraph_occurrences(
    graph: GraphLike,
    pattern: GraphLike,
    *,
    induced: bool,
    max_occurrences: int | None = None,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> list[frozenset[Hashable]]:
    """Distinct vertex sets matching ``pattern`` under explicit containment semantics.

    A finite ``max_occurrences`` uses the lazy VF2 iterator and stops as soon as
    the cap is reached.
    """
    if max_occurrences is not None:
        assert max_occurrences >= 1, f"max_occurrences must be positive, got {max_occurrences}"
    return default_portfolio().occurrences(_query(graph, pattern, induced, node_attr, edge_attr), cap=max_occurrences)


def find_induced_occurrences(
    graph: GraphLike,
    pattern: GraphLike,
    *,
    max_occurrences: int | None = None,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> list[frozenset[Hashable]]:
    """Distinct induced occurrences as vertex sets."""
    return find_subgraph_occurrences(
        graph,
        pattern,
        induced=True,
        max_occurrences=max_occurrences,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )


def count_induced_occurrences_capped(graph: GraphLike, pattern: GraphLike, cap: int) -> int:
    """Number of distinct induced occurrences of *pattern*, counted up to *cap*."""
    return count_subgraph_positions(graph, pattern, induced=True, cap=cap)


def count_automorphisms(pattern: GraphLike) -> int:
    """``|Aut(pattern)|`` under topological isomorphism."""
    return _lad().automorphisms_of(pattern)


def automorphism_orbits(pattern: GraphLike) -> list[frozenset[Hashable]]:
    """Partition of the pattern's vertices into automorphism-group orbits."""
    return _lad().automorphism_orbits(pattern)


def are_isomorphic(
    left: GraphLike,
    right: GraphLike,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> bool:
    """Whether equal-order graphs are isomorphic, treating ``right`` as the pattern."""

    def order(graph: GraphLike) -> int:
        return node_count(graph)

    if order(left) != order(right):
        return False
    return contains_subgraph(left, right, induced=True, node_attr=node_attr, edge_attr=edge_attr)
