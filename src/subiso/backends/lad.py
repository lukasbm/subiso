"""igraph LAD backend.

Exact existence, counting and mapping enumeration for unlabelled graphs,
including empty and edge-less ones.  It materialises every embedding before
returning, so it must not be used for capped enumeration on large hosts with
large patterns; the portfolio sends those to VF2.

LAD additionally accepts per-pattern-role ``domains`` (allowed target vertices,
anchored matching) and a ``time_limit`` in seconds; both are igraph arguments
that VF2/VF3/FastISO cannot express, so anchored queries come here.
``automorphism_orbits`` exposes the automorphism group as a vertex partition.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping

import igraph as ig
import networkx as nx

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_query
from ..semantics import Semantics


def _no_limit(time_limit: float | None) -> float:
    return 0.0 if time_limit is None else float(time_limit)


def _ensure_igraph(graph: nx.Graph | ig.Graph) -> ig.Graph:
    if isinstance(graph, ig.Graph):
        return graph
    return ig.Graph.from_networkx(graph)


def _vertex_names(graph: ig.Graph) -> list:
    if "_nx_name" in graph.vs.attributes():
        return list(graph.vs["_nx_name"])
    return list(range(graph.vcount()))


def contains(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    *,
    induced: bool,
    domains: list[list[int]] | None = None,
    time_limit: float | None = None,
) -> bool:
    # igraph 1.0.0 rejects ``time_limit`` as a keyword and wants the four
    # positional arguments, so the call below is deliberately positional.
    return bool(
        _ensure_igraph(target).subisomorphic_lad(_ensure_igraph(pattern), domains, induced, _no_limit(time_limit))
    )


def count(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    *,
    induced: bool,
    domains: list[list[int]] | None = None,
) -> int:
    target_ig = _ensure_igraph(target)
    pattern_ig = _ensure_igraph(pattern)
    embeddings = target_ig.get_subisomorphisms_lad(pattern_ig, domains=domains, induced=induced)
    if not embeddings:
        return 0
    if induced and domains is None:
        # Every induced hosting set is stabilised by the full automorphism group.
        return len(embeddings) // automorphisms_of(pattern_ig)
    # Anchored induced or ordinary subgraph: hosting sets do not share a
    # stabiliser, so deduplicate images.
    return len({frozenset(embedding) for embedding in embeddings})


def occurrences(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    *,
    induced: bool,
    domains: list[list[int]] | None = None,
) -> list[frozenset[Hashable]]:
    seen: set[frozenset[Hashable]] = set()
    ordered: list[frozenset[Hashable]] = []
    for mapping in mappings(target, pattern, induced=induced, domains=domains):
        node_set = frozenset(mapping.values())
        if node_set not in seen:
            seen.add(node_set)
            ordered.append(node_set)
    return ordered


def mappings(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    *,
    induced: bool,
    domains: list[list[int]] | None = None,
) -> list[dict[Hashable, Hashable]]:
    target_ig = _ensure_igraph(target)
    pattern_ig = _ensure_igraph(pattern)
    target_names = _vertex_names(target_ig)
    pattern_names = _vertex_names(pattern_ig)
    embeddings = target_ig.get_subisomorphisms_lad(pattern_ig, domains=domains, induced=induced)
    return [
        {pattern_names[pattern_idx]: target_names[target_idx] for pattern_idx, target_idx in enumerate(embedding)}
        for embedding in embeddings
    ]


def first_mapping(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    *,
    induced: bool,
    domains: list[list[int]] | None = None,
    time_limit: float | None = None,
) -> dict[Hashable, Hashable] | None:
    target_ig = _ensure_igraph(target)
    pattern_ig = _ensure_igraph(pattern)
    # igraph 1.0.0 cannot combine a time limit with ``return_mapping``: the
    # limit bounds the existence answer, then a second, unbounded search
    # extracts the mapping (a solution is already known to exist).
    if time_limit is not None and not target_ig.subisomorphic_lad(pattern_ig, domains, induced, _no_limit(time_limit)):
        return None
    is_iso, embedding = target_ig.subisomorphic_lad(pattern_ig, domains=domains, induced=induced, return_mapping=True)
    if not is_iso:
        return None
    target_names = _vertex_names(target_ig)
    pattern_names = _vertex_names(pattern_ig)
    return {pattern_names[i]: target_names[embedding[i]] for i in range(len(embedding))}


def automorphisms_of(pattern: nx.Graph | ig.Graph) -> int:
    return int(_ensure_igraph(pattern).count_automorphisms_vf2())


def automorphism_orbits(pattern: nx.Graph | ig.Graph) -> list[frozenset[Hashable]]:
    """Vertices the automorphism group can map onto each other, as a partition."""
    pattern_ig = _ensure_igraph(pattern)
    parent = list(range(pattern_ig.vcount()))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for permutation in pattern_ig.get_automorphisms_vf2():
        for vertex, image in enumerate(permutation):
            union(vertex, image)

    groups: dict[int, list[int]] = {}
    for vertex in range(pattern_ig.vcount()):
        groups.setdefault(find(vertex), []).append(vertex)
    names = _vertex_names(pattern_ig)
    ordered = sorted(groups.values(), key=min)
    return [frozenset(names[vertex] for vertex in members) for members in ordered]


def positional_domains(
    target: nx.Graph | ig.Graph,
    pattern: nx.Graph | ig.Graph,
    domains: Mapping[Hashable, Collection[Hashable]] | None,
) -> list[list[int]] | None:
    """Translate ``{pattern_node: allowed_target_nodes}`` into igraph's positional domains.

    Pattern nodes absent from ``domains`` are unconstrained.  Allowed values are
    NetworkX node ids for NetworkX targets; for igraph targets they are vertex
    indices, since an igraph graph carries no id dictionary.
    """
    if domains is None:
        return None
    target_index: dict[Hashable, int] = (
        {vertex: vertex for vertex in range(target.vcount())}
        if isinstance(target, ig.Graph)
        else {node: index for index, node in enumerate(target.nodes())}
    )
    unrestricted = list(target_index.values())
    pattern_order: list[Hashable] = (
        list(range(pattern.vcount())) if isinstance(pattern, ig.Graph) else list(pattern.nodes())
    )
    return [
        [int(target_index[node]) for node in domains.get(pattern_node, unrestricted)] for pattern_node in pattern_order
    ]


class LadMatcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of this module."""

    name = "lad"
    capabilities = Capabilities(
        induced=True,
        ordinary=True,
        contains=True,
        first_mapping=True,
        mappings=True,
        occurrences=True,
        count=True,
        anchors=True,
        time_limit=True,
    )

    def available(self) -> bool:
        return True

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """Convert the host once; repeated queries reuse the igraph graph."""
        return PreparedTarget(self.name, _ensure_igraph(target), target)

    def automorphisms_of(self, pattern: nx.Graph | ig.Graph) -> int:
        """Automorphism group order |Aut(pattern)|."""
        return automorphisms_of(_ensure_igraph(pattern))

    def automorphism_orbits(self, pattern: nx.Graph | ig.Graph) -> list[frozenset[Hashable]]:
        """Partition of pattern vertices into automorphism orbits."""
        return automorphism_orbits(_ensure_igraph(pattern))

    def _check(self, query: Query, *, domains, time_limit, cap) -> None:
        require_semantics(self.capabilities, query.semantics, self.name)
        if query.labelled:
            raise UnsupportedOperation("LAD reads no attribute dictionaries; labelled queries are VF2-only.")
        if cap is not None:
            raise UnsupportedOperation("LAD materialises every embedding and cannot stop at a cap.")

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        self._check(query, domains=domains, time_limit=time_limit, cap=None)
        query = unpack_query(self.name, query)
        return contains(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            domains=positional_domains(query.target, query.pattern, domains),
            time_limit=time_limit,
        )

    def first_mapping(self, query, *, domains=None, time_limit=None):
        self._check(query, domains=domains, time_limit=time_limit, cap=None)
        query = unpack_query(self.name, query)
        return first_mapping(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            domains=positional_domains(query.target, query.pattern, domains),
            time_limit=time_limit,
        )

    def mappings(self, query, *, cap=None, domains=None):
        self._check(query, domains=domains, time_limit=None, cap=cap)
        query = unpack_query(self.name, query)
        return mappings(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            domains=positional_domains(query.target, query.pattern, domains),
        )

    def count(self, query, *, cap=None, domains=None) -> int:
        self._check(query, domains=domains, time_limit=None, cap=cap)
        query = unpack_query(self.name, query)
        return count(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            domains=positional_domains(query.target, query.pattern, domains),
        )

    def occurrences(self, query, *, cap=None, domains=None):
        self._check(query, domains=domains, time_limit=None, cap=cap)
        query = unpack_query(self.name, query)
        return occurrences(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            domains=positional_domains(query.target, query.pattern, domains),
        )
