"""Glasgow subgraph solver backend for subiso.

Glasgow is a constraint-programming solver by Ciaran McCreesh and collaborators
(MIT licence).  This package wraps the C++ library in-process with pybind11 and
implements the :class:`subiso.matcher.Matcher` protocol.

v0 scope: simple undirected graphs, vertex labels as equality classes (no
wildcards), induced and ordinary matching, existence, first mapping, mapping
enumeration, counts, and a coarse timeout.  The fast induced count divides the
embedding count by ``|Aut(pattern)|``; ordinary counts deduplicate images.

``prepare`` converts the host once in C++ (``PreparedGraph``); repeated queries
against the same host then skip graph construction entirely.
"""

from __future__ import annotations

from typing import cast

import networkx as nx

from subiso.capabilities import Capabilities
from subiso.matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_target
from subiso.semantics import Semantics

from . import _glasgow


def _automorphism_count(pattern: nx.Graph, node_attr: str | None) -> int:
    if node_attr is None:
        from subiso.backends import lad

        return lad.automorphisms_of(pattern)
    from subiso.backends import vf2

    return vf2.automorphism_count(pattern, node_attr=node_attr)


class GlasgowMatcher(BaseMatcher):
    """The :class:`subiso.matcher.Matcher` view of the Glasgow solver."""

    name = "glasgow"
    capabilities = Capabilities(
        induced=True,
        ordinary=True,
        node_labels=True,
        contains=True,
        first_mapping=True,
        mappings=True,
        occurrences=True,
        count=True,
        count_capped=True,
        count_embeddings=True,
        time_limit=True,
    )

    def available(self) -> bool:
        return True

    # -- graph preparation -------------------------------------------------

    def _graph_parts(self, graph: nx.Graph, node_attr: str | None, *, side: str):
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise UnsupportedOperation("the Glasgow shim reads simple undirected NetworkX graphs.")
        nodes = list(graph.nodes())
        index = {node: position for position, node in enumerate(nodes)}
        edges = [(index[left], index[right]) for left, right in graph.edges()]
        labels: list[str | None] = [None] * len(nodes)
        if node_attr is not None:
            for node in nodes:
                label = graph.nodes[node].get(node_attr)
                if label is None:
                    explanation = (
                        "wildcards are not expressible"
                        if side == "pattern"
                        else "the target is missing labels the pattern requires"
                    )
                    raise UnsupportedOperation(f"Glasgow labels are equality classes; {explanation}.")
                labels[index[node]] = str(label)
        return nodes, edges, labels

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """Convert the host once in C++; repeated queries reuse it."""
        if edge_attr is not None:
            raise UnsupportedOperation("edge labels are not wired through the Glasgow shim yet.")
        nodes, edges, labels = self._graph_parts(target, node_attr, side="target")
        return PreparedTarget(self.name, _glasgow.prepare_graph(len(nodes), edges, labels), target)

    def _target_for_call(self, query: Query) -> _glasgow.PreparedGraph:
        """Prepared host for this query; converts on the fly if the caller did not prepare."""
        target, was_prepared = unpack_target(self.name, query.target)
        if was_prepared:
            return cast(_glasgow.PreparedGraph, target)
        nodes, edges, labels = self._graph_parts(cast(nx.Graph, target), query.node_attr, side="target")
        return _glasgow.prepare_graph(len(nodes), edges, labels)

    def _target_nodes(self, query: Query) -> list:
        if isinstance(query.target, PreparedTarget) and query.target.engine == self.name:
            return list(cast(nx.Graph, query.target.source).nodes())
        return [node for node in cast(nx.Graph, query.target)]

    def _check(self, query: Query, *, domains, cap: int | None) -> None:
        require_semantics(self.capabilities, query.semantics, self.name)
        if domains is not None:
            raise UnsupportedOperation("Glasgow cannot express per-pattern-node domains.")
        if query.edge_attr is not None:
            raise UnsupportedOperation("edge labels are not wired through the Glasgow shim yet.")
        if cap is not None and cap < 1:
            raise UnsupportedOperation(f"cap must be at least 1, got {cap}")

    # -- driver ------------------------------------------------------------

    def _pattern(self, query: Query):
        _, edges, labels = self._graph_parts(query.pattern, query.node_attr, side="pattern")
        return len(labels), edges, labels

    def _find(self, query: Query, timeout: float):
        pattern_n, pattern_edges, pattern_labels = self._pattern(query)
        return _glasgow.find_mapping_prepared(
            self._target_for_call(query),
            pattern_n,
            pattern_edges,
            pattern_labels,
            query.semantics is Semantics.INDUCED,
            timeout,
        )

    def _count_embeddings(self, query: Query, timeout: float) -> int:
        pattern_n, pattern_edges, pattern_labels = self._pattern(query)
        return int(
            _glasgow.count_embeddings_prepared(
                self._target_for_call(query),
                pattern_n,
                pattern_edges,
                pattern_labels,
                query.semantics is Semantics.INDUCED,
                timeout,
            )
        )

    def _count_images(self, query: Query, limit: int) -> int:
        pattern_n, pattern_edges, pattern_labels = self._pattern(query)
        return int(
            _glasgow.count_images_prepared(
                self._target_for_call(query),
                pattern_n,
                pattern_edges,
                pattern_labels,
                query.semantics is Semantics.INDUCED,
                0.0,
                limit,
            )
        )

    def _enumerate(self, query: Query, limit: int) -> list[dict[int, int]]:
        pattern_n, pattern_edges, pattern_labels = self._pattern(query)
        raw = _glasgow.enumerate_mappings_prepared(
            self._target_for_call(query),
            pattern_n,
            pattern_edges,
            pattern_labels,
            query.semantics is Semantics.INDUCED,
            0.0,
            limit,
        )
        return [dict(mapping) for mapping in raw]

    # -- Matcher protocol --------------------------------------------------

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        self._check(query, domains=domains, cap=None)
        return self._find(query, time_limit or 0.0) is not None

    def first_mapping(self, query, *, domains=None, time_limit=None):
        self._check(query, domains=domains, cap=None)
        found = self._find(query, time_limit or 0.0)
        if found is None:
            return None
        target_nodes = self._target_nodes(query)
        pattern_nodes = list(query.pattern.nodes())
        return {
            pattern_nodes[pattern_index]: target_nodes[target_index] for pattern_index, target_index in found.items()
        }

    def mappings(self, query, *, cap=None, domains=None):
        self._check(query, domains=domains, cap=cap)
        raw = self._enumerate(query, 0 if cap is None else cap)
        target_nodes = self._target_nodes(query)
        pattern_nodes = list(query.pattern.nodes())
        return [
            {
                pattern_nodes[pattern_index]: target_nodes[target_index]
                for pattern_index, target_index in mapping.items()
            }
            for mapping in raw
        ]

    def occurrences(self, query, *, cap=None, domains=None):
        self._check(query, domains=domains, cap=cap)
        mappings = self.mappings(query, cap=cap, domains=domains)
        seen: set[frozenset] = set()
        ordered: list[frozenset] = []
        for mapping in mappings:
            image = frozenset(mapping.values())
            if image not in seen:
                seen.add(image)
                ordered.append(image)
            if cap is not None and len(ordered) >= cap:
                break
        return ordered

    def count(self, query, *, cap=None, domains=None) -> int:
        self._check(query, domains=domains, cap=cap)
        if query.semantics is Semantics.INDUCED:
            # Exact: embedding count divided by |Aut(pattern)|.
            embeddings = self._count_embeddings(query, 0.0)
            if embeddings == 0:
                return 0
            count = embeddings // _automorphism_count(query.pattern, query.node_attr)
            return count if cap is None else min(count, cap)
        # Ordinary matching: count distinct images in the C++ callback.
        return self._count_images(query, 0 if cap is None else cap)

    def count_embeddings(self, query: Query, *, time_limit: float | None = None) -> int:
        """Raw injective embedding count, without dividing by automorphisms."""
        self._check(query, domains=None, cap=None)
        return self._count_embeddings(query, time_limit or 0.0)

    def count_images(self, query: Query, *, cap: int | None = None) -> int:
        """Count distinct image vertex sets directly via C++ callback."""
        self._check(query, domains=None, cap=cap)
        return self._count_images(query, 0 if cap is None else cap)
