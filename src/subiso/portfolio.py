"""The router: one place that chooses an engine per query.

A :class:`Portfolio` is a :class:`~subiso.matcher.Matcher` itself, so callers
never branch on engines.  Dispatch is decided once per call from four inputs:
labelling (attributes are VF2-only), anchoring (``domains`` are LAD-only, a
``time_limit`` is honoured by LAD and Glasgow), capping (only VF2 and FastISO
can stop early; LAD materialises), and availability (VF3, FastISO and LAD are
optional builds).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from typing import cast

import networkx as nx

from .backends.fastiso import FastisoMatcher
from .backends.vf2 import Vf2Matcher
from .backends.vf2pp import Vf2ppMatcher
from .backends.vf3 import Vf3lMatcher, Vf3Matcher
from .capabilities import Capabilities
from .dispatch import DispatchPolicy, host_stats
from .graphs import has_edges, is_simple_undirected, node_count
from .matcher import BaseMatcher, GraphLike, Matcher, PreparedTarget, Query, UnsupportedOperation
from .semantics import Semantics


def _n_nodes(graph: GraphLike) -> int:
    return node_count(graph)


def _has_edges(graph: GraphLike) -> bool:
    return has_edges(graph)


def _compiled_usable(target: GraphLike, pattern: GraphLike) -> bool:
    """VF3 and FastISO reject empty and edge-less graphs; LAD handles those."""
    return is_simple_undirected(target) and is_simple_undirected(pattern) and _has_edges(target) and _has_edges(pattern)


def _require_networkx(*graphs: GraphLike) -> None:
    for graph in graphs:
        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "Label-aware matching needs networkx graphs: igraph carries no attribute "
                "dictionaries, so a label comparator has nothing to read."
            )


def _require_anchorable(
    query: Query,
    domains: Mapping[Hashable, Collection[Hashable]] | None,
    time_limit: float | None,
) -> bool:
    """Whether ``domains``/``time_limit`` anchor the query; attributes make that impossible."""
    if domains is None and time_limit is None:
        return False
    if query.labelled:
        raise ValueError(
            "domains and time_limit are unlabelled-engine features; attribute-aware matching uses VF2, "
            "which cannot express them."
        )
    return True


class Portfolio(BaseMatcher):
    """Dispatch across a fixed set of matchers; itself a :class:`Matcher`."""

    name = "portfolio"

    def __init__(
        self,
        matchers: Sequence[Matcher] | None = None,
        policy: Callable[[Query], Matcher] | None = None,
    ) -> None:
        defaults: list[Matcher] = [Vf2Matcher(), Vf2ppMatcher()]
        try:  # optional extra: igraph is GPL, so it is not a core dependency
            from .backends.lad import LadMatcher

            defaults.append(LadMatcher())
        except ImportError:
            pass
        defaults.extend([Vf3Matcher(), Vf3lMatcher(), FastisoMatcher()])
        try:  # optional separate distribution; the portfolio degrades without it
            from subiso_glasgow import GlasgowMatcher

            defaults.append(GlasgowMatcher())
        except ImportError:
            pass
        self._matchers: Sequence[Matcher] = list(matchers) if matchers is not None else defaults
        self._policy = policy if policy is not None else DispatchPolicy(self._matchers)
        combined = Capabilities()
        for matcher in self._matchers:
            if matcher.available():
                combined = combined.combine(matcher.capabilities)
        self.capabilities: Capabilities = combined

    def available(self) -> bool:
        return any(matcher.available() for matcher in self._matchers)

    # -- dispatch helpers -------------------------------------------------

    def _by_name(self, name: str) -> Matcher:
        for matcher in self._matchers:
            if matcher.name == name and matcher.available():
                return matcher
        raise UnsupportedOperation(f"{name} is not available (install the matching subiso extra).")

    def _labelled_or_python(self, compiled: str) -> Matcher:
        """A compiled engine when installed, else VF2, which handles every graph in Python."""
        if self._has_available(compiled):
            return self._by_name(compiled)
        return self._by_name("vf2")

    def _check_semantics(self, query: Query) -> None:
        if not self.capabilities.supports(query.semantics):
            raise UnsupportedOperation(f"no available matcher supports {query.semantics.value} matching.")

    def _candidates(
        self, query: Query, prepared: bool = False, host: tuple[float, float, float] | None = None
    ) -> list[Matcher]:
        if isinstance(self._policy, DispatchPolicy):
            return self._policy.candidates(query, prepared=prepared, host=host)
        if hasattr(self._policy, "candidates"):
            return self._policy.candidates(query)  # type: ignore[attr-defined]
        return [self._policy(query)]

    def _choose_count(self, query: Query, anchored: bool, cap: int | None) -> Matcher:
        if query.labelled:
            _require_networkx(query.target, query.pattern)
            if self._fastiso_usable(query):
                return self._by_name("fastiso")
            if self._glasgow_count_usable(query):
                return self._by_name("glasgow")
            return self._by_name("vf2")
        if anchored:
            return self._labelled_or_python("lad")
        if query.semantics is Semantics.INDUCED and self._fastiso_usable(query):
            return self._by_name("fastiso")
        if self._glasgow_count_usable(query):
            return self._by_name("glasgow")
        if cap is not None:
            _require_networkx(query.target, query.pattern)
            return self._by_name("vf2")
        return self._labelled_or_python("lad")

    def _choose_count_embeddings(self, query: Query) -> Matcher:
        """Embedding counts (not hosting sets): Glasgow, else the MIT suite."""
        if self._has_available("glasgow"):
            return self._by_name("glasgow")
        for matcher in self._matchers:
            if matcher.capabilities.count_embeddings and matcher.available():
                return matcher
        raise UnsupportedOperation("no available matcher reports embedding counts.")

    def _choose_timed(self, query: Query, domains) -> Matcher:
        """The engine for a query with a deadline: Glasgow unless per-node domains are given."""
        if domains is None and self._has_available("glasgow"):
            return self._by_name("glasgow")
        if self._has_available("lad"):
            return self._by_name("lad")
        raise UnsupportedOperation("time limits need the Glasgow or LAD engine; install subiso[lad].")

    def _choose_mappings(self, query: Query, anchored: bool) -> Matcher:
        if query.labelled:
            _require_networkx(query.target, query.pattern)
            return self._by_name("vf2")
        if anchored:
            return self._labelled_or_python("lad")
        if _compiled_usable(query.target, query.pattern) and self._has_available("vf3"):
            return self._by_name("vf3")
        return self._labelled_or_python("lad")

    def _choose_occurrences(self, query: Query, cap: int | None) -> Matcher:
        if cap is not None or query.labelled:
            _require_networkx(query.target, query.pattern)
            return self._by_name("vf2")
        if _compiled_usable(query.target, query.pattern) and self._has_available("vf3"):
            return self._by_name("vf3")
        return self._labelled_or_python("lad")

    def _has_available(self, name: str) -> bool:
        return any(matcher.name == name and matcher.available() for matcher in self._matchers)

    def _fastiso_usable(self, query: Query) -> bool:
        """The same gates the FastISO matcher applies, asked before dispatching."""
        if not self._has_available("fastiso") or query.semantics is not Semantics.INDUCED:
            return False
        return self._by_name("fastiso")._gates_open(query)  # type: ignore[attr-defined]

    def _glasgow_count_usable(self, query: Query) -> bool:
        """Glasgow counts both relations and accepts edgeless patterns and isolated vertices."""
        if not self._has_available("glasgow") or query.edge_attr is not None:
            return False
        for graph in (query.target, query.pattern):
            if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
                return False
        if query.node_attr is not None:
            for graph in (query.target, query.pattern):
                if any(graph.nodes[node].get(query.node_attr) is None for node in graph):
                    return False  # wildcards or missing labels are not expressible
        return True

    def prepare(self, target, *, node_attr=None, edge_attr=None, lazy: bool = True) -> PreparedTarget:
        """Prepare a host for repeated queries; converts per engine on first use when ``lazy=True``."""
        artifacts: dict = {}
        if not lazy:
            for matcher in self._matchers:
                if matcher.available():
                    try:
                        artifacts[(matcher.name, node_attr, edge_attr)] = matcher.prepare(
                            target, node_attr=node_attr, edge_attr=edge_attr
                        )
                    except UnsupportedOperation:
                        continue
        return PreparedTarget(self.name, artifacts, target, host_stats(target))

    def _route(
        self, query: Query[GraphLike | PreparedTarget]
    ) -> tuple[Query[GraphLike], dict | None, tuple[float, float, float] | None]:
        """Split a query into the engine-side query, prepared artifacts, and cached host stats."""
        target = query.target
        if isinstance(target, PreparedTarget):
            own = target.engine == self.name
            raw = Query(target.source, query.pattern, query.semantics, query.node_attr, query.edge_attr)
            return raw, (cast(dict, target.artifact) if own else None), (target.host_stats if own else None)
        return query, None, None

    def _call_query(
        self, matcher: Matcher, query: Query[GraphLike], prepared: dict | None
    ) -> Query[GraphLike | PreparedTarget]:
        if prepared is None:
            return query
        key = (matcher.name, query.node_attr, query.edge_attr)
        artifact = prepared.get(key) or prepared.get(matcher.name)
        if artifact is None:
            artifact = matcher.prepare(query.target, node_attr=query.node_attr, edge_attr=query.edge_attr)
            prepared[key] = artifact
        return Query(artifact, query.pattern, query.semantics, query.node_attr, query.edge_attr)

    # -- Matcher protocol --------------------------------------------------

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query, prepared, host = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return True
        if _n_nodes(query.target) < _n_nodes(query.pattern):
            return False
        anchored = _require_anchorable(query, domains, time_limit)
        if anchored:
            candidates = [self._choose_timed(query, domains)]
        else:
            candidates = self._candidates(query, prepared is not None, host)
        last_error: Exception | None = None
        for matcher in candidates:
            try:
                return matcher.contains(
                    self._call_query(matcher, query, prepared), domains=domains, time_limit=time_limit
                )
            except UnsupportedOperation as error:
                last_error = error
        raise last_error or UnsupportedOperation("no available matcher")

    def first_mapping(self, query, *, domains=None, time_limit=None):
        query, prepared, host = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return {}
        if _n_nodes(query.target) < _n_nodes(query.pattern):
            return None
        anchored = _require_anchorable(query, domains, time_limit)
        if anchored:
            candidates = [self._choose_timed(query, domains)]
        else:
            candidates = self._candidates(query, prepared is not None, host)
        last_error: Exception | None = None
        for matcher in candidates:
            try:
                return matcher.first_mapping(
                    self._call_query(matcher, query, prepared), domains=domains, time_limit=time_limit
                )
            except UnsupportedOperation as error:
                last_error = error
        raise last_error or UnsupportedOperation("no available matcher")

    def mappings(self, query, *, cap=None, domains=None):
        query, prepared, _ = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return [{}]
        if _n_nodes(query.target) < _n_nodes(query.pattern):
            return []
        anchored = _require_anchorable(query, domains, None)
        matcher = self._choose_mappings(query, anchored)
        if cap is not None and matcher.name in ("lad", "vf3"):
            matcher = self._by_name("vf2")  # only VF2 can stop early
        return matcher.mappings(self._call_query(matcher, query, prepared), cap=cap, domains=domains)

    def count(self, query, *, cap=None, domains=None) -> int:
        assert cap is None or cap >= 1, f"cap must be at least 1, got {cap}"
        query, prepared, _ = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return 1
        if _n_nodes(query.target) < _n_nodes(query.pattern):
            return 0
        anchored = _require_anchorable(query, domains, None)
        if anchored and cap is not None:
            raise ValueError("cap is not supported for anchored counting: LAD materialises every embedding.")
        matcher = self._choose_count(query, anchored, cap)
        return matcher.count(self._call_query(matcher, query, prepared), cap=cap, domains=domains)

    def count_embeddings(self, query, *, cap: int | None = None) -> int:
        """Embedding counts, the count semantics of the MIT suite and Glasgow.

        Unlike :meth:`count` (distinct vertex sets), this counts mappings, i.e.
        embeddings: ``count = embeddings / |Aut(pattern)|`` for induced
        matching.  Only engines that declare ``count_embeddings`` serve it.
        """
        assert cap is None or cap >= 1, f"cap must be at least 1, got {cap}"
        query, prepared, _ = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return 1
        if _n_nodes(query.target) < _n_nodes(query.pattern):
            return 0
        matcher = self._choose_count_embeddings(query)
        result = matcher.count_embeddings(self._call_query(matcher, query, prepared))  # type: ignore[attr-defined]
        return result if cap is None else min(result, cap)

    def occurrences(self, query, *, cap=None, domains=None):
        query, prepared, _ = self._route(query)
        self._check_semantics(query)
        if _n_nodes(query.pattern) == 0:
            return [frozenset()]
        anchored = _require_anchorable(query, domains, None)
        if anchored:
            if cap is not None:
                raise ValueError("cap is not supported for anchored enumeration: LAD materialises every embedding.")
            matcher = self._choose_timed(query, domains)
            return matcher.occurrences(self._call_query(matcher, query, prepared), cap=None, domains=domains)
        matcher = self._choose_occurrences(query, cap)
        return matcher.occurrences(self._call_query(matcher, query, prepared), cap=cap, domains=domains)
