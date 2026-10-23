"""VF3 (``vf3py``) backend.

Fast existence and mapping enumeration for unlabelled graphs.  VF3 rejects
empty and edge-less graphs, so the portfolio only reaches it through
``compiled_usable``; it cannot match attributes and cannot count hosting sets.
Optional: the module imports cleanly and :func:`available` reports whether the
build is present.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import networkx as nx

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_query
from ..semantics import Semantics

try:
    import vf3py as _vf3py
except Exception:  # pragma: no cover - optional backend
    _vf3py = None


Vf3Variant = Literal["B", "L", "P"]


def available() -> bool:
    return _vf3py is not None


def contains(target, pattern, *, induced: bool, variant: Vf3Variant = "B", num_threads: int = 1) -> bool:
    assert _vf3py is not None, "VF3 is not importable."
    matcher = _vf3py.has_subgraph if induced else _vf3py.has_monomorphic_subgraph
    return bool(matcher(pattern, target, variant=variant, num_threads=num_threads))


def mappings(
    target, pattern, *, induced: bool, variant: Vf3Variant = "B", num_threads: int = 1
) -> list[dict[Hashable, Hashable]]:
    assert _vf3py is not None, "VF3 is not importable."
    enumerator = _vf3py.get_subgraph_isomorphisms if induced else _vf3py.get_subgraph_monomorphisms
    return list(enumerator(pattern, target, variant=variant, num_threads=num_threads))


def first_mapping(
    target, pattern, *, induced: bool, variant: Vf3Variant = "B", num_threads: int = 1
) -> dict[Hashable, Hashable] | None:
    assert _vf3py is not None, "VF3 is not importable."
    matcher = _vf3py.has_subgraph if induced else _vf3py.has_monomorphic_subgraph
    matched, mapping = matcher(pattern, target, get_mapping=True, variant=variant, num_threads=num_threads)  # type: ignore
    return dict(mapping) if matched and mapping is not None else None  # type: ignore[arg-type]


def compiled_usable(target: nx.Graph, pattern: nx.Graph | None = None) -> bool:
    """VF3 rejects edge-less graphs; the portfolio gates on this."""
    return target.number_of_edges() > 0 and (pattern is None or pattern.number_of_edges() > 0)


def _applicable(target, pattern) -> bool:
    """VF3 accepts simple, undirected, edge-carrying NetworkX graphs only."""
    return (
        isinstance(target, nx.Graph)
        and isinstance(pattern, nx.Graph)
        and not target.is_directed()
        and not pattern.is_directed()
        and not target.is_multigraph()
        and not pattern.is_multigraph()
        and compiled_usable(target, pattern)
    )


class Vf3Matcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of this module.

    ``variant`` selects the upstream engine: ``"B"`` (base), ``"L"`` (labelled
    workloads) and ``"P"`` (parallel, requires ``num_threads > 1``).  The L
    variant is exercised here on the same unlabelled semantics as B: vf3py's
    attribute handling goes through Python callables that are not wired for
    the wildcard label model.
    """

    capabilities = Capabilities(
        induced=True,
        ordinary=True,
        contains=True,
        first_mapping=True,
        mappings=True,
        occurrences=True,
    )

    def __init__(self, variant: Vf3Variant = "B", num_threads: int = 1) -> None:
        assert variant in ("B", "L", "P"), f"unknown VF3 variant {variant!r}"
        if variant == "P":
            assert num_threads > 1, "the parallel variant needs num_threads > 1"
        self._variant: Vf3Variant = variant
        self._num_threads = num_threads
        self.name = "vf3" if variant == "B" else f"vf3{variant.lower()}"

    def available(self) -> bool:
        return available()

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """This engine converts inside its calls, so preparation is a no-op holder."""
        return PreparedTarget(self.name, target, target)

    def _check(self, query: Query, *, domains, time_limit, cap) -> None:
        require_semantics(self.capabilities, query.semantics, self.name)
        if query.labelled:
            raise UnsupportedOperation("VF3 cannot match attributes.")
        if domains is not None or time_limit is not None:
            raise UnsupportedOperation("VF3 cannot express domains or time limits.")
        if cap is not None:
            raise UnsupportedOperation("VF3 enumerates the complete result and cannot stop at a cap.")
        if not _applicable(query.target, query.pattern):
            raise UnsupportedOperation("VF3 rejects edge-less, directed, or multigraph inputs.")

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit, cap=None)
        return contains(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            variant=self._variant,
            num_threads=self._num_threads,
        )

    def first_mapping(self, query, *, domains=None, time_limit=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=time_limit, cap=None)
        return first_mapping(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            variant=self._variant,
            num_threads=self._num_threads,
        )

    def mappings(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None, cap=cap)
        return mappings(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            variant=self._variant,
            num_threads=self._num_threads,
        )

    def occurrences(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        self._check(query, domains=domains, time_limit=None, cap=cap)
        found = mappings(
            query.target,
            query.pattern,
            induced=query.semantics is Semantics.INDUCED,
            variant=self._variant,
            num_threads=self._num_threads,
        )
        seen: set[frozenset] = set()
        ordered: list[frozenset] = []
        for mapping in found:
            node_set = frozenset(mapping.values())
            if node_set not in seen:
                seen.add(node_set)
                ordered.append(node_set)
        return ordered

    def count(self, query, *, cap=None, domains=None) -> int:
        raise UnsupportedOperation("VF3 enumerates mappings; counting is LAD/FastISO territory.")


class Vf3lMatcher(Vf3Matcher):
    """VF3's labelled-graph variant, run with the same unlabelled contract as base VF3."""

    def __init__(self) -> None:
        super().__init__(variant="L")


class Vf3pMatcher(Vf3Matcher):
    """VF3's parallel variant; pass ``num_threads`` explicitly (default 4)."""

    def __init__(self, num_threads: int = 4) -> None:
        super().__init__(variant="P", num_threads=num_threads)
