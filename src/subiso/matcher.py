"""The one interface every engine implements, and the router too.

Backends differ in graphs, labels, operations and limits; a :class:`Query`
says *what* is asked and each :class:`Matcher` declares in
:class:`~subiso.capabilities.Capabilities` what it can answer.  Requests a
matcher cannot express raise :class:`UnsupportedOperation` -- never a silent
fallback to different semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from .capabilities import Capabilities
from .graphs import GraphLike
from .semantics import Semantics


class UnsupportedOperation(Exception):
    """The chosen matcher cannot express the requested query."""


class MatcherCrash(RuntimeError):
    """A backend process died with a native crash (negative return code)."""


@dataclass(frozen=True)
class Query[T]:
    """A matching request: two graphs, semantics, and label attribute names.

    ``target`` may also be a :class:`PreparedTarget`, letting a caller convert
    a host once and reuse it across many queries.
    """

    target: T
    pattern: GraphLike
    semantics: Semantics = Semantics.INDUCED
    node_attr: str | None = None
    edge_attr: str | None = None

    @property
    def labelled(self) -> bool:
        return self.node_attr is not None or self.edge_attr is not None


@dataclass(frozen=True)
class PreparedTarget:
    """A host converted once for one engine (or a portfolio of them).

    ``artifact`` is engine-specific and opaque; ``source`` keeps the original
    graph so any matcher that cannot use the artifact can fall back.
    ``host_stats`` caches the O(1) dispatch features ``(nodes, edges, density)``
    when the portfolio prepared the host, so dispatch does not rescan it.
    """

    engine: str
    artifact: object
    source: GraphLike
    host_stats: tuple[float, float, float] | None = None


def unpack_target(engine: str, target: GraphLike | PreparedTarget) -> tuple[object, bool]:
    """Return ``(graph, was_prepared)`` for ``engine``; foreign artifacts fall back to the source."""
    if isinstance(target, PreparedTarget):
        if target.engine == engine:
            return target.artifact, True
        return target.source, False
    return target, False


def unpack_query(engine: str, query: Query[GraphLike | PreparedTarget]) -> Query[GraphLike]:
    """Rewrite ``query.target`` to the engine's prepared artifact (or the raw source)."""
    if isinstance(query.target, PreparedTarget):
        graph, _ = unpack_target(engine, query.target)
        return Query(cast(GraphLike, graph), query.pattern, query.semantics, query.node_attr, query.edge_attr)
    return cast("Query[GraphLike]", query)


@runtime_checkable
class Matcher(Protocol):
    """The shared backend interface; ``Portfolio`` implements it as well."""

    name: str
    capabilities: Capabilities

    def available(self) -> bool: ...

    def prepare(
        self,
        target: GraphLike,
        *,
        node_attr: str | None = None,
        edge_attr: str | None = None,
    ) -> PreparedTarget: ...

    def contains(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
        time_limit: float | None = None,
    ) -> bool: ...

    def first_mapping(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
        time_limit: float | None = None,
    ) -> dict[Hashable, Hashable] | None: ...

    def mappings(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> list[dict[Hashable, Hashable]]: ...

    def count(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> int: ...

    def occurrences(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> list[frozenset[Hashable]]: ...


def require_semantics(capabilities: Capabilities, semantics: Semantics, name: str) -> None:
    if not capabilities.supports(semantics):
        raise UnsupportedOperation(f"{name} does not support {semantics.value} matching.")


class BaseMatcher(ABC):
    """Convenience ABC for in-tree engines.

    :class:`Matcher` is the public, structural interface: a third-party engine
    never needs to import subiso to be usable, and tests can pass fakes.
    In-tree engines subclass this ABC instead, so a missing method is an
    instantiation error rather than a late AttributeError.
    """

    name: str
    capabilities: Capabilities

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def prepare(
        self,
        target: GraphLike,
        *,
        node_attr: str | None = None,
        edge_attr: str | None = None,
    ) -> PreparedTarget: ...

    @abstractmethod
    def contains(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
        time_limit: float | None = None,
    ) -> bool: ...

    @abstractmethod
    def first_mapping(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
        time_limit: float | None = None,
    ) -> dict[Hashable, Hashable] | None: ...

    @abstractmethod
    def mappings(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> list[dict[Hashable, Hashable]]: ...

    @abstractmethod
    def count(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> int: ...

    @abstractmethod
    def occurrences(
        self,
        query: Query[GraphLike | PreparedTarget],
        *,
        cap: int | None = None,
        domains: Mapping[Hashable, Collection[Hashable]] | None = None,
    ) -> list[frozenset[Hashable]]: ...
