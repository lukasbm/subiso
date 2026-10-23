"""FastISO backend.

Counts induced embeddings in C++ without materialising them, so it is the fast
path for induced counting.  The count is divided by the labelled/unlabelled
``|Aut(pattern)|`` to recover the hosting-set count, which is exact precisely
because each induced hosting set is stabilised by the full automorphism group.

Node and edge labels are handed to FastISO as colour classes (equality only,
and no wildcards -- those remain VF2-only).  Optional: a declared dependency
that builds from the sdist.
"""

from __future__ import annotations

import networkx as nx

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, require_semantics, unpack_query
from ..semantics import Semantics

try:
    import fastiso as _fastiso
except Exception:  # pragma: no cover - optional backend
    _fastiso = None


def available() -> bool:
    return _fastiso is not None


def count_induced_embeddings(
    target: nx.Graph,
    pattern: nx.Graph,
    *,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> int:
    """Number of induced *embeddings* (not hosting sets) of ``pattern`` in ``target``.

    ``node_attr``/``edge_attr`` name the label keys FastISO reads as colour
    classes.  Every pattern node/edge must carry a concrete value under them;
    a missing value would silently become the empty-string class rather than a
    wildcard, so the portfolio checks for that before calling.
    """
    assert _fastiso is not None, "FastISO is not importable."
    # all_solution=1 + count_solution=1 returns the count without materialising
    # the embeddings.  (all_solution=0 is a boolean "found one", not a count.)
    return int(
        _fastiso.nx_iso(  # type: ignore[arg-type]
            pattern,
            target,
            "sub-iso",
            all_solution=True,
            count_solution=True,
            node_label_key=node_attr,
            edge_label_key=edge_attr,
        )
    )


def labels_total(pattern: nx.Graph, node_attr: str | None, edge_attr: str | None) -> bool:
    """Whether every pattern node/edge carries a concrete label (no wildcards)."""
    node_wildcards = node_attr is not None and any(pattern.nodes[node].get(node_attr) is None for node in pattern)
    edge_wildcards = edge_attr is not None and any(
        data.get(edge_attr) is None for _, _, data in pattern.edges(data=True)
    )
    return not (node_wildcards or edge_wildcards)


def labels_on_edges_only(pattern: nx.Graph) -> bool:
    """Whether every pattern vertex has an incident edge.

    FastISO's NetworkX interface writes vertex labels only on edge lines, so an
    isolated pattern vertex would lose its colour and could then match only
    unmentioned (isolated) target vertices.  Node labels are only safe to hand
    over when no pattern vertex is isolated.
    """
    return all(degree > 0 for _, degree in pattern.degree())


class FastisoMatcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of this module.

    It only answers induced hosting-set counts; the dividing automorphism count
    is taken from VF2 (labelled) or LAD (unlabelled).
    """

    name = "fastiso"
    capabilities = Capabilities(
        induced=True,
        node_labels=True,
        edge_labels=True,
        count=True,
        count_capped=True,
        count_embeddings=True,
    )

    def available(self) -> bool:
        return available()

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """This engine converts inside its calls, so preparation is a no-op holder."""
        return PreparedTarget(self.name, target, target)

    def _gates_open(self, query: Query) -> bool:
        query = unpack_query(self.name, query)
        target, pattern = query.target, query.pattern
        if not isinstance(target, nx.Graph) or not isinstance(pattern, nx.Graph):
            return False
        if target.number_of_edges() == 0 or pattern.number_of_edges() == 0:
            return False  # FastISO rejects edge-less graphs.
        if not labels_total(pattern, query.node_attr, query.edge_attr):
            return False  # wildcards are not expressible as colour classes.
        return query.node_attr is None or labels_on_edges_only(pattern)

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        raise UnsupportedOperation("FastISO only counts; it cannot answer existence cheaply.")

    def first_mapping(self, query, *, domains=None, time_limit=None):
        raise UnsupportedOperation("FastISO counts embeddings without materialising mappings.")

    def mappings(self, query, *, cap=None, domains=None):
        raise UnsupportedOperation("FastISO counts embeddings without materialising mappings.")

    def occurrences(self, query, *, cap=None, domains=None):
        raise UnsupportedOperation("FastISO counts embeddings without materialising mappings.")

    def count_embeddings(self, query: Query) -> int:
        """Number of raw induced embeddings (not hosting sets)."""
        query = unpack_query(self.name, query)
        require_semantics(self.capabilities, query.semantics, self.name)
        if query.semantics is not Semantics.INDUCED:
            raise UnsupportedOperation("FastISO counts induced embeddings only.")
        if not self._gates_open(query):
            raise UnsupportedOperation("FastISO needs edgy, fully labelled graphs without isolated pattern vertices.")
        return count_induced_embeddings(
            query.target, query.pattern, node_attr=query.node_attr, edge_attr=query.edge_attr
        )

    def count(self, query, *, cap=None, domains=None) -> int:
        query = unpack_query(self.name, query)
        require_semantics(self.capabilities, query.semantics, self.name)
        if query.semantics is not Semantics.INDUCED:
            raise UnsupportedOperation("FastISO counts induced embeddings only.")
        if domains is not None:
            raise UnsupportedOperation("FastISO cannot express domains.")
        if not self._gates_open(query):
            raise UnsupportedOperation("FastISO needs edgy, fully labelled graphs without isolated pattern vertices.")
        embeddings = count_induced_embeddings(
            query.target, query.pattern, node_attr=query.node_attr, edge_attr=query.edge_attr
        )
        if embeddings == 0:
            return 0
        if query.labelled:
            from . import vf2

            divisor = vf2.automorphism_count(query.pattern, node_attr=query.node_attr, edge_attr=query.edge_attr)
        else:
            from . import lad

            divisor = lad.automorphisms_of(query.pattern)
        count = embeddings // divisor
        return count if cap is None else min(count, cap)
