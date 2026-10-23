"""Graph shape helpers that work without importing any optional engine.

The core package must stay importable with only NetworkX installed, so it
cannot do ``isinstance(graph, igraph.Graph)``.  igraph graphs are recognised
structurally (they expose ``vcount``/``ecount`` and no ``nodes``), which is
enough for the handful of shape checks the router needs.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx

GraphLike = nx.Graph | Any


def is_igraph(graph: object) -> bool:
    """Whether ``graph`` looks like an ``igraph.Graph`` (without importing igraph)."""
    return not isinstance(graph, nx.Graph) and hasattr(graph, "vcount") and hasattr(graph, "ecount")


def node_count(graph: GraphLike) -> int:
    if is_igraph(graph):
        return int(graph.vcount())  # type: ignore[union-attr]
    return graph.number_of_nodes()


def edge_count(graph: GraphLike) -> int:
    if is_igraph(graph):
        return int(graph.ecount())  # type: ignore[union-attr]
    return graph.number_of_edges()


def density(graph: GraphLike) -> float:
    nodes, edges = node_count(graph), edge_count(graph)
    return 2 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0


def has_edges(graph: GraphLike) -> bool:
    return edge_count(graph) > 0


def is_simple_undirected(graph: object) -> bool:
    """NetworkX simple-undirected shape; igraph graphs fail this and fall back to their source."""
    return isinstance(graph, nx.Graph) and not graph.is_directed() and not graph.is_multigraph()


def is_supported_shape(graph: object) -> bool:
    """Simple undirected input, on either graph type."""
    if is_igraph(graph):
        return True
    return is_simple_undirected(graph)


def label_of(graph: GraphLike, node: Hashable) -> object:
    """A node's attribute dictionary entry, or ``None`` when the type carries none."""
    if is_igraph(graph):
        return None
    return graph.nodes[node]
