"""Host selection shared by the sweep and the offline evaluator.

A host must be able to supply a *connected* induced fragment of the largest
pattern order, otherwise the pattern builder either fails or (as in an earlier
version) silently returns a smaller pattern.  Disconnected molecular hosts --
salts, ions, multi-component complexes -- are therefore excluded up front.
"""

from __future__ import annotations

import networkx as nx

from subiso.datasets.molecules import fragment_pattern
from subiso.datasets.tudataset import sample
from subiso.matcher import Query
from subiso.semantics import Semantics


def largest_component(graph: nx.Graph) -> int:
    return max((len(component) for component in nx.connected_components(graph)), default=0)


def eligible_hosts(graphs: list[nx.Graph], sizes: tuple[int, ...]) -> list[nx.Graph]:
    """Hosts with a connected component large enough for every pattern order."""
    return [graph for graph in graphs if largest_component(graph) >= max(sizes)]


def fragment_queries(name: str, data_directory: str, host_count: int = 20, sizes: tuple[int, ...] = (4, 6, 8)):
    """Deterministic host/fragment cells for one TU collection."""
    from subiso.datasets.tudataset import load

    graphs = load(name, data_directory, limit=1000)
    chosen = sample(eligible_hosts(graphs, sizes), host_count, seed=0)
    queries: dict[str, Query] = {}
    for index, host in enumerate(chosen):
        for k in sizes:
            cell = f"{name.lower()}_{index:02d}_k{k}"
            queries[cell] = Query(host, fragment_pattern(host, k, seed=index * 10 + k), Semantics.INDUCED)
    return queries
