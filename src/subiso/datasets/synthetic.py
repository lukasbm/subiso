"""Seeded synthetic hosts and patterns.

Identical seeds give identical graphs; generation follows the protocol used by
the matcher sweep: a host is G(n, p) (or a random regular graph), a planted
pattern is an induced subgraph sampled from the host, and a random pattern is
denser than the host by ``miss_boost`` so both hits and absences are covered.
"""

from __future__ import annotations

import random

import networkx as nx


def attach_labels(graph: nx.Graph, label_count: int, rng: random.Random, *, node_attr: str = "label") -> nx.Graph:
    """Attach integer labels drawn uniformly from ``range(label_count)``."""
    if label_count == 0:
        return graph
    for node in graph:
        graph.nodes[node][node_attr] = rng.randrange(label_count)
    return graph


def gnp_host(order: int, density: float, *, seed: int, label_count: int = 0, node_attr: str = "label") -> nx.Graph:
    """G(n, p) host with optional node labels."""
    assert order >= 1
    assert 0.0 <= density <= 1.0
    rng = random.Random(seed)
    host = nx.gnp_random_graph(order, density, seed=rng.randrange(2**31))
    return attach_labels(host, label_count, rng, node_attr=node_attr)


def random_regular_host(
    order: int, degree: int, *, seed: int, label_count: int = 0, node_attr: str = "label"
) -> nx.Graph:
    """Random d-regular host; the known worst case for pruning-based matchers."""
    assert degree < order
    rng = random.Random(seed)
    host = nx.random_regular_graph(degree, order, seed=rng.randrange(2**31))
    return attach_labels(host, label_count, rng, node_attr=node_attr)


def planted_pattern(host: nx.Graph, pattern_order: int, *, seed: int) -> nx.Graph:
    """Induced subgraph of ``host`` on a uniformly sampled vertex subset."""
    assert 1 <= pattern_order <= host.number_of_nodes()
    rng = random.Random(seed)
    nodes = rng.sample(list(host.nodes()), pattern_order)
    pattern = host.subgraph(nodes).copy()
    pattern.graph["pattern_source"] = "planted"
    return pattern


def random_pattern(
    pattern_order: int,
    density: float,
    *,
    seed: int,
    label_count: int = 0,
    miss_boost: float = 0.75,
    node_attr: str = "label",
) -> nx.Graph:
    """G(k, min(0.95, density + miss_boost)) pattern, usually an absence."""
    assert 0.0 <= miss_boost <= 0.9
    rng = random.Random(seed)
    pattern = nx.gnp_random_graph(pattern_order, min(0.95, density + miss_boost), seed=rng.randrange(2**31))
    attach_labels(pattern, label_count, rng, node_attr=node_attr)
    pattern.graph["pattern_source"] = "random"
    return pattern
