"""Large-host cells for the probe and for large-host training.

Two corpora share the same cell naming and builder:

* ``probe`` -- the fixed comparison grid (n=1000/5000, two densities, hosts
  h0-h2), used for evaluation;
* ``train`` -- training/validation hosts (h10-h12 train, h13 validation, n up
  to 5000) plus HPRD, a real 9,460-vertex host vendored with the MIT suite.

Hosts are shared across k/source/relation inside a cell family: the seed
depends on (order, density, host index) only.  Patterns are connected (BFS
fragments for planted, resampled random patterns), because the MIT suite
rejects disconnected patterns.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import networkx as nx

from subiso.benchmark.harness import Cell
from subiso.datasets.molecules import fragment_pattern
from subiso.datasets.synthetic import gnp_host, random_pattern
from subiso.matcher import Query
from subiso.semantics import Semantics

#: host order -> densities for the probe, chosen for mean degree ~5 and ~10.
PROBE_DENSITIES = {1000: (0.005, 0.01), 5000: (0.001, 0.002)}
PROBE_HOSTS = (0, 1, 2)

#: training corpus: one density per order (mean degree ~5), hosts h10-h13.
TRAIN_DENSITY = {300: 0.0167, 1000: 0.005, 2000: 0.0025, 5000: 0.001}
TRAIN_ORDERS = (300, 1000, 5000)
VAL_ORDERS = (300, 1000, 2000, 5000)  # n=2000 is validation-only
TRAIN_HOSTS = (10, 11, 12)
VAL_HOSTS = (13,)
PATTERN_ORDERS = (8, 16, 32)
SOURCES = ("planted", "random")
RELATIONS = ("induced", "ordinary")
#: HPRD uses the same 9,460-vertex host with different pattern seeds per index
#: (h0-h2 train, h3-h4 held-out test).
HPRD_HOSTS = (0, 1, 2, 3, 4)

PROBE_ENGINES = (
    "vf2",
    "vf2pp",
    "lad",
    "vf3",
    "vf3l",
    "glasgow",
    "portfolio",
    "quicksi",
    "graphql",
    "cfl",
    "ceci",
    "dpiso",
    "ri",
    "lftj",
)
#: large-host training covers the engines the shipped Portfolio carries.
TRAIN_ENGINES = ("vf2", "vf2pp", "lad", "vf3", "vf3l", "glasgow")

HPRD_PATH = Path(__file__).resolve().parents[1] / "vendor/subgraphmatching/test/data_graph/HPRD.graph"

_SYNTHETIC = re.compile(r"^n(\d+)_p([0-9.eE+-]+)_h(\d+)_k(\d+)_(planted|random)_(induced|ordinary)$")
_HPRD = re.compile(r"^hprd_h(\d+)_k(\d+)_(planted|random)_(induced|ordinary)$")


def cell_name(order: int, density: float, host: int, k: int, source: str, relation: str) -> str:
    return f"n{order}_p{density:g}_h{host}_k{k}_{source}_{relation}"


def probe_cell_names() -> list[str]:
    return [
        cell_name(order, density, host, k, source, relation)
        for order, densities in PROBE_DENSITIES.items()
        for density in densities
        for host in PROBE_HOSTS
        for k in PATTERN_ORDERS
        for source in SOURCES
        for relation in RELATIONS
    ]


def train_cell_names() -> list[str]:
    synthetic = [
        cell_name(order, TRAIN_DENSITY[order], host, k, source, relation)
        for order, hosts in ((order, TRAIN_HOSTS) for order in TRAIN_ORDERS)
        for host in hosts
        for k in PATTERN_ORDERS
        for source in SOURCES
        for relation in RELATIONS
    ]
    validation = [
        cell_name(order, TRAIN_DENSITY[order], host, k, source, relation)
        for order, hosts in ((order, VAL_HOSTS) for order in VAL_ORDERS)
        for host in hosts
        for k in PATTERN_ORDERS
        for source in SOURCES
        for relation in RELATIONS
    ]
    hprd = [
        f"hprd_h{host}_k{k}_{source}_{relation}"
        for host in HPRD_HOSTS
        for k in PATTERN_ORDERS
        for source in SOURCES
        for relation in RELATIONS
    ]
    return synthetic + validation + hprd


def corpus_names(corpus: str) -> list[str]:
    if corpus == "probe":
        return probe_cell_names()
    if corpus == "train":
        return train_cell_names()
    raise ValueError(f"unknown corpus {corpus!r}; expected probe or train")


def corpus_engines(corpus: str) -> tuple[str, ...]:
    return PROBE_ENGINES if corpus == "probe" else TRAIN_ENGINES


def _parse_suite_graph(path: Path) -> nx.Graph:
    """Read the suite's ``t N M / v id label degree / e u v`` format."""
    graph = nx.Graph()
    edges: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            graph.add_node(int(parts[1]), label=int(parts[2]))
        elif parts[0] == "e":
            edges.append((int(parts[1]), int(parts[2])))
    graph.add_edges_from(edges)
    return graph


def _pattern(host: nx.Graph, k: int, source: str, seed: int) -> nx.Graph:
    if source == "planted":
        return fragment_pattern(host, k, seed=seed)
    density = 2 * host.number_of_edges() / (host.number_of_nodes() * (host.number_of_nodes() - 1))
    pattern = random_pattern(k, density, seed=seed)
    while not nx.is_connected(pattern):
        seed += 1
        pattern = random_pattern(k, density, seed=seed)
    return pattern


def build_cell(name: str) -> Cell:
    synthetic = _SYNTHETIC.match(name)
    hprd = _HPRD.match(name)
    if synthetic is not None:
        order, density = int(synthetic[1]), float(synthetic[2])
        host_index, k = int(synthetic[3]), int(synthetic[4])
        source, relation = synthetic[5], synthetic[6]
        host_seed = zlib.crc32(f"n{order}_p{density:g}_h{host_index}".encode())
        host = gnp_host(order, density, seed=host_seed)
    elif hprd is not None:
        k = int(hprd[2])
        source, relation = hprd[3], hprd[4]
        host = _parse_suite_graph(HPRD_PATH)
    else:
        raise ValueError(f"unrecognised cell name {name!r}")
    pattern = _pattern(host, k, source, zlib.crc32(name.encode()) + 1)
    semantics = Semantics.INDUCED if relation == "induced" else Semantics.ORDINARY
    return Cell(name=name, query=Query(host, pattern, semantics))
