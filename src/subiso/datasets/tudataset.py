"""TU Dortmund graph collections (PROTEINS, ENZYMES, MUTAG, AIDS, ...).

Downloads the public per-dataset zip from TU Dortmund and parses its plain-text
format into a list of NetworkX graphs with integer ``label`` node attributes.
Nothing is bundled; files are cached under the caller-provided directory (or
``$SUBISO_DATA_DIR`` / ``~/.cache/subiso/datasets``).

Licence note: the collections are distributed by TU Dortmund for research use;
keep the attribution and do not re-host the archives.
"""

from __future__ import annotations

import os
import random
import urllib.request
import zipfile
from pathlib import Path

import networkx as nx

DATA_URL = "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"


def data_dir(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory)
    env = os.environ.get("SUBISO_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "subiso" / "datasets"


def _read_rows(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _parse(directory: Path, name: str) -> list[nx.Graph]:
    indicators = [int(row.split(",")[0]) for row in _read_rows(directory / f"{name}_graph_indicator.txt")]
    node_labels_path = directory / f"{name}_node_labels.txt"
    node_labels = (
        [int(row.split(",")[0]) for row in _read_rows(node_labels_path)] if node_labels_path.exists() else None
    )

    edge_label_files = [
        directory / f"{name}_edge_labels.txt",
        directory / f"{name}_edge_attributes.txt",
    ]
    edge_labels: list[int] | None = None
    for candidate in edge_label_files:
        if candidate.exists():
            edge_labels = [int(row.split(",")[0]) for row in _read_rows(candidate)]
            break

    adjacency = _read_rows(directory / f"{name}_A.txt")
    graph_count = max(indicators)
    graphs = [nx.Graph() for _ in range(graph_count)]
    for node_id, graph_id in enumerate(indicators, start=1):
        graph = graphs[graph_id - 1]
        graph.add_node(node_id)
        if node_labels is not None:
            graph.nodes[node_id]["label"] = node_labels[node_id - 1]
    for edge_index, row in enumerate(adjacency):
        left, right = (int(value) for value in row.split(",")[:2])
        graph = graphs[indicators[left - 1] - 1]
        graph.add_edge(left, right)
        if edge_labels is not None and edge_index < len(edge_labels):
            graph.edges[left, right]["label"] = edge_labels[edge_index]
    # relabel to consecutive integers and compact the node/edge attributes
    compacted = []
    for graph in graphs:
        mapping = {node: index for index, node in enumerate(graph.nodes())}
        compacted.append(nx.relabel_nodes(graph, mapping))
    return compacted


def sample(graphs: list[nx.Graph], limit: int | None, seed: int = 0) -> list[nx.Graph]:
    """Deterministic, unbiased subsample; identity when ``limit`` is None or larger."""
    if limit is None or limit >= len(graphs):
        return graphs
    return random.Random(seed).sample(graphs, limit)


def load(name: str, directory: str | Path | None = None, limit: int | None = None, seed: int = 0) -> list[nx.Graph]:
    """Download (once) and parse a TU collection; ``limit`` caps it deterministically."""
    root = data_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / name
    if not dataset_dir.exists():
        archive = root / f"{name}.zip"
        url = DATA_URL.format(name=name)
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root)
    graphs = sample(_parse(dataset_dir, name), limit, seed)
    print(f"loaded {len(graphs)} graphs from {dataset_dir}")
    return graphs
