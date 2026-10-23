"""MIT ``SubgraphMatching`` suite backend (subprocess driver).

Wraps the common framework of Sun & Luo (SIGMOD 2020, MIT licence): QuickSI,
GraphQL, CFL-Match, CECI, DP-iso, RI, and LFTJ, selected by filter/order/engine
combinations.  The upstream binary only reports the number of enumerated
embeddings (no mappings), solves the ordinary (non-induced) problem, and reads
vertex-labelled ``.graph`` files, so this matcher advertises existence and an
extra ``count_embeddings`` helper -- not hosting-set counts or mappings.

The driver pays process start-up and file I/O per query; that is visible at
millisecond scale and is a deliberate first step before an in-process shim.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import networkx as nx

from ..capabilities import Capabilities
from ..matcher import (
    BaseMatcher,
    MatcherCrash,
    PreparedTarget,
    Query,
    UnsupportedOperation,
    require_semantics,
    unpack_query,
)
from ..semantics import Semantics

#: algorithm name -> (filter, order, engine) as accepted by the upstream binary.
ALGORITHMS: dict[str, tuple[str, str, str]] = {
    "quicksi": ("LDF", "QSI", "QSI"),
    "graphql": ("GQL", "GQL", "GQL"),
    "cfl": ("CFL", "CFL", "EXPLORE"),
    "ceci": ("CECI", "CECI", "CECI"),
    "dpiso": ("DPiso", "DPiso", "DPiso"),
    "ri": ("GQL", "RI", "LFTJ"),
    "lftj": ("LDF", "CFL", "LFTJ"),
}

_EMBEDDINGS_RE = re.compile(r"#Embeddings:\s*(\d+)")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BINARY = _REPO_ROOT / "vendor" / "subgraphmatching" / "build" / "matching" / "SubgraphMatching.out"


def resolve_binary(binary_path: str | Path | None = None) -> Path | None:
    """Binary from the argument, ``SUBISO_SUBMATCH_BINARY``, or the vendored build."""
    if binary_path is not None:
        path = Path(binary_path)
        return path if path.exists() else None
    env = os.environ.get("SUBISO_SUBMATCH_BINARY")
    if env:
        path = Path(env)
        return path if path.exists() else None
    return _DEFAULT_BINARY if _DEFAULT_BINARY.exists() else None


def _write_graph(path: Path, graph: nx.Graph, labels: dict, index: dict) -> None:
    """Write ``graph`` with consecutive vertex ids; the suite requires ids in [0, N-1]."""
    lines = [f"t {graph.number_of_nodes()} {graph.number_of_edges()}"]
    for node in graph.nodes():
        lines.append(f"v {index[node]} {labels[node]} {graph.degree(node)}")
    for left, right in graph.edges():
        lines.append(f"e {index[left]} {index[right]}")
    path.write_text("\n".join(lines) + "\n")


def _label_maps(query: Query) -> tuple[dict, dict]:
    """Integer labels for pattern and target; requires concrete values under ``node_attr``."""
    attribute = query.node_attr
    if attribute is None:
        return {node: 0 for node in query.pattern}, {node: 0 for node in query.target}
    for graph, side in ((query.pattern, "pattern"), (query.target, "target")):
        if any(graph.nodes[node].get(attribute) is None for node in graph):
            raise UnsupportedOperation(
                f"the SubgraphMatching suite needs concrete integer labels; {side} has missing or None values."
            )
    values = sorted({str(graph.nodes[node][attribute]) for graph in (query.pattern, query.target) for node in graph})
    mapping = {value: index for index, value in enumerate(values)}
    return (
        {node: mapping[str(query.pattern.nodes[node][attribute])] for node in query.pattern},
        {node: mapping[str(query.target.nodes[node][attribute])] for node in query.target},
    )


def _applicable(query: Query) -> bool:
    """Simple, undirected, edge-carrying, connected inputs.

    Inputs are relabelled to consecutive vertex ids in :meth:`_run`, because
    the suite requires ids in ``[0, N-1]``; skipping that relabel crashes most
    engines (a real wrapper bug found by the sweep's agreement assertion).

    Genuine upstream limits: edgeless patterns crash every engine except
    QuickSI, and patterns with isolated vertices crash or return wrong
    answers, so both shapes are gated off.
    """
    for graph in (query.target, query.pattern):
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            return False
    return query.target.number_of_edges() > 0 and query.pattern.number_of_edges() > 0 and nx.is_connected(query.pattern)


class SubgraphMatchingMatcher(BaseMatcher):
    """One algorithm from the MIT framework, run as a subprocess."""

    ordinary_only = True

    def __init__(self, algorithm: str, binary_path: str | Path | None = None) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; known: {sorted(ALGORITHMS)}")
        self.algorithm = algorithm
        self.name = algorithm
        self._binary = resolve_binary(binary_path)
        self.capabilities = Capabilities(ordinary=True, node_labels=True, contains=True, count_embeddings=True)

    def available(self) -> bool:
        return self._binary is not None

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        """File-based driver: the host file is rewritten per call, so this is a no-op."""
        return PreparedTarget(self.name, target, target)

    # -- driver ------------------------------------------------------------

    def _run(self, query: Query, num: int | str) -> int:
        require_semantics(self.capabilities, query.semantics, self.name)
        if query.edge_attr is not None:
            raise UnsupportedOperation("the SubgraphMatching suite reads vertex labels only.")
        if not _applicable(query):
            raise UnsupportedOperation(
                "the suite rejects edgeless or disconnected patterns and edgeless targets (it segfaults)."
            )
        filter_name, order_name, engine_name = ALGORITHMS[self.algorithm]
        pattern_labels, target_labels = _label_maps(query)
        pattern_index = {node: position for position, node in enumerate(query.pattern.nodes())}
        target_index = {node: position for position, node in enumerate(query.target.nodes())}
        with tempfile.TemporaryDirectory(prefix="subiso_sm_") as directory:
            pattern_path = Path(directory) / "pattern.graph"
            target_path = Path(directory) / "target.graph"
            _write_graph(pattern_path, query.pattern, pattern_labels, pattern_index)
            _write_graph(target_path, query.target, target_labels, target_index)
            result = subprocess.run(
                [
                    str(self._binary),
                    "-d",
                    str(target_path),
                    "-q",
                    str(pattern_path),
                    "-filter",
                    filter_name,
                    "-order",
                    order_name,
                    "-engine",
                    engine_name,
                    "-num",
                    str(num),
                ],
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            message = f"{self.name} failed (rc={result.returncode}): {result.stderr.strip()[:400]}"
            if result.returncode < 0:
                raise MatcherCrash(message)  # native crash: the sweep records and skips it
            raise RuntimeError(message)
        match = _EMBEDDINGS_RE.search(result.stdout)
        if match is None:
            raise RuntimeError(f"{self.name} printed no #Embeddings line.")
        return int(match.group(1))

    # -- Matcher protocol --------------------------------------------------

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query = unpack_query(self.name, query)
        if domains is not None or time_limit is not None:
            raise UnsupportedOperation("the SubgraphMatching suite cannot express domains or time limits.")
        if not isinstance(query.target, nx.Graph) or not isinstance(query.pattern, nx.Graph):
            raise UnsupportedOperation("the SubgraphMatching suite reads NetworkX graphs.")
        if query.semantics is Semantics.INDUCED:
            raise UnsupportedOperation("the SubgraphMatching suite solves the ordinary problem only.")
        return self._run(query, 1) > 0

    def count_embeddings(self, query: Query, *, num: int | None = None) -> int:
        """Number of enumerated embeddings, capped at ``num`` when given."""
        query = unpack_query(self.name, query)
        if query.semantics is Semantics.INDUCED:
            raise UnsupportedOperation("the SubgraphMatching suite solves the ordinary problem only.")
        return self._run(query, "MAX" if num is None else num)

    def first_mapping(self, query, *, domains=None, time_limit=None):
        raise UnsupportedOperation("the SubgraphMatching suite does not print mappings.")

    def mappings(self, query, *, cap=None, domains=None):
        raise UnsupportedOperation("the SubgraphMatching suite does not print mappings.")

    def count(self, query, *, cap=None, domains=None) -> int:
        raise UnsupportedOperation("the SubgraphMatching suite counts embeddings, not hosting sets.")

    def occurrences(self, query, *, cap=None, domains=None):
        raise UnsupportedOperation("the SubgraphMatching suite does not print mappings.")
