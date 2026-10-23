"""Cross-engine benchmark harness.

Times every capable engine next to the portfolio on the same cells, asserts
that all answers agree (counts equal, mapping validity checked) before a row is
written, and stores CSVs plus a manifest with versions and the git commit.
"""

from __future__ import annotations

import csv
import datetime
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from ..matcher import Matcher, MatcherCrash, Query, UnsupportedOperation

OPERATIONS = ("contains", "count", "first_mapping")
DEPENDENCY_NAMES = ("subiso", "networkx", "igraph", "vf3py", "fastiso", "rdkit", "subiso-glasgow")


@dataclass(frozen=True)
class Cell:
    """One query to time, under a stable name."""

    name: str
    query: Query


def time_call(call: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``call`` and return ``(value, seconds)``; GC stays off inside the timer."""
    gc.disable()
    try:
        start = time.perf_counter()
        value = call()
        return value, time.perf_counter() - start
    finally:
        gc.enable()


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _git_commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def supports(engine: Matcher, operation: str, query: Query) -> bool:
    """Whether ``engine`` declares this operation for this query."""
    caps = engine.capabilities
    if not caps.supports(query.semantics):
        return False
    if query.labelled and not caps.node_labels:
        return False
    if query.node_attr and not caps.node_labels:
        return False
    if query.edge_attr and not caps.edge_labels:
        return False
    return getattr(caps, {"contains": "contains", "count": "count", "first_mapping": "first_mapping"}[operation])


def _validate_mapping(query: Query, mapping: dict) -> None:
    pattern, target = query.pattern, query.target
    assert set(mapping) == set(pattern.nodes()), (sorted(mapping), sorted(pattern.nodes()))
    assert len(set(mapping.values())) == len(mapping), "mapping is not injective"
    assert all(node in target for node in mapping.values()), "mapping leaves the target"
    if query.node_attr is not None:
        for pattern_node, target_node in mapping.items():
            required = pattern.nodes[pattern_node].get(query.node_attr)
            assert required is None or target.nodes[target_node].get(query.node_attr) == required
    for left, right in pattern.edges():
        assert target.has_edge(mapping[left], mapping[right]), (left, right)
    if query.semantics.value == "induced":
        for left, right in nx.non_edges(pattern):
            assert not target.has_edge(mapping[left], mapping[right]), (left, right)


def _call(engine: Matcher, operation: str, query: Query):
    if operation == "contains":
        return engine.contains(query)
    if operation == "count":
        return engine.count(query)
    return engine.first_mapping(query)


def _row_value(operation: str, value: Any) -> dict[str, Any]:
    if operation == "count":
        return {"result_bool": "", "result_count": int(value)}
    if operation == "contains":
        return {"result_bool": bool(value), "result_count": ""}
    return {"result_bool": value is not None, "result_count": ""}


def benchmark_cell(
    cell: Cell,
    engines: Sequence[Matcher],
    *,
    operations: Sequence[str] = OPERATIONS,
    repeats: int = 3,
    warmups: int = 1,
    protocol: str = "one-shot",
) -> list[dict[str, Any]]:
    """Time one cell across engines, asserting cross-engine agreement first.

    ``protocol``:
      * ``one-shot`` -- every timed call pays its own host conversion (the
        naive single-query usage);
      * ``amortized`` -- each engine prepares the host once (recorded as a
        ``prepare`` row) and the timed queries reuse it (many patterns per host).
    """
    assert repeats >= 1
    assert warmups >= 0
    assert protocol in ("one-shot", "amortized")
    rows: list[dict[str, Any]] = []
    queries: dict[str, Query] = {}
    if protocol == "amortized":
        for engine in engines:
            try:
                prepared, seconds = time_call(
                    lambda engine=engine: engine.prepare(
                        cell.query.target, node_attr=cell.query.node_attr, edge_attr=cell.query.edge_attr
                    )
                )
            except UnsupportedOperation:
                continue
            queries[engine.name] = Query(
                prepared, cell.query.pattern, cell.query.semantics, cell.query.node_attr, cell.query.edge_attr
            )
            rows.append(
                {
                    "cell": cell.name,
                    "operation": "prepare",
                    "engine": engine.name,
                    "repeat": 0,
                    "seconds": seconds,
                    "result_bool": "",
                    "result_count": "",
                }
            )
    for operation in operations:
        assert operation in OPERATIONS, f"unknown operation {operation!r}"
        results: dict[str, Any] = {}
        for engine in engines:
            if not supports(engine, operation, cell.query):
                continue
            query = queries.get(engine.name, cell.query)
            try:
                for _ in range(warmups):
                    _call(engine, operation, query)
                for repeat in range(repeats):
                    value, seconds = time_call(
                        lambda engine=engine, operation=operation, query=query: _call(engine, operation, query)
                    )
                    results[engine.name] = value
                    rows.append(
                        {
                            "cell": cell.name,
                            "operation": operation,
                            "engine": engine.name,
                            "repeat": repeat,
                            "seconds": seconds,
                            **_row_value(operation, value),
                        }
                    )
            except UnsupportedOperation:
                continue
            except MatcherCrash as crash:
                print(f"warning: {engine.name} crashed on {cell.name} ({operation}); skipping it: {crash}")
                continue
        if not results:
            continue  # every engine skipped this operation for this query
        if operation == "first_mapping":
            found = {engine: value is not None for engine, value in results.items()}
            assert len(set(found.values())) <= 1, (cell.name, found)
            for value in results.values():
                if value is not None:
                    _validate_mapping(cell.query, value)
        else:
            assert len(set(results.values())) == 1, (cell.name, operation, results)
    return rows


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["cell"], row["operation"], row["engine"])].append(row)
    summary = []
    for (cell, operation, engine), group in sorted(grouped.items()):
        seconds = [row["seconds"] for row in group]
        summary.append(
            {
                "cell": cell,
                "operation": operation,
                "engine": engine,
                "calls": len(group),
                "min_seconds": min(seconds),
                "median_seconds": statistics.median(seconds),
            }
        )
    return summary


def run_sweep(
    cells: Sequence[Cell],
    engines: Sequence[Matcher],
    *,
    operations: Sequence[str] = OPERATIONS,
    repeats: int = 3,
    warmups: int = 1,
    protocol: str = "one-shot",
    output_directory: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
) -> Path:
    """Run every cell and write ``results.csv``, ``summary.csv`` and ``manifest.json``."""
    if output_directory is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_directory = Path("output") / f"benchmark_{timestamp}"
    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"benchmark: {len(cells)} cells x {len(engines)} engines -> {output_path}")

    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        rows.extend(
            benchmark_cell(
                cell, engines, operations=operations, repeats=repeats, warmups=warmups, protocol=protocol
            )
        )
        print(f"  [{index}/{len(cells)}] {cell.name}")

    results_path = output_path / "results.csv"
    summary_path = output_path / "summary.csv"
    manifest_path = output_path / "manifest.json"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cell", "operation", "engine", "repeat", "seconds", "result_bool", "result_count"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cell", "operation", "engine", "calls", "min_seconds", "median_seconds"]
        )
        writer.writeheader()
        writer.writerows(_summary_rows(rows))
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
        "dependencies": {name: _dependency_version(name) for name in DEPENDENCY_NAMES},
        "engines": [engine.name for engine in engines],
        "protocol": protocol,
        "parameters": parameters or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {results_path} ({len(rows)} rows)")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    return output_path
