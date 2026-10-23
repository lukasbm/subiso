"""Time one engine on one large-host cell; write a single CSV row.

Designed for GNU Parallel: one process per (cell, engine, protocol), no shared
state, output file named by the triple.  A job killed by the driver's wall-clock
timeout simply leaves no row; the summary counts those as missing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import fire
from large_cells import build_cell

from subiso.backends.lad import LadMatcher
from subiso.backends.subgraphmatching import ALGORITHMS, SubgraphMatchingMatcher
from subiso.backends.vf2 import Vf2Matcher
from subiso.backends.vf2pp import Vf2ppMatcher
from subiso.backends.vf3 import Vf3lMatcher, Vf3Matcher
from subiso.benchmark.harness import time_call
from subiso.matcher import Query, UnsupportedOperation
from subiso.portfolio import Portfolio

DIRECT = {
    "vf2": Vf2Matcher,
    "vf2pp": Vf2ppMatcher,
    "lad": LadMatcher,
    "vf3": Vf3Matcher,
    "vf3l": Vf3lMatcher,
    "portfolio": Portfolio,
}


def make_engine(name: str):
    if name in ALGORITHMS:
        return SubgraphMatchingMatcher(name)
    if name == "glasgow":
        from subiso_glasgow import GlasgowMatcher

        return GlasgowMatcher()
    return DIRECT[name]()


def main(
    cell: str,
    engine: str,
    protocol: str = "one-shot",
    output_dir: str = "/scratch/lukas/tmp/opencode/large_hosts/rows",
    repeats: int = 2,
    warmups: int = 1,
) -> None:
    assert protocol in ("one-shot", "amortized"), f"unknown protocol {protocol!r}"
    row = {
        "cell": cell,
        "engine": engine,
        "protocol": protocol,
        "operation": "contains",
        "seconds": "",
        "result": "",
        "prepare_seconds": "",
        "status": "",
        "detail": "",
    }
    try:
        query = build_cell(cell).query
        matcher = make_engine(engine)
        if not matcher.available():
            row["status"] = "unavailable"
        elif not matcher.capabilities.contains or not matcher.capabilities.supports(query.semantics):
            row["status"] = "unsupported"
        else:
            if protocol == "amortized":
                prepared, prepare_seconds = time_call(
                    lambda: matcher.prepare(query.target, node_attr=query.node_attr, edge_attr=query.edge_attr)
                )
                row["prepare_seconds"] = prepare_seconds
                timed = Query(prepared, query.pattern, query.semantics, query.node_attr, query.edge_attr)
            else:
                timed = query
            for _ in range(warmups):
                matcher.contains(timed)
            best, result = float("inf"), None
            for _ in range(repeats):
                value, seconds = time_call(lambda: matcher.contains(timed))
                best, result = min(best, seconds), value
            row["seconds"] = best
            row["result"] = bool(result)
            row["status"] = "ok"
    except UnsupportedOperation as error:
        row["status"] = "unsupported"
        row["detail"] = str(error)[:300]
    except Exception as error:  # noqa: BLE001 - the row records the failure
        row["status"] = "error"
        row["detail"] = f"{type(error).__name__}: {error}"[:300]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{cell}__{engine}__{protocol}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    fire.Fire(main)
