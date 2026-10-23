"""Per-axis win/oracle/regret tables from a benchmark output directory.

Usage: python scripts/axis_report.py <output-directory> [operation]

Reads results.csv, aggregates per (cell, operation, engine) by minimum time,
and reports which engine is the per-cell oracle overall and conditioned on
each experimental axis (relation, hit/miss, source, label count, host order,
density, pattern order).  Portfolio regret is reported per axis value.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CELL_RE = re.compile(r"^n(\d+)_p([0-9.]+)_k(\d+)_(planted|random)_(induced|ordinary)(?:_l(\d+))?$")


def _load(directory: Path) -> tuple[dict[tuple, float], dict[str, dict]]:
    times: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[str, dict] = {}
    hits: dict[str, bool] = {}
    for row in csv.DictReader((directory / "results.csv").open()):
        match = CELL_RE.match(row["cell"])
        if not match:
            continue
        cell = row["cell"]
        meta[cell] = {
            "order": int(match.group(1)),
            "density": float(match.group(2)),
            "k": int(match.group(3)),
            "source": match.group(4),
            "relation": match.group(5),
            "labels": int(match.group(6) or 0),
        }
        if row["operation"] == "contains" and row["result_bool"]:
            hits[cell] = row["result_bool"] == "True"
        if row["operation"] != "prepare":
            times[(cell, row["operation"], row["engine"])].append(float(row["seconds"]))
    for cell, hit in hits.items():
        meta[cell]["hit"] = hit
    return {key: min(values) for key, values in times.items()}, meta


def _report(directory: Path, operation: str) -> None:
    times, meta = _load(directory)
    cells = sorted({cell for cell, op, _ in times if op == operation and cell in meta})
    engines = sorted({engine for _, op, engine in times if op == operation and engine != "portfolio"})

    def oracle(cell: str) -> tuple[str, float] | None:
        candidates = [
            (engine, times[(cell, operation, engine)]) for engine in engines if (cell, operation, engine) in times
        ]
        return min(candidates, key=lambda item: item[1]) if candidates else None

    print(f"\n### {directory.name} | operation={operation} | {len(cells)} cells")
    overall: dict[str, int] = defaultdict(int)
    for cell in cells:
        best = oracle(cell)
        if best:
            overall[best[0]] += 1
    total = sum(overall.values())
    print(
        "oracle overall: "
        + ", ".join(f"{engine} {count}/{total}" for engine, count in sorted(overall.items(), key=lambda kv: -kv[1]))
    )

    for axis in ("relation", "hit", "source", "labels", "order", "density", "k"):
        grouped: dict[object, list[str]] = defaultdict(list)
        for cell in cells:
            grouped[meta[cell].get(axis, "?")].append(cell)
        lines = []
        for value, group in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            wins: dict[str, int] = defaultdict(int)
            for cell in group:
                best = oracle(cell)
                if best:
                    wins[best[0]] += 1
            winners = ", ".join(
                f"{engine}:{count}" for engine, count in sorted(wins.items(), key=lambda kv: -kv[1])[:3]
            )
            regrets = []
            for cell in group:
                best = oracle(cell)
                if best and (cell, operation, "portfolio") in times:
                    regrets.append(times[(cell, operation, "portfolio")] / best[1])
            regret = f" med-regret {statistics.median(regrets):.2f}" if regrets else ""
            lines.append(f"  {axis}={value} ({len(group)}): {winners}{regret}")
        print("\n".join(lines))


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    directory = Path(argv[1])
    operation = argv[2] if len(argv) > 2 else "contains"
    _report(directory, operation)


if __name__ == "__main__":
    main(sys.argv)
