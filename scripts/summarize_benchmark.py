"""Summarize a subiso benchmark output directory.

Reports, per operation: per-engine totals, per-cell wins, the static-best
engine, the per-cell oracle, and the portfolio's regret distribution.

Usage::

    python scripts/summarize_benchmark.py <output-directory> [operation ...]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    directory = Path(argv[1])
    operations = argv[2:]
    with (directory / "summary.csv").open() as handle:
        rows = [{**row, "min_seconds": float(row["min_seconds"])} for row in csv.DictReader(handle)]

    for operation in operations or sorted({row["operation"] for row in rows}):
        cells: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            if row["operation"] == operation:
                cells[row["cell"]][row["engine"]] = row["min_seconds"]
        if not cells:
            continue
        engines = sorted({engine for cell in cells.values() for engine in cell} - {"portfolio"})
        totals = {engine: sum(cell[engine] for cell in cells.values() if engine in cell) for engine in engines}
        oracle_total = sum(
            min(value for engine, value in cell.items() if engine != "portfolio") for cell in cells.values()
        )
        portfolio_total = sum(cell["portfolio"] for cell in cells.values() if "portfolio" in cell)

        print(f"\n== {operation}: {len(cells)} cells ==")
        for engine in sorted(engines, key=lambda name: totals[name]):
            wins = sum(
                1
                for cell in cells.values()
                if engine in cell and cell[engine] == min(v for e, v in cell.items() if e != "portfolio")
            )
            print(f"  {engine:<9} total {totals[engine] * 1000:9.1f} ms   wins {wins:3d}")
        best = min(engines, key=lambda name: totals[name])
        print(f"  static best engine: {best} ({totals[best] * 1000:.1f} ms)")
        print(f"  per-cell oracle:    {oracle_total * 1000:.1f} ms")
        if portfolio_total:
            ratios = sorted(
                cell["portfolio"] / min(v for e, v in cell.items() if e != "portfolio")
                for cell in cells.values()
                if "portfolio" in cell
            )
            print(
                f"  portfolio:          {portfolio_total * 1000:.1f} ms "
                f"(x{portfolio_total / oracle_total:.2f} vs oracle, "
                f"x{portfolio_total / sum(totals.values()) * len(totals):.2f} vs mean engine)"
            )
            print(
                f"  portfolio regret:   p50 {ratios[len(ratios) // 2]:.2f}  "
                f"p90 {ratios[min(len(ratios) - 1, int(len(ratios) * 0.9))]:.2f}  max {ratios[-1]:.2f}"
            )


if __name__ == "__main__":
    main(sys.argv)
