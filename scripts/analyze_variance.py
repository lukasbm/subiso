"""Aggregate the three small-graph runs: per-source spread + time-bucket crossover."""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from evaluate_policy import _dataset_queries, _matchers  # noqa: E402

from subiso.dispatch import DispatchPolicy  # noqa: E402

ROOT = Path(os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode"))
RUNS = ("subiso_dataset_sweeps_v5", "subiso_dataset_sweeps_v6", "subiso_dataset_sweeps_v7")
SOURCES = (("proteins", "PROTEINS"), ("enzymes", "ENZYMES"), ("mutag", "MUTAG"), ("aids", "AIDS"))
KEEP = ("vf2", "vf2pp", "lad", "vf3", "vf3l", "vf3p", "glasgow")
policy = DispatchPolicy(_matchers())


def load(directory: Path) -> dict[tuple[str, str], float]:
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in csv.DictReader((directory / "results.csv").open()):
        if row["operation"] == "contains":
            times[(row["cell"], row["engine"])].append(float(row["seconds"]))
    return {key: min(values) for key, values in times.items()}


table: dict[tuple[str, str, str], dict[str, float]] = {}
cells = []
for run in RUNS:
    for protocol, prepared, suffix in (("one-shot", False, ""), ("prepared", True, "_amortized")):
        for name, upper in SOURCES:
            times = load(ROOT / run / f"{name}{suffix}")
            queries = _dataset_queries(upper, ROOT / "subiso_datasets")
            choice, served = [], []
            for cell, query in queries.items():
                available = {engine: times[(cell, engine)] for engine in KEEP if (cell, engine) in times}
                if not available:
                    continue
                oracle = min(available.values())
                chosen = next(
                    (
                        matcher.name
                        for matcher in policy.candidates(query, prepared=prepared)
                        if (cell, matcher.name) in times
                    ),
                    None,
                )
                if chosen is not None:
                    choice.append(times[(cell, chosen)] / oracle)
                if (cell, "portfolio") in times:
                    regret = times[(cell, "portfolio")] / oracle
                    served.append(regret)
                    cells.append(
                        {
                            "run": run,
                            "protocol": protocol,
                            "source": name,
                            "cell": cell,
                            "oracle_us": oracle * 1e6,
                            "served": regret,
                            "static_vf3l": times[(cell, "vf3l")] / oracle if (cell, "vf3l") in times else None,
                        }
                    )
            table[(protocol, name, run)] = {
                "choice_mean": statistics.mean(choice),
                "served_mean": statistics.mean(served),
                "served_median": statistics.median(served),
            }

print(f"{'protocol':<10} {'source':<10} {'run means (served)':<34} {'mean':>6} {'sd':>5}")
macro = {}
for protocol in ("one-shot", "prepared"):
    for name, _ in SOURCES:
        values = [table[(protocol, name, run)]["served_mean"] for run in RUNS]
        macro.setdefault(protocol, []).append(values)
        print(
            f"{protocol:<10} {name:<10} "
            + " ".join(f"{value:6.2f}" for value in values)
            + f"   {statistics.mean(values):6.2f} {statistics.stdev(values):5.2f}"
        )
    per_run = [
        statistics.mean(table[(protocol, name, run)]["served_mean"] for name, _ in SOURCES) for run in RUNS
    ]
    print(
        f"{protocol:<10} {'MACRO':<10} "
        + " ".join(f"{value:6.2f}" for value in per_run)
        + f"   {statistics.mean(per_run):6.2f} {statistics.stdev(per_run):5.2f}"
    )

print("\nchoice (dispatch-free) means:")
for protocol in ("one-shot", "prepared"):
    per_run = []
    for run in RUNS:
        per_run.append(statistics.mean(table[(protocol, name, run)]["choice_mean"] for name, _ in SOURCES))
    print(
        f"  {protocol:<10} "
        + " ".join(f"{value:5.2f}" for value in per_run)
        + f"  mean {statistics.mean(per_run):5.2f}"
    )

print("\ntime-bucket crossover (pooled runs):")
for protocol in ("one-shot", "prepared"):
    print(f"  {protocol}")
    for low, high in ((0, 25), (25, 50), (50, 100), (100, 200), (200, 10**9)):
        bucket = [row for row in cells if row["protocol"] == protocol and low <= row["oracle_us"] < high]
        if not bucket:
            continue
        served = statistics.mean(row["served"] for row in bucket)
        static = statistics.mean(row["static_vf3l"] for row in bucket if row["static_vf3l"] is not None)
        print(f"    {f'{low}-{high}':<10} cells {len(bucket):>4}  portfolio {served:6.2f}  static VF3L {static:6.2f}")

(ROOT / "variance.json").write_text(
    json.dumps({"table": {str(key): value for key, value in table.items()}, "cells": cells}, indent=2)
)
print(f"\nwrote {ROOT / 'variance.json'}")
