"""Evaluate the shipped DispatchPolicy offline against recorded sweeps.

Dispatch only selects among engines whose times were already recorded, so the
policy can be evaluated without re-running any matcher: reconstruct each
cell's query, ask DispatchPolicy for candidates, take the first with a
recorded time, and score regret/MRR against the per-cell oracle.  Also scores
the recorded `portfolio` rows (the old hand rule) for comparison.

Usage: python scripts/evaluate_policy.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import fire

from subiso.backends.lad import LadMatcher
from subiso.backends.vf2 import Vf2Matcher
from subiso.backends.vf2pp import Vf2ppMatcher
from subiso.backends.vf3 import Vf3lMatcher, Vf3Matcher
from subiso.benchmark.grid import synthetic_cells
from subiso.dispatch import DispatchPolicy
from subiso.matcher import Query

SWEEP_ROOT = os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode")
KEEP = ("vf2", "vf2pp", "lad", "vf3", "vf3l", "vf3p", "glasgow")


def _matchers():
    matchers = [Vf2Matcher(), Vf2ppMatcher(), LadMatcher(), Vf3Matcher(), Vf3lMatcher()]
    try:
        from subiso_glasgow import GlasgowMatcher

        matchers.append(GlasgowMatcher())
    except ImportError:
        pass
    return matchers


def _load(sweep: Path, operation: str) -> dict[tuple[str, str], float]:
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in csv.DictReader((sweep / "results.csv").open()):
        if row["operation"] == operation:
            times[(row["cell"], row["engine"])].append(float(row["seconds"]))
    return {key: min(values) for key, values in times.items()}


def _synthetic_queries(sweep: Path) -> dict[str, Query]:
    parameters = json.loads((sweep / "manifest.json").read_text())["parameters"]
    cells = synthetic_cells(
        tuple(parameters["orders"]),
        tuple(parameters["densities"]),
        tuple(parameters["pattern_orders"]),
        induced_relations=tuple(parameters["induced"]),
        label_counts=tuple(parameters["label_counts"]),
        pattern_sources=tuple(parameters["pattern_sources"]),
        seed=parameters["seed"],
        miss_boost=parameters["miss_boost"],
    )
    return {cell.name: cell.query for cell in cells}


def _dataset_queries(name: str, data_directory: str, host_count: int = 20, sizes=(4, 6, 8)) -> dict[str, Query]:
    from hosts import fragment_queries

    return fragment_queries(name, data_directory, host_count=host_count, sizes=tuple(sizes))


def _score(choices: dict[str, float], times, oracle: dict[str, float]) -> tuple[float, float, float, float]:
    regrets, ranks = [], []
    for cell, chosen_time in choices.items():
        best = oracle[cell]
        regrets.append(chosen_time / best)
        ranked = sorted((e for (c, e) in times if c == cell), key=lambda e: times[(cell, e)])
        rank = 1 + sum(1 for engine in ranked if times[(cell, engine)] < chosen_time)
        ranks.append(1 / rank)
    regrets.sort()
    return (
        statistics.median(regrets),
        sum(regrets) / len(regrets),
        regrets[min(len(regrets) - 1, int(len(regrets) * 0.9))],
        sum(ranks) / len(ranks),
    )


def main(
    sweep_root: str = SWEEP_ROOT,
    data_directory: str = f"{SWEEP_ROOT}/subiso_datasets",
    operation: str = "contains",
) -> None:
    root = Path(sweep_root)
    sources: dict[str, tuple[Path, dict[str, Query]]] = {
        "sweepD": (root / "subiso_sweepD_fixed", _synthetic_queries(root / "subiso_sweepD")),
    }
    for name in ("PROTEINS", "ENZYMES", "MUTAG", "AIDS"):
        sources[name.lower()] = (
            root / "subiso_dataset_sweeps_fixed" / name.lower(),
            _dataset_queries(name, data_directory),
        )

    policy = DispatchPolicy(_matchers())
    print(
        f"{'source':<10} {'policy med':>10} {'policy mean':>11} {'policy p90':>10} {'policy MRR':>10} "
        f"{'old med':>8} {'old mean':>9} {'skipped':>7}"
    )
    macro = []
    for name, (sweep, queries) in sources.items():
        times = _load(sweep, operation)
        cells = [cell for cell in queries if any((cell, engine) in times for engine in KEEP)]
        oracle = {cell: min(times[(cell, engine)] for engine in KEEP if (cell, engine) in times) for cell in cells}
        policy_choices, skipped = {}, 0
        for cell in cells:
            for matcher in policy.candidates(queries[cell]):
                if (cell, matcher.name) in times:
                    policy_choices[cell] = times[(cell, matcher.name)]
                    break
            else:
                skipped += 1
        med, mean, p90, mrr = _score(policy_choices, times, oracle)
        old = {cell: times[(cell, "portfolio")] for cell in cells if (cell, "portfolio") in times}
        old_med, old_mean, _, _ = _score(old, times, oracle)
        macro.append((med, mean, p90, mrr))
        print(
            f"{name:<10} {med:>10.2f} {mean:>11.2f} {p90:>10.2f} {mrr:>10.2f} {old_med:>8.2f} {old_mean:>9.2f} {skipped:>7}"
        )
    print(
        f"{'MACRO':<10} {statistics.mean(m[0] for m in macro):>10.2f} {statistics.mean(m[1] for m in macro):>11.2f} "
        f"{statistics.mean(m[2] for m in macro):>10.2f} {statistics.mean(m[3] for m in macro):>10.2f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
