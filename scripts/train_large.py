"""Train the large-host dispatch trees (host order > 200).

Training cells: the probe corpus (hosts h0-h2) plus train-corpus hosts h10-h12,
orders 300/1000/5000, and HPRD pattern seeds h0-h2 (the same 9,460-vertex real
host, different query patterns).  Validation: held-out synthetic hosts h13,
including the unseen order 2000.  HPRD pattern seeds h3-h4 are held out and
are the external real-graph test.

Missing engine rows are timeouts (or crashes); they are censored at
``timeout_cap`` when scoring and imputed pessimistically when fitting, so the
tree learns to avoid engines that hang.  Hyperparameters are selected on the
validation hosts only, then the final trees are refit on all synthetic hosts.

Usage: python scripts/train_large.py   (run from the repository root)
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from pathlib import Path

import fire
import joblib
import numpy as np
from large_cells import TRAIN_ENGINES, build_cell
from lodo_train import _literal, _render
from sklearn.tree import DecisionTreeRegressor

from subiso.dispatch import cheap_features

ROOT = Path(os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode"))
ROW_DIRS = (ROOT / "large_hosts_v2" / "rows", ROOT / "large_hosts_train" / "rows")
RESULT_JSON = ROOT / "large_train_results.json"
LITERAL_ONE = ROOT / "lodo_rule_literal_large_oneshot.txt"
LITERAL_PREP = ROOT / "lodo_rule_literal_large_prepared.txt"
RESULT_TREE = ROOT / "lodo_trees_large.joblib"

_SYNTHETIC = re.compile(r"^n(\d+)_p([0-9.eE+-]+)_h(\d+)_k(\d+)_(planted|random)_(induced|ordinary)$")
_HPRD = re.compile(r"^hprd_h(\d+)_k(\d+)_(planted|random)_(induced|ordinary)$")
_GRID = ((2, 5), (2, 20), (3, 5), (3, 10), (3, 20), (4, 5), (4, 10), (4, 20), (5, 10), (5, 20))
#: largest host order in the small-tree training corpus (PROTEINS); above this
#: the dispatch policy switches to these large-host trees.
SWITCH = 620


def _load_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    records = {}
    for directory in ROW_DIRS:
        for path in sorted(directory.glob("*.csv")):
            with path.open() as handle:
                row = next(csv.DictReader(handle))
            records[(row["cell"], row["protocol"], row["engine"])] = row
    return records


def _ok(records, cell: str, protocol: str, engine: str) -> bool:
    row = records.get((cell, protocol, engine))
    return row is not None and row["status"] == "ok"


def _times(records, cell: str, protocol: str) -> dict[str, float]:
    return {
        engine: float(records[(cell, protocol, engine)]["seconds"])
        for engine in TRAIN_ENGINES
        if _ok(records, cell, protocol, engine)
    }


def _split(cell: str) -> str:
    hprd = _HPRD.match(cell)
    if hprd is not None:
        return "hprd" if int(hprd[1]) >= 3 else "train"
    match = _SYNTHETIC.match(cell)
    assert match is not None, cell
    return "val" if int(match[3]) == 13 else "train"


def _host_order(cell: str) -> int:
    if cell.startswith("hprd"):
        return 9460
    match = _SYNTHETIC.match(cell)
    assert match is not None, cell
    return int(match[1])


def _fit(cells, features, records, protocol: str, depth: int, leaf: int):
    kept = [engine for engine in TRAIN_ENGINES if any(_ok(records, cell, protocol, engine) for cell in cells)]
    targets = np.full((len(cells), len(kept)), np.nan)
    for i, cell in enumerate(cells):
        times = _times(records, cell, protocol)
        if not times:
            continue
        oracle = min(times.values())
        for j, engine in enumerate(kept):
            if engine in times:
                targets[i, j] = np.log(times[engine] / oracle)
    for j in range(len(kept)):
        column = targets[:, j]
        mask = np.isnan(column)
        column[mask] = 10.0 if mask.all() else np.nanmax(column) + np.log(10.0)
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    tree.fit(np.array([features[cell] for cell in cells]), targets)
    return kept, tree


def _order(kept, tree, features):
    def choose(cell):
        predicted = tree.predict(np.array([features[cell]]))[0]
        return [kept[i] for i in np.argsort(predicted)] + [e for e in TRAIN_ENGINES if e not in kept]

    return choose


def _evaluate(choose, cells, records, protocol: str, cap: float = 1800.0):
    regrets, timeouts = [], 0
    rank1 = 0
    for cell in cells:
        times = _times(records, cell, protocol)
        if not times:
            continue
        oracle = min(times.values())
        chosen = choose(cell)[0]
        if chosen in times:
            regret = times[chosen] / oracle
            rank1 += regret == 1.0
        else:
            status = records.get((cell, protocol, chosen), {}).get("status", "missing")
            if status not in ("timeout", "error", "missing"):
                continue
            regret = cap / oracle
            timeouts += 1
        regrets.append(regret)
    regrets.sort()
    return {
        "cells": len(regrets),
        "median": statistics.median(regrets),
        "mean": statistics.mean(regrets),
        "p90": regrets[min(len(regrets) - 1, int(len(regrets) * 0.9))],
        "rank1": rank1 / len(regrets),
        "timeouts": timeouts,
    }


def main() -> None:
    records = _load_rows()
    cells = sorted({cell for cell, _, _ in records})
    features = {cell: cheap_features(build_cell(cell).query) for cell in cells}
    train_cells = [cell for cell in cells if _split(cell) == "train" and _host_order(cell) > SWITCH]
    val_cells = [cell for cell in cells if _split(cell) == "val" and _host_order(cell) > SWITCH]
    hprd_cells = [cell for cell in cells if _split(cell) == "hprd"]
    print(f"cells: train {len(train_cells)}  val {len(val_cells)}  hprd {len(hprd_cells)}")

    results = {}
    for protocol in ("one-shot", "amortized"):
        grid = []
        for depth, leaf in _GRID:
            kept, tree = _fit(train_cells, features, records, protocol, depth, leaf)
            metrics = _evaluate(_order(kept, tree, features), val_cells, records, protocol)
            grid.append({"depth": depth, "leaf": leaf, **metrics})
            print(
                f"{protocol:<10} depth {depth} leaf {leaf:<2} val mean {metrics['mean']:6.2f} "
                f"median {metrics['median']:5.2f} p90 {metrics['p90']:7.1f} timeouts {metrics['timeouts']}"
            )
        best = min(grid, key=lambda entry: entry["mean"])
        kept, tree = _fit(train_cells + val_cells, features, records, protocol, best["depth"], best["leaf"])
        hprd = _evaluate(_order(kept, tree, features), hprd_cells, records, protocol)
        print(f"{protocol:<10} selected depth {best['depth']} leaf {best['leaf']} | HPRD {hprd}")
        results[protocol] = {"grid": grid, "selected": best, "hprd": hprd, "leaves": tree.get_n_leaves()}

    # static baselines for context
    for protocol in ("one-shot", "amortized"):
        for engine in TRAIN_ENGINES:
            metrics = _evaluate(
                lambda _cell, e=engine: [e] + [x for x in TRAIN_ENGINES if x != e], val_cells, records, protocol
            )
            print(f"static {engine:<8} {protocol:<10} val mean {metrics['mean']:8.2f} median {metrics['median']:7.2f}")

    for protocol, suffix, literal_path in (
        ("one-shot", "one-shot", LITERAL_ONE),
        ("amortized", "prepared", LITERAL_PREP),
    ):
        best = results[protocol]["selected"]
        kept, tree = _fit(train_cells + val_cells, features, records, protocol, best["depth"], best["leaf"])
        literal_path.write_text(_literal(tree, kept, 0, 0) + "\n")
        print(f"wrote {literal_path} ({tree.get_n_leaves()} leaves) [protocol {suffix}]")
        results[protocol]["rule"] = _render(tree, kept, 0, 0)

    RESULT_JSON.write_text(json.dumps(results, indent=2, default=lambda value: value.item()))
    print(f"wrote {RESULT_JSON}")

    trees = {}
    for protocol in ("one-shot", "amortized"):
        best = results[protocol]["selected"]
        trees[protocol] = _fit(train_cells + val_cells, features, records, protocol, best["depth"], best["leaf"])
    joblib.dump(trees, RESULT_TREE)
    print(f"wrote {RESULT_TREE}")


if __name__ == "__main__":
    fire.Fire(main)
