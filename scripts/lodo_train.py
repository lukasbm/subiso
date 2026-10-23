"""Leave-one-source-out retraining and export of the dispatch policy.

The shipped policy (``subiso.dispatch``) is one regression tree over the same
cheap O(1) features, trained directly on per-engine log time ratios
``log(t_e / oracle)``; there is one tree per protocol (``prepared`` selects
between them), the leaf decision is the predicted argmin, which minimizes
expected log-regret within the leaf's region.  The superseded per-engine
forests and the argmin-label classifier are kept as one-shot baselines so the
comparison stays reproducible.

Fixed hyperparameters, no nested selection: depth 6 and min_samples_leaf 5,
chosen from a small depth x leaf-size grid on the LODO macro (write-up,
`docs/writeup.md`, records the sweep).

Usage: python scripts/lodo_train.py   (run from the repository root)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import fire
import joblib
import numpy as np
from evaluate_policy import _dataset_queries, _synthetic_queries
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from subiso.dispatch import CHEAP_FEATURE_NAMES, cheap_features
from subiso.matcher import Query

ROOT = Path(os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode"))
DATA = ROOT / "subiso_datasets"
KEEP = ("vf2", "vf2pp", "lad", "vf3", "vf3l", "vf3p", "glasgow")
# Per-protocol fixed hyperparameters, chosen from the small depth x leaf grid
# on the LODO macro of the corrected sweeps (docs/writeup.md records the grid).
ONE_SHOT_DEPTH, ONE_SHOT_LEAF = 6, 10
AMORTIZED_DEPTH, AMORTIZED_LEAF = 6, 3
RESULT_JSON = ROOT / "lodo_results.json"
RESULT_RULE = ROOT / "lodo_rule.txt"
RESULT_LITERAL_ONE_SHOT = ROOT / "lodo_rule_literal_oneshot.txt"
RESULT_LITERAL_AMORTIZED = ROOT / "lodo_rule_literal_amortized.txt"
RESULT_TREE = ROOT / "lodo_trees.joblib"

Row = tuple[str, bool, str]  # (source, prepared, cell)


def _load(sweep: Path) -> dict[tuple[str, str], float]:
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in csv.DictReader((sweep / "results.csv").open()):
        if row["operation"] == "contains":
            times[(row["cell"], row["engine"])].append(float(row["seconds"]))
    return {key: min(values) for key, values in times.items()}


def _units() -> list[tuple[str, bool, Path, dict[str, Query]]]:
    """Every (source, protocol) sweep holding contains times for the shared cells."""
    units = [
        ("sweepD", False, ROOT / "subiso_sweepD_fixed", _synthetic_queries(ROOT / "subiso_sweepD_fixed")),
        (
            "sweepD",
            True,
            ROOT / "subiso_sweepD_fixed_amortized",
            _synthetic_queries(ROOT / "subiso_sweepD_fixed_amortized"),
        ),
    ]
    for name in ("PROTEINS", "ENZYMES", "MUTAG", "AIDS"):
        lower = name.lower()
        units.append((lower, False, ROOT / "subiso_dataset_sweeps_fixed" / lower, _dataset_queries(name, DATA)))
        units.append(
            (lower, True, ROOT / "subiso_dataset_sweeps_fixed" / f"{lower}_amortized", _dataset_queries(name, DATA))
        )
    return units


def _complete(order):
    return order + [engine for engine in KEEP if engine not in order]


def _fit_tree(rows, features, times, depth: int, leaf: int):
    """Regression tree on log(t_e / oracle); median-imputed where gated."""
    columns = [j for j, engine in enumerate(KEEP) if any((row[2], engine) in times[(row[0], row[1])] for row in rows)]
    kept = [KEEP[j] for j in columns]
    targets = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        table = times[(row[0], row[1])]
        available = [engine for engine in KEEP if (row[2], engine) in table]
        oracle = min(table[(row[2], engine)] for engine in available)
        for j, engine in enumerate(kept):
            value = table.get((row[2], engine))
            if value is not None:
                targets[i, j] = np.log(value / oracle)
    for j in range(len(kept)):
        column = targets[:, j]
        mask = np.isnan(column)
        assert not mask.all(), f"engine {kept[j]} has no recorded time in the training rows"
        if mask.any():
            column[mask] = np.nanmedian(column)
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    tree.fit(np.array([features[row] for row in rows]), targets)
    return kept, tree


def _fit_forests(rows, features, times):
    """Superseded baseline: per-engine random forests on log time."""
    models = {}
    for engine in KEEP:
        engine_rows = [row for row in rows if (row[2], engine) in times[(row[0], row[1])]]
        if len(engine_rows) >= 10:
            model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
            values = [times[(row[0], row[1])][(row[2], engine)] for row in engine_rows]
            model.fit(np.array([features[row] for row in engine_rows]), np.log(values))
            models[engine] = model
    return models


def _fit_class_tree(rows, features, forests):
    """Superseded baseline: depth-3 classification on the forest's argmin labels."""
    labels = []
    for row in rows:
        predicted = {engine: model.predict(np.array([features[row]]))[0] for engine, model in forests.items()}
        labels.append(KEEP.index(min(predicted, key=predicted.get)))
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(np.array([features[row] for row in rows]), labels)
    return tree


def _forest_choose(models, features):
    def choose(row):
        predicted = {engine: model.predict(np.array([features[row]]))[0] for engine, model in models.items()}
        return _complete(sorted(predicted, key=predicted.get))

    return choose


def _class_choose(tree, features):
    def choose(row):
        proba = tree.predict_proba(np.array([features[row]]))[0]
        ranked = sorted(range(len(tree.classes_)), key=lambda i: -proba[i])
        return _complete([KEEP[tree.classes_[i]] for i in ranked])

    return choose


def _tree_choose(kept, tree, features):
    def choose(row):
        predicted = tree.predict(np.array([features[row]]))[0]
        return _complete([kept[i] for i in np.argsort(predicted)])

    return choose


def _score(choose, rows, times):
    regrets, ranks = [], []
    for row in rows:
        table = times[(row[0], row[1])]
        available = [engine for engine in KEEP if (row[2], engine) in table]
        if not available:
            continue
        chosen = next((engine for engine in choose(row) if (row[2], engine) in table), None)
        if chosen is None:
            continue
        best = min(table[(row[2], engine)] for engine in available)
        regrets.append(table[(row[2], chosen)] / best)
        ranks.append(1.0 / (1 + sum(1 for engine in available if table[(row[2], engine)] < table[(row[2], chosen)])))
    regrets.sort()
    return {
        "cells": len(regrets),
        "median": statistics.median(regrets),
        "mean": sum(regrets) / len(regrets),
        "p90": regrets[min(len(regrets) - 1, int(len(regrets) * 0.9))],
        "mrr": sum(ranks) / len(ranks),
    }


def _leaf_order(tree, kept, node):
    order = [kept[i] for i in np.argsort(tree.tree_.value[node].ravel())]
    return _complete(order)


def _render(tree, kept, node, indent):
    """Human-readable rule: conditions plus the leaf's engine ranking."""
    if tree.tree_.children_left[node] == -1:
        return "  " * indent + "-> " + ", ".join(_leaf_order(tree, kept, node))
    feature = tree.tree_.feature[node]
    threshold = tree.tree_.threshold[node]
    return "\n".join(
        [
            f"{'  ' * indent}if {CHEAP_FEATURE_NAMES[feature]} <= {threshold:.6g}:",
            _render(tree, kept, tree.tree_.children_left[node], indent + 1),
            f"{'  ' * indent}else:",
            _render(tree, kept, tree.tree_.children_right[node], indent + 1),
        ]
    )


def _literal(tree, kept, node, indent):
    """Python literal pasted into subiso/dispatch.py (``_Split`` nodes, leaf ranking tuples)."""
    if tree.tree_.children_left[node] == -1:
        return "(" + ", ".join(f'"{engine}"' for engine in _leaf_order(tree, kept, node)) + ")"
    feature = tree.tree_.feature[node]
    threshold = tree.tree_.threshold[node]
    pad = "    " * (indent + 1)
    below = _literal(tree, kept, tree.tree_.children_left[node], indent + 1)
    above = _literal(tree, kept, tree.tree_.children_right[node], indent + 1)
    return f"_Split(\n{pad}{feature},\n{pad}{threshold:.6g},\n{pad}{below},\n{pad}{above},\n{'    ' * indent})"


def main() -> None:
    units = _units()
    features: dict[Row, list[float]] = {}
    times: dict[tuple[str, bool], dict[tuple[str, str], float]] = {}
    for source, prepared, sweep, queries in units:
        times[(source, prepared)] = _load(sweep)
        for cell, query in queries.items():
            features[(source, prepared, cell)] = cheap_features(query)

    rows = list(features)
    sources = ["sweepD", "proteins", "enzymes", "mutag", "aids"]
    report: dict[str, dict[str, dict[str, float]]] = {
        "forest": {},
        "class-d3": {},
        "tree-one-shot": {},
        "tree-amortized": {},
    }
    for held_out in sources:
        train_rows = [row for row in rows if row[0] != held_out]
        train_one_shot = [row for row in train_rows if not row[1]]
        test_one_shot = [row for row in rows if row[0] == held_out and not row[1]]
        test_amortized = [row for row in rows if row[0] == held_out and row[1]]
        train_amortized = [row for row in train_rows if row[1]]
        forests = _fit_forests(train_one_shot, features, times)
        class_tree = _fit_class_tree(train_one_shot, features, forests)
        kept_one, tree_one = _fit_tree(train_one_shot, features, times, ONE_SHOT_DEPTH, ONE_SHOT_LEAF)
        kept_amortized, tree_amortized = _fit_tree(train_amortized, features, times, AMORTIZED_DEPTH, AMORTIZED_LEAF)
        report["forest"][held_out] = _score(_forest_choose(forests, features), test_one_shot, times)
        report["class-d3"][held_out] = _score(_class_choose(class_tree, features), test_one_shot, times)
        report["tree-one-shot"][held_out] = _score(_tree_choose(kept_one, tree_one, features), test_one_shot, times)
        report["tree-amortized"][held_out] = _score(
            _tree_choose(kept_amortized, tree_amortized, features), test_amortized, times
        )
        print(
            f"LODO {held_out:<9} "
            f"forest {report['forest'][held_out]['mean']:.2f}  "
            f"class-d3 {report['class-d3'][held_out]['mean']:.2f}  "
            f"tree one-shot {report['tree-one-shot'][held_out]['mean']:.2f}  "
            f"tree amortized {report['tree-amortized'][held_out]['mean']:.2f}",
            flush=True,
        )

    for model, entries in report.items():
        macro_mean = statistics.mean(entry["mean"] for entry in entries.values())
        macro_mrr = statistics.mean(entry["mrr"] for entry in entries.values())
        print(f"macro {model:<14} mean {macro_mean:.3f}  mrr {macro_mrr:.3f}")
    RESULT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {RESULT_JSON}")

    all_one_shot = [row for row in rows if not row[1]]
    all_amortized = [row for row in rows if row[1]]
    kept_one, tree_one = _fit_tree(all_one_shot, features, times, ONE_SHOT_DEPTH, ONE_SHOT_LEAF)
    kept_amortized, tree_amortized = _fit_tree(all_amortized, features, times, AMORTIZED_DEPTH, AMORTIZED_LEAF)
    print(f"pooled one-shot tree: depth {ONE_SHOT_DEPTH}, min leaf {ONE_SHOT_LEAF}, leaves {tree_one.get_n_leaves()}")
    print(
        f"pooled amortized tree: depth {AMORTIZED_DEPTH}, min leaf {AMORTIZED_LEAF}, "
        f"leaves {tree_amortized.get_n_leaves()}"
    )
    rule = (
        f"one-shot:\n{_render(tree_one, kept_one, 0, 0)}\n\namortized:\n{_render(tree_amortized, kept_amortized, 0, 0)}"
    )
    print(rule)
    RESULT_RULE.write_text(rule)
    print(f"wrote {RESULT_RULE}")
    RESULT_LITERAL_ONE_SHOT.write_text(_literal(tree_one, kept_one, 0, 0) + "\n")
    RESULT_LITERAL_AMORTIZED.write_text(_literal(tree_amortized, kept_amortized, 0, 0) + "\n")
    print(f"wrote {RESULT_LITERAL_ONE_SHOT}")
    print(f"wrote {RESULT_LITERAL_AMORTIZED}")
    joblib.dump(
        {"kept": kept_one, "tree": tree_one, "kept_amortized": kept_amortized, "tree_amortized": tree_amortized},
        RESULT_TREE,
    )
    print(f"wrote {RESULT_TREE}")


if __name__ == "__main__":
    fire.Fire(main)
