"""Train a dispatch policy from a benchmark sweep, Sublime-informed.

Usage::

    python scripts/train_dispatch.py /scratch/.../subiso_sweepD

Dispatch target is per-engine log time (never the query answer); features are
pre-match only.  Following Kuraya et al. (SIGMOD 2026) we add their two most
impactful features -- LDF candidate size and label ratio -- plus query
diameter/core, compare a RandomForest regressor (argmin) with an IncY
multi-label classifier, distil a depth-3 tree for interpretability, and charge
feature-extraction time to the portfolio.  Train on the smaller host orders,
test on the largest.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import fire
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from subiso.benchmark.grid import synthetic_cells

CELL_RE = re.compile(r"^n(\d+)_p([0-9.]+)_k(\d+)_(planted|random)_(induced|ordinary)(?:_l(\d+))?$")
ENGINES = ["vf2", "vf2pp", "lad", "vf3", "vf3l", "vf3p", "glasgow"]
FEATURE_NAMES = [
    "host_nodes",
    "host_edges",
    "host_density",
    "host_avg_degree",
    "host_max_degree",
    "host_core",
    "pattern_nodes",
    "pattern_edges",
    "pattern_density",
    "pattern_max_degree",
    "pattern_isolated",
    "pattern_connected",
    "pattern_core",
    "pattern_diameter",
    "density_ratio",
    "size_ratio",
    "induced",
    "labelled",
    "label_count",
    "label_match",
    "label_ratio",
    "candidate_size",
]


def _features(query) -> list[float]:
    host, pattern = query.target, query.pattern
    n, m = host.number_of_nodes(), host.number_of_edges()
    k, pk = pattern.number_of_nodes(), pattern.number_of_edges()
    host_density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    pattern_density = 2 * pk / (k * (k - 1)) if k > 1 else 0.0
    degrees = [degree for _, degree in host.degree()]
    pattern_degrees = [degree for _, degree in pattern.degree()]
    host_degree = dict(host.degree())

    label_count = 0
    label_match = 0.0
    label_ratio = 1.0
    candidates: list[int] = []
    if query.node_attr is not None:
        host_by_label: dict[object, list] = defaultdict(list)
        for node in host:
            host_by_label[host.nodes[node].get(query.node_attr)].append(node)
        label_count = len(host_by_label)
        frequency_terms = []
        for node in pattern:
            required = pattern.nodes[node].get(query.node_attr)
            sharing = host_by_label.get(required, [])
            frequency_terms.append(max(len(sharing), 1) / n)
            pattern_degree = pattern.degree(node)
            candidates.append(sum(1 for other in sharing if host_degree[other] >= pattern_degree))
        label_match = float(np.mean(np.log(frequency_terms))) if frequency_terms else 0.0
        shared_labels = set(host_by_label) & {pattern.nodes[node].get(query.node_attr) for node in pattern}
        label_ratio = sum(
            (sum(1 for u in pattern if pattern.nodes[u].get(query.node_attr) == label) / k)
            * (len(host_by_label[label]) / n)
            for label in shared_labels
        )
    else:
        for node in pattern:
            pattern_degree = pattern.degree(node)
            candidates.append(sum(1 for other in host if host_degree[other] >= pattern_degree))
    candidate_size = float(np.mean(candidates)) if candidates else 0.0

    pattern_core = max(nx.core_number(pattern).values()) if k else 0
    pattern_diameter = float(nx.diameter(pattern)) if k > 0 and nx.is_connected(pattern) else 0.0
    return [
        n,
        m,
        host_density,
        float(np.mean(degrees)) if degrees else 0.0,
        max(degrees, default=0),
        max(nx.core_number(host).values()) if n else 0,
        k,
        pk,
        pattern_density,
        max(pattern_degrees, default=0),
        sum(1 for degree in pattern_degrees if degree == 0),
        1.0 if k > 0 and nx.is_connected(pattern) else 0.0,
        pattern_core,
        pattern_diameter,
        pattern_density / max(host_density, 1e-9),
        k / max(n, 1),
        1.0 if query.semantics.value == "induced" else 0.0,
        1.0 if query.node_attr is not None else 0.0,
        label_count,
        label_match,
        label_ratio,
        candidate_size,
    ]


CHEAP_NAMES = [
    "host_nodes",
    "host_edges",
    "host_density",
    "pattern_nodes",
    "pattern_edges",
    "pattern_density",
    "density_ratio",
    "size_ratio",
    "induced",
    "labelled",
]


def _cheap_features(query) -> list[float]:
    """O(1) features: no iteration over vertices or edges."""
    host, pattern = query.target, query.pattern
    n, m = host.number_of_nodes(), host.number_of_edges()
    k, pk = pattern.number_of_nodes(), pattern.number_of_edges()
    host_density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    pattern_density = 2 * pk / (k * (k - 1)) if k > 1 else 0.0
    return [
        n,
        m,
        host_density,
        k,
        pk,
        pattern_density,
        pattern_density / max(host_density, 1e-9),
        k / max(n, 1),
        1.0 if query.semantics.value == "induced" else 0.0,
        1.0 if query.node_attr is not None else 0.0,
    ]



def _load(sweep: Path, operation: str):
    times: dict[tuple, list[float]] = defaultdict(list)
    hit: dict[str, bool] = {}
    for row in csv.DictReader((sweep / "results.csv").open()):
        if row["operation"] != operation or not CELL_RE.match(row["cell"]):
            continue
        times[(row["cell"], row["engine"])].append(float(row["seconds"]))
        if operation == "contains" and row["result_bool"]:
            hit[row["cell"]] = row["result_bool"] == "True"
    return {key: min(values) for key, values in times.items()}, hit


def _cost(times, cell: str, engine: str) -> float:
    value = times.get((cell, engine))
    if value is not None:
        return value
    for fallback in ("vf2", "glasgow", "lad"):
        value = times.get((cell, fallback))
        if value is not None:
            return value
    raise KeyError((cell, engine))


def _regret(costs: dict[str, float], oracle: dict[str, float]) -> dict[str, float]:
    ratios = sorted(costs[cell] / oracle[cell] for cell in costs)
    return {
        "median": statistics.median(ratios),
        "mean": sum(ratios) / len(ratios),
        "p90": ratios[min(len(ratios) - 1, int(len(ratios) * 0.9))],
        "max": ratios[-1],
    }


def _mrr(choices: dict[str, str], times, cells: list[str]) -> float:
    values = []
    for name in cells:
        ranked = sorted((engine for engine in ENGINES if (name, engine) in times), key=lambda e: times[(name, e)])
        chosen = choices[name]
        chosen_cost = _cost(times, name, chosen)
        rank = 1 + sum(1 for engine in ranked if times[(name, engine)] < chosen_cost)
        values.append(1 / rank)
    return sum(values) / len(values)


def main(sweep_directory: str) -> None:
    sweep = Path(sweep_directory)
    manifest = json.loads((sweep / "manifest.json").read_text())
    parameters = manifest["parameters"]
    times, hit = _load(sweep, "contains")

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
    names = [cell.name for cell in cells]
    features: dict[str, list[float]] = {}
    feature_seconds: dict[str, float] = {}
    for cell in cells:
        best_time = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            values = _features(cell.query)
            best_time = min(best_time, time.perf_counter() - start)
        features[cell.name] = values
        feature_seconds[cell.name] = best_time

    orders = {name: int(CELL_RE.match(name).group(1)) for name in names}
    train_names = [name for name in names if orders[name] < max(orders.values())]
    test_names = [name for name in names if orders[name] == max(orders.values())]
    x_train = np.array([features[name] for name in train_names])
    print(f"== feature extraction: {statistics.median(feature_seconds.values()) * 1000:.3f} ms per query (median) ==")

    # ---- 1. hit/miss classifier (diagnostic only, never a dispatch target) ----
    y_all = np.array([1 if hit.get(name) else 0 for name in names])
    x_all = np.array([features[name] for name in names])
    hit_model = RandomForestClassifier(n_estimators=400, max_depth=5, random_state=0)
    cross_validated = cross_val_predict(hit_model, x_all, y_all, cv=5)
    tn, fp, fn, tp = confusion_matrix(y_all, cross_validated, labels=[0, 1]).ravel()
    print("== hit/miss classifier (diagnostic, 5-fold CV) ==")
    print(f"   misses {int((y_all == 0).sum())}/{len(names)}: detected {tn}/{tn + fp}, false alarms {fn}/{fn + tp}")
    hit_model.fit(x_all, y_all)
    importances = sorted(zip(FEATURE_NAMES, hit_model.feature_importances_, strict=False), key=lambda kv: -kv[1])[:6]
    print("   top features: " + ", ".join(f"{name} {value:.2f}" for name, value in importances))

    cheap_features: dict[str, list[float]] = {}
    cheap_seconds: dict[str, float] = {}
    for cell in cells:
        best_time = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            values = _cheap_features(cell.query)
            best_time = min(best_time, time.perf_counter() - start)
        cheap_features[cell.name] = values
        cheap_seconds[cell.name] = best_time
    print(
        f"== cheap features: {statistics.median(cheap_seconds.values()) * 1000:.4f} ms per query (median) =="
    )

    # ---- 2. per-engine regressors + distilled tree ----
    models: dict[str, RandomForestRegressor] = {}
    regressor_importances = np.zeros(len(FEATURE_NAMES))
    for engine in ENGINES:
        rows = [name for name in train_names if (name, engine) in times]
        if len(rows) < 10:
            continue
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
        model.fit(np.array([features[name] for name in rows]), np.log([times[(name, engine)] for name in rows]))
        models[engine] = model
        regressor_importances += model.feature_importances_
    top_regressor = sorted(zip(FEATURE_NAMES, regressor_importances, strict=False), key=lambda kv: -kv[1])[:8]
    print("\n== dispatch regressor feature importances ==")
    print("   " + ", ".join(f"{name} {value:.2f}" for name, value in top_regressor))

    def forest_choice(name: str) -> str:
        predicted = {engine: model.predict(np.array([features[name]]))[0] for engine, model in models.items()}
        return min(predicted, key=predicted.get)

    y_forest = [ENGINES.index(forest_choice(name)) for name in train_names]
    surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
    surrogate.fit(x_train, y_forest)
    tree_text = export_text(surrogate, feature_names=FEATURE_NAMES, decimals=2)
    for class_index, engine in enumerate(ENGINES):
        tree_text = tree_text.replace(f"class: {class_index}\n", f"class: {engine}\n")
    print("\n== distilled rule (depth 3, imitating the regressors) ==")
    print(tree_text)

    # ---- 3. IncY multi-label classifier (Sublime's labeling idea) ----
    y_incy = np.zeros((len(train_names), len(ENGINES)))
    for row, name in enumerate(train_names):
        available = [engine for engine in ENGINES if (name, engine) in times]
        best = min(times[(name, engine)] for engine in available)
        for engine in available:
            if times[(name, engine)] <= best / 0.7:  # within ~1.43x of the best
                y_incy[row, ENGINES.index(engine)] = 1
    incy_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0)
    incy_model.fit(x_train, y_incy)

    def incy_choice(name: str) -> str:
        probabilities = [p[0][1] for p in incy_model.predict_proba(np.array([features[name]]))]
        return ENGINES[int(np.argmax(probabilities))]

    # ---- 4. held-out evaluation (test = largest order) ----
    oracle = {name: min(times[(name, engine)] for engine in ENGINES if (name, engine) in times) for name in test_names}

    def evaluate(
        label: str,
        choices: dict[str, str],
        charge_features: bool = False,
        cheap_cost: bool = False,
    ) -> None:
        costs = {}
        for name in test_names:
            overhead = feature_seconds[name] if charge_features else (cheap_seconds[name] if cheap_cost else 0.0)
            costs[name] = _cost(times, name, choices[name]) + overhead
        metrics = _regret(costs, oracle)
        rank = _mrr(choices, times, test_names)
        print(
            f"   {label:<34} mrr {rank:5.2f}  med {metrics['median']:5.2f}  mean {metrics['mean']:6.2f}  "
            f"p90 {metrics['p90']:6.2f}  max {metrics['max']:7.1f}"
        )

    cheap_train = [cheap_features[name] for name in train_names]
    cheap_models: dict[str, RandomForestRegressor] = {}
    for engine in ENGINES:
        rows = [name for name in train_names if (name, engine) in times]
        if len(rows) < 10:
            continue
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
        model.fit(np.array([cheap_features[name] for name in rows]), np.log([times[(name, engine)] for name in rows]))
        cheap_models[engine] = model

    def cheap_choice(name: str) -> str:
        predicted = {engine: model.predict(np.array([cheap_features[name]]))[0] for engine, model in cheap_models.items()}
        return min(predicted, key=predicted.get)

    y_cheap = [ENGINES.index(cheap_choice(name)) for name in train_names]
    cheap_surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
    cheap_surrogate.fit(np.array(cheap_train), y_cheap)
    cheap_tree_text = export_text(cheap_surrogate, feature_names=CHEAP_NAMES, decimals=3)
    for class_index, engine in enumerate(ENGINES):
        cheap_tree_text = cheap_tree_text.replace(f"class: {class_index}\n", f"class: {engine}\n")
    print("\n== distilled rule on cheap features (depth 3) ==")
    print(cheap_tree_text)

    tree_choices = {name: ENGINES[int(surrogate.predict(np.array([features[name]]))[0])] for name in test_names}
    cheap_choices = {
        name: ENGINES[int(cheap_surrogate.predict(np.array([cheap_features[name]]))[0])] for name in test_names
    }
    inc_choices = {name: incy_choice(name) for name in test_names}
    print("\n== held-out evaluation (test = largest order) ==")
    evaluate("forest argmin", {name: forest_choice(name) for name in test_names})
    evaluate("distilled depth-3 tree", tree_choices)
    evaluate("incy classifier", inc_choices)
    evaluate("forest argmin (cheap)", {name: cheap_choice(name) for name in test_names})
    evaluate("cheap depth-3 tree", cheap_choices)
    evaluate("cheap tree + feature cost", cheap_choices, charge_features=False, cheap_cost=True)
    evaluate("distilled tree + feature cost", tree_choices, charge_features=True)
    evaluate("incY + feature cost", inc_choices, charge_features=True)
    evaluate("static vf2", dict.fromkeys(test_names, "vf2"))
    evaluate("recorded portfolio", dict.fromkeys(test_names, "portfolio"))

    # ---- 5. transfer to first_mapping with the same choices ----
    mapping_times, _ = _load(sweep, "first_mapping")
    if mapping_times:
        mapping_oracle = {
            name: min(mapping_times[(name, engine)] for engine in ENGINES if (name, engine) in mapping_times)
            for name in test_names
            if any((name, engine) in mapping_times for engine in ENGINES)
        }
        mapping_names = [name for name in test_names if name in mapping_oracle]
        for label, choices in (("distilled tree", tree_choices), ("incy classifier", inc_choices)):
            costs = {name: _cost(mapping_times, name, choices[name]) for name in mapping_names}
            metrics = _regret(costs, mapping_oracle)
            rank = _mrr({name: choices[name] for name in mapping_names}, mapping_times, mapping_names)
            print(
                f"   first_mapping {label:<20} mrr {rank:5.2f}  med {metrics['median']:5.2f}  "
                f"mean {metrics['mean']:6.2f}  p90 {metrics['p90']:6.2f}  max {metrics['max']:7.1f}"
            )

    # ---- 6. where misses live ----
    misses = [name for name in names if hit.get(name) is False]
    print(f"\n== misses: {len(misses)}/{len(names)} cells (all random patterns) ==")
    breakdown: dict[tuple, int] = defaultdict(int)
    for name in misses:
        match = CELL_RE.match(name)
        breakdown[(match.group(2), match.group(3), match.group(6) or "0")] += 1
    for key, count in sorted(breakdown.items()):
        print(f"   host p={key[0]:<5} k={key[1]} labels={key[2]:<2} : {count}")


if __name__ == "__main__":
    fire.Fire(main)
