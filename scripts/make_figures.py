"""Generate the two note figures from the sweep outputs.

Figure 1: served regret (portfolio vs best fixed engine) across regimes.
Figure 2: small real collections, regret by pattern size, showing that no
single static engine wins every k.

Usage: python scripts/make_figures.py   (run from the repository root)
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_policy import _dataset_queries  # noqa: E402
from large_cells import TRAIN_ENGINES  # noqa: E402

ROOT = Path(os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode"))
DATA = ROOT / "subiso_datasets"
SMALL_RUNS = ("subiso_dataset_sweeps_v5", "subiso_dataset_sweeps_v6", "subiso_dataset_sweeps_v7")
SOURCES = (("proteins", "PROTEINS"), ("enzymes", "ENZYMES"), ("mutag", "MUTAG"), ("aids", "AIDS"))
SMALL_ENGINES = ("vf2", "vf2pp", "lad", "vf3", "vf3l", "glasgow")
LARGE_REPEATS = (ROOT / "large_hosts_train" / "rows", ROOT / "large_var1" / "rows", ROOT / "large_var2" / "rows")
OUT = REPO / "docs" / "figures"


def load_times(directory: Path) -> dict[tuple[str, str], float]:
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in csv.DictReader((directory / "results.csv").open()):
        if row["operation"] == "contains":
            times[(row["cell"], row["engine"])].append(float(row["seconds"]))
    return {key: min(values) for key, values in times.items()}


def load_engine_rows(directory: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = {}
    for path in directory.glob("*.csv"):
        with path.open() as handle:
            row = next(csv.DictReader(handle))
        rows[(row["cell"], row["protocol"], row["engine"])] = row
    return rows


def small_by_k():
    """(protocol, k) -> {metric: [per-run values]} plus per-protocol static macro."""
    values: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    static_macro: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in SMALL_RUNS:
        for protocol, _prepared, suffix in (("one-shot", False, ""), ("prepared", True, "_amortized")):
            per_engine: dict[str, list[float]] = defaultdict(list)
            for source, upper in SOURCES:
                times = load_times(ROOT / run / f"{source}{suffix}")
                queries = _dataset_queries(upper, DATA)
                for cell, _ in queries.items():
                    available = {e: times[(cell, e)] for e in SMALL_ENGINES if (cell, e) in times}
                    if not available or (cell, "portfolio") not in times:
                        continue
                    oracle = min(available.values())
                    k = int(cell.rsplit("_k", 1)[1])
                    values[(protocol, k)]["portfolio"].append(times[(cell, "portfolio")] / oracle)
                    for engine, seconds in available.items():
                        values[(protocol, k)][engine].append(seconds / oracle)
                        per_engine[engine].append(seconds / oracle)
            for engine, regrets in per_engine.items():
                static_macro[(protocol, engine)].append(statistics.mean(regrets))
    return values, static_macro


def large_scores():
    """Per regime/protocol: served portfolio means (3 repeats) and best static mean."""
    engine_rows = load_engine_rows(LARGE_REPEATS[0])
    cells = sorted({cell for (cell, protocol, engine) in engine_rows if engine == "portfolio"})
    regimes = {
        "large": [cell for cell in cells if "_h13_" in cell],
        "hprd": [cell for cell in cells if cell.startswith("hprd_h3") or cell.startswith("hprd_h4")],
    }
    result = {}
    for regime, members in regimes.items():
        for protocol in ("one-shot", "amortized"):
            served_runs = []
            for repeat in LARGE_REPEATS:
                rows = load_engine_rows(repeat)
                regrets = []
                for cell in members:
                    times = {
                        e: float(engine_rows[(cell, protocol, e)]["seconds"])
                        for e in TRAIN_ENGINES
                        if (cell, protocol, e) in engine_rows and engine_rows[(cell, protocol, e)]["status"] == "ok"
                    }
                    port = rows.get((cell, protocol, "portfolio"))
                    if not times or port is None or port["status"] != "ok":
                        continue
                    regrets.append(float(port["seconds"]) / min(times.values()))
                served_runs.append(statistics.mean(regrets))
            static_means = {}
            for engine in TRAIN_ENGINES:
                regrets = []
                for cell in members:
                    times = {
                        e: float(engine_rows[(cell, protocol, e)]["seconds"])
                        for e in TRAIN_ENGINES
                        if (cell, protocol, e) in engine_rows and engine_rows[(cell, protocol, e)]["status"] == "ok"
                    }
                    if not times or engine not in times:
                        continue
                    regrets.append(times[engine] / min(times.values()))
                static_means[engine] = statistics.mean(regrets)
            best = min(static_means, key=static_means.get)
            result[(regime, protocol)] = {
                "served": statistics.mean(served_runs),
                "sd": statistics.stdev(served_runs),
                "static": static_means[best],
                "static_engine": best,
            }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    values, static_macro = small_by_k()
    large = large_scores()

    # ---- Figure 1: scoreboard -------------------------------------------
    small_served = {
        protocol: statistics.mean(value for k in (4, 6, 8) for value in values[(protocol, k)]["portfolio"])
        for protocol in ("one-shot", "prepared")
    }
    small_static = {}
    for protocol in ("one-shot", "prepared"):
        engine = min(SMALL_ENGINES, key=lambda name: statistics.mean(static_macro[(protocol, name)]))
        small_static[protocol] = statistics.mean(static_macro[(protocol, engine)])
        print(
            f"small {protocol}: best static {engine} {small_static[protocol]:.2f}, served {small_served[protocol]:.2f}"
        )

    groups = [
        "small\none-shot",
        "small\nprepared",
        "large\none-shot",
        "large\nprepared",
        "HPRD\none-shot",
        "HPRD\nprepared",
    ]
    served = [small_served["one-shot"], small_served["prepared"]]
    static = [small_static["one-shot"], small_static["prepared"]]
    served_sd = [0.0, 0.0]
    for regime in ("large", "hprd"):
        for protocol in ("one-shot", "prepared"):
            entry = large[(regime, "amortized" if protocol == "prepared" else protocol)]
            served.append(entry["served"])
            static.append(entry["static"])
            served_sd.append(entry["sd"])
            print(
                f"{regime} {protocol}: served {entry['served']:.2f} +/- {entry['sd']:.2f}, "
                f"best static {entry['static_engine']} {entry['static']:.2f}"
            )

    x = np.arange(len(groups))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 4.2), dpi=200)
    axis.bar(x - width / 2, served, width, label="portfolio (served)", color="#4C72B0", yerr=served_sd, capsize=3)
    axis.bar(x + width / 2, static, width, label="best fixed engine", color="#DD8452")
    axis.set_yscale("log")
    axis.set_ylabel("mean regret vs per-cell oracle (log)")
    axis.set_xticks(x, groups)
    axis.set_ylim(0.9, 30)
    axis.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    for index, (a, b) in enumerate(zip(served, static, strict=True)):
        axis.text(index - width / 2, (a + served_sd[index]) * 1.05, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        axis.text(index + width / 2, b * 1.05, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "scoreboard.png")
    print(f"wrote {OUT / 'scoreboard.png'}")

    # ---- Figure 2: small collections by k ------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharey=True)
    positions = np.arange(3)
    width = 0.26
    for axis, protocol in zip(axes, ("one-shot", "prepared"), strict=True):
        for offset, (engine, color) in enumerate((("vf2", "#4C72B0"), ("vf3l", "#DD8452"), ("portfolio", "#55A868"))):
            means = [statistics.mean(values[(protocol, k)][engine]) for k in (4, 6, 8)]
            axis.bar(
                positions + (offset - 1) * width,
                means,
                width,
                label=engine if protocol == "one-shot" else None,
                color=color,
            )
        axis.set_xticks(positions, ["k=4", "k=6", "k=8"])
        axis.set_xlabel(f"{protocol}: pattern size")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean regret vs oracle")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "by_pattern_size.png")
    print(f"wrote {OUT / 'by_pattern_size.png'}")


if __name__ == "__main__":
    fire.Fire(main)
