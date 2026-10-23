"""Run one benchmark sweep per TU dataset with fragment patterns.

Usage: python scripts/dataset_sweep.py --datasets PROTEINS,ENZYMES,MUTAG,AIDS
"""

from __future__ import annotations

import os

import fire
from hosts import eligible_hosts

from subiso.benchmark.__main__ import default_engines
from subiso.benchmark.harness import Cell, run_sweep
from subiso.datasets.molecules import fragment_pattern
from subiso.datasets.tudataset import load, sample
from subiso.matcher import Query
from subiso.portfolio import Portfolio
from subiso.semantics import Semantics

SWEEP_ROOT = os.environ.get("SUBISO_SWEEP_ROOT", "/scratch/lukas/tmp/opencode")
ENGINE_KEEP = {"vf2", "vf2pp", "lad", "vf3", "vf3l", "vf3p", "glasgow"}


def _tokens(values) -> list[str]:
    if isinstance(values, str):
        return [token.strip() for token in values.split(",") if token.strip()]
    return [str(value) for value in values]


def main(
    datasets="PROTEINS,ENZYMES,MUTAG,AIDS",
    limit=1000,
    host_count=20,
    pattern_sizes="4,6,8",
    labelled=False,
    repeats=2,
    warmups=1,
    protocol="one-shot",
    data_directory=f"{SWEEP_ROOT}/subiso_datasets",
    output_root=f"{SWEEP_ROOT}/subiso_dataset_sweeps",
) -> None:
    sizes = [int(token) for token in _tokens(pattern_sizes)]
    for name in _tokens(datasets):
        graphs = load(name, data_directory, limit=limit)
        chosen = sample(eligible_hosts(graphs, tuple(sizes)), host_count, seed=0)
        cells = []
        for index, host in enumerate(chosen):
            for k in sizes:
                pattern = fragment_pattern(host, k, seed=index * 10 + k)
                cells.append(
                    Cell(
                        f"{name.lower()}_{index:02d}_k{k}",
                        Query(host, pattern, Semantics.INDUCED, node_attr="label" if labelled else None),
                    )
                )
        engines = [matcher for matcher in default_engines() if matcher.name in ENGINE_KEEP]
        engines.append(Portfolio(matchers=engines))
        suffix = "_labelled" if labelled else ""
        amortized_suffix = "_amortized" if protocol == "amortized" else ""
        run_sweep(
            cells,
            engines,
            operations=("contains", "first_mapping"),
            repeats=repeats,
            warmups=warmups,
            protocol=protocol,
            output_directory=f"{output_root}/{name.lower()}{suffix}{amortized_suffix}",
        )


if __name__ == "__main__":
    fire.Fire(main)
