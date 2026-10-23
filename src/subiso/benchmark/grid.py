"""Synthetic sweep construction over the dataset generators."""

from __future__ import annotations

import itertools

from ..datasets.synthetic import gnp_host, planted_pattern, random_pattern
from ..matcher import Query
from ..semantics import Semantics
from .harness import Cell

MISS_BOOST = 0.75


def synthetic_cells(
    orders: tuple[int, ...],
    densities: tuple[float, ...],
    pattern_orders: tuple[int, ...],
    *,
    induced_relations: tuple[bool, ...] = (True, False),
    label_counts: tuple[int, ...] = (0,),
    pattern_sources: tuple[str, ...] = ("planted", "random"),
    seed: int = 0,
    miss_boost: float = MISS_BOOST,
) -> list[Cell]:
    """Build seeded cells: host order x density x pattern order x relation x labels x source."""
    cells: list[Cell] = []
    combinations = list(
        itertools.product(orders, densities, pattern_orders, induced_relations, label_counts, pattern_sources)
    )
    for index, (order, density, pattern_order, induced, label_count, source) in enumerate(combinations):
        if pattern_order > order:
            continue
        cell_seed = seed + 1_000_003 * index
        host = gnp_host(order, density, seed=cell_seed, label_count=label_count)
        if source == "planted":
            pattern = planted_pattern(host, pattern_order, seed=cell_seed + 1)
        else:
            pattern = random_pattern(
                pattern_order, density, seed=cell_seed + 1, label_count=label_count, miss_boost=miss_boost
            )
        semantics = Semantics.INDUCED if induced else Semantics.ORDINARY
        node_attr = "label" if label_count else None
        name = f"n{order}_p{density:g}_k{pattern_order}_{source}_{semantics.value}"
        if label_count:
            name += f"_l{label_count}"
        cells.append(Cell(name=name, query=Query(host, pattern, semantics, node_attr=node_attr)))
    return cells
