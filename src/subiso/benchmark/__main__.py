"""CLI: ``python -m subiso.benchmark`` runs a synthetic sweep.

Fire options accept comma strings or repeated flags, e.g.::

    python -m subiso.benchmark --orders 50,100 --densities 0.1,0.2 --pattern_orders 3,4
"""

from __future__ import annotations

from collections.abc import Sequence

import fire

from ..backends.fastiso import FastisoMatcher
from ..backends.lad import LadMatcher
from ..backends.subgraphmatching import ALGORITHMS, SubgraphMatchingMatcher
from ..backends.vf2 import Vf2Matcher
from ..backends.vf2pp import Vf2ppMatcher
from ..backends.vf3 import Vf3lMatcher, Vf3Matcher, Vf3pMatcher
from ..matcher import Matcher
from ..portfolio import Portfolio
from .grid import synthetic_cells
from .harness import run_sweep


def _tokens(values: str | Sequence) -> list[str]:
    if isinstance(values, str):
        return [token.strip() for token in values.replace(",", " ").split() if token.strip()]
    if isinstance(values, (list, tuple)):
        return [str(value).strip() for value in values]
    return [str(values).strip()]


def _ints(values: str | Sequence) -> tuple[int, ...]:
    return tuple(int(token) for token in _tokens(values))


def _floats(values: str | Sequence) -> tuple[float, ...]:
    return tuple(float(token) for token in _tokens(values))


def _relations(values: str | Sequence) -> tuple[bool, ...]:
    tokens = {token.lower() for token in _tokens(values)}
    if tokens <= {"both"} or tokens >= {"induced", "ordinary"}:
        return (True, False)
    if tokens == {"induced"}:
        return (True,)
    if tokens == {"ordinary"}:
        return (False,)
    raise ValueError(f"unknown relation selection {sorted(tokens)}; expected both, induced, or ordinary.")


def default_engines() -> list[Matcher]:
    """Every importable engine; Glasgow and the MIT suite join when available."""
    engines: list[Matcher] = [Vf2Matcher(), Vf2ppMatcher(), LadMatcher()]
    vf3 = Vf3Matcher()
    if vf3.available():
        engines.extend([vf3, Vf3lMatcher(), Vf3pMatcher()])
    fastiso = FastisoMatcher()
    if fastiso.available():
        engines.append(fastiso)
    try:
        from subiso_glasgow import GlasgowMatcher

        engines.append(GlasgowMatcher())
    except ImportError:
        pass
    suite = [SubgraphMatchingMatcher(algorithm) for algorithm in ALGORITHMS]
    engines.extend(matcher for matcher in suite if matcher.available())
    return engines


def main(
    orders: str | Sequence = "30,50",
    densities: str | Sequence = "0.1,0.2",
    pattern_orders: str | Sequence = "3,4,5",
    label_counts: str | Sequence = "0",
    induced: str | Sequence = "both",
    pattern_sources: str | Sequence = "planted,random",
    operations: str | Sequence = "contains,count,first_mapping",
    engines: str | Sequence = "",
    exclude_engines: str | Sequence = "",
    protocol: str = "one-shot",
    repeats: int = 3,
    warmups: int = 1,
    seed: int = 0,
    miss_boost: float = 0.75,
    output_directory: str = "",
) -> None:
    """Time every capable engine plus the portfolio over a synthetic sweep."""
    host_orders = _ints(orders)
    density_values = _floats(densities)
    pattern_order_values = _ints(pattern_orders)
    label_count_values = _ints(label_counts)
    relation_values = _relations(induced)
    source_values = tuple(token.lower() for token in _tokens(pattern_sources))
    operation_values = tuple(token.lower() for token in _tokens(operations))
    assert set(source_values) <= {"planted", "random"}
    assert all(value >= 0 for value in label_count_values)
    assert protocol in ("one-shot", "amortized"), f"unknown protocol {protocol!r}"

    selected = default_engines()
    if _tokens(engines):
        wanted = set(_tokens(engines))
        selected = [matcher for matcher in selected if matcher.name in wanted]
    if _tokens(exclude_engines):
        unwanted = set(_tokens(exclude_engines))
        selected = [matcher for matcher in selected if matcher.name not in unwanted]
    selected.append(Portfolio(matchers=selected))
    cells = synthetic_cells(
        host_orders,
        density_values,
        pattern_order_values,
        induced_relations=relation_values,
        label_counts=label_count_values,
        pattern_sources=source_values,
        seed=seed,
        miss_boost=miss_boost,
    )
    run_sweep(
        cells,
        selected,
        operations=operation_values,
        repeats=repeats,
        warmups=warmups,
        protocol=protocol,
        output_directory=output_directory or None,
        parameters={
            "orders": host_orders,
            "densities": density_values,
            "pattern_orders": pattern_order_values,
            "label_counts": label_count_values,
            "induced": relation_values,
            "pattern_sources": source_values,
            "operations": operation_values,
            "engines": [matcher.name for matcher in selected],
            "protocol": protocol,
            "repeats": repeats,
            "warmups": warmups,
            "seed": seed,
            "miss_boost": miss_boost,
        },
    )


if __name__ == "__main__":
    fire.Fire(main)
