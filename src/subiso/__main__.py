"""CLI: ``python -m subiso capabilities`` and ``python -m subiso check``."""

from __future__ import annotations

from dataclasses import fields

import fire
import networkx as nx

from .dispatch import is_fast_path_enabled, set_fast_path
from .matcher import Query
from .portfolio import Portfolio
from .semantics import Semantics


def capabilities(fast_path: bool = False) -> None:
    """Print every configured matcher, its availability and capability flags."""
    if fast_path:
        set_fast_path(True)
    portfolio = Portfolio()
    for matcher in portfolio._matchers:  # noqa: SLF001 - the CLI inspects its own portfolio
        flags = [field.name for field in fields(matcher.capabilities) if getattr(matcher.capabilities, field.name)]
        print(f"{matcher.name:<10} available={str(matcher.available()):<5} {', '.join(flags)}")
    supported = [semantics.value for semantics in Semantics if portfolio.capabilities.supports(semantics)]
    print(f"\nportfolio semantics: {supported}")
    if is_fast_path_enabled():
        print("fast-path router: ENABLED")
    if not portfolio.capabilities.count_embeddings:
        print("note: no embedding-count engine installed (install subiso[engines] or subiso-glasgow).")


def check(fast_path: bool = False) -> None:
    """Compare the portfolio against NetworkX brute force on the small graph atlas."""
    if fast_path:
        set_fast_path(True)
    patterns = [nx.path_graph(3), nx.cycle_graph(4), nx.star_graph(3)]

    def brute(host: nx.Graph, pattern: nx.Graph, induced: bool) -> set[frozenset[int]]:
        matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
        iterator = matcher.subgraph_isomorphisms_iter() if induced else matcher.subgraph_monomorphisms_iter()
        return {frozenset(mapping) for mapping in iterator}

    portfolio = Portfolio()
    checked = 0
    for host in nx.graph_atlas_g():
        if host.number_of_nodes() > 6:
            continue
        for pattern in patterns:
            for induced in (True, False):
                semantics = Semantics.INDUCED if induced else Semantics.ORDINARY
                expected = brute(host, pattern, induced)
                assert portfolio.contains(Query(host, pattern, semantics)) == bool(expected)
                assert portfolio.count(Query(host, pattern, semantics)) == len(expected)
                checked += 1
    print(f"check: {checked} queries agreed with brute force")


if __name__ == "__main__":
    fire.Fire({"capabilities": capabilities, "check": check})
