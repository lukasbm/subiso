"""Prepared-target parity: amortized host conversion must not change answers."""

from __future__ import annotations

import networkx as nx

from subiso.backends.lad import LadMatcher
from subiso.backends.vf2 import Vf2Matcher
from subiso.datasets.synthetic import gnp_host
from subiso.matcher import Query
from subiso.portfolio import Portfolio
from subiso.semantics import Semantics


def _engines():
    engines = [Vf2Matcher(), LadMatcher()]
    try:
        from subiso_glasgow import GlasgowMatcher

        engines.append(GlasgowMatcher())
    except ImportError:
        pass
    return engines


def test_prepared_targets_agree_with_raw():
    host = gnp_host(30, 0.3, seed=3)
    pattern = nx.path_graph(4)
    for engine in _engines():
        prepared = engine.prepare(host)
        for semantics in (Semantics.INDUCED, Semantics.ORDINARY):
            raw = Query(host, pattern, semantics)
            amortized = Query(prepared, pattern, semantics)
            assert engine.contains(raw) == engine.contains(amortized)
            assert engine.count(raw) == engine.count(amortized)
            raw_images = {frozenset(m.values()) for m in engine.mappings(raw)}
            prepared_images = {frozenset(m.values()) for m in engine.mappings(amortized)}
            assert raw_images == prepared_images


def test_portfolio_prepared_target_agrees_with_raw():
    host = gnp_host(30, 0.4, seed=5)
    pattern = nx.path_graph(4)
    engines = _engines()
    portfolio = Portfolio(matchers=engines)
    prepared = portfolio.prepare(host)
    for semantics in (Semantics.INDUCED, Semantics.ORDINARY):
        raw = Query(host, pattern, semantics)
        amortized = Query(prepared, pattern, semantics)
        assert portfolio.contains(raw) == portfolio.contains(amortized)
        assert portfolio.count(raw) == portfolio.count(amortized)
        first = portfolio.first_mapping(amortized)
        assert (first is None) == (portfolio.first_mapping(raw) is None)


def test_labelled_prepared_counts_agree():
    host = gnp_host(25, 0.3, seed=7, label_count=3)
    pattern = nx.path_graph(3)
    for node in pattern:
        pattern.nodes[node]["label"] = node % 3
    engines = _engines()
    portfolio = Portfolio(matchers=engines)
    prepared = portfolio.prepare(host, node_attr="label")
    for semantics in (Semantics.INDUCED, Semantics.ORDINARY):
        raw = Query(host, pattern, semantics, node_attr="label")
        amortized = Query(prepared, pattern, semantics, node_attr="label")
        assert portfolio.count(raw) == portfolio.count(amortized)


def test_portfolio_prepare_is_lazy_and_handles_edge_labels():
    host = gnp_host(20, 0.3, seed=11)
    for u, v in host.edges():
        host.edges[u, v]["bond"] = 1
    pattern = nx.path_graph(3)
    for u, v in pattern.edges():
        pattern.edges[u, v]["bond"] = 1
    portfolio = Portfolio(matchers=_engines())
    prepared = portfolio.prepare(host, edge_attr="bond")
    assert prepared.artifact == {}  # nothing converted until a query runs
    assert prepared.host_stats is not None
    query = Query(prepared, pattern, Semantics.INDUCED, edge_attr="bond")
    assert portfolio.contains(query) == portfolio.contains(Query(host, pattern, Semantics.INDUCED, edge_attr="bond"))
    assert ("vf2", None, "bond") in prepared.artifact
    assert all(key[0] != "lad" for key in prepared.artifact)

