import os
import networkx as nx
import pytest

from subiso import Query, Semantics, Portfolio, is_fast_path_enabled, set_fast_path
from subiso.backends.vf2 import Vf2Matcher
from subiso.backends.vf3 import Vf3lMatcher
from subiso.dispatch import DispatchPolicy


def test_fast_path_toggle():
    # Default is off
    assert not is_fast_path_enabled()

    set_fast_path(True)
    assert is_fast_path_enabled()

    set_fast_path(False)
    assert not is_fast_path_enabled()

    set_fast_path(None)


def test_fast_path_routing_behavior():
    host = nx.cycle_graph(20)
    pat_k4 = nx.path_graph(4)
    pat_k6 = nx.path_graph(6)

    q4 = Query(host, pat_k4, Semantics.INDUCED)
    q6 = Query(host, pat_k6, Semantics.INDUCED)

    policy_normal = DispatchPolicy([Vf2Matcher(), Vf3lMatcher()], fast_path=False)
    policy_fast = DispatchPolicy([Vf2Matcher(), Vf3lMatcher()], fast_path=True)

    candidates_4 = policy_fast.candidates(q4)
    assert candidates_4[0].name == "vf2"

    candidates_6 = policy_fast.candidates(q6)
    assert candidates_6[0].name == "vf3l"


def test_fast_path_portfolio_execution():
    set_fast_path(True)
    try:
        host = nx.cycle_graph(10)
        pattern = nx.path_graph(3)
        portfolio = Portfolio()
        assert portfolio.contains(Query(host, pattern, Semantics.INDUCED))
        assert portfolio.count(Query(host, pattern, Semantics.INDUCED)) == 10
    finally:
        set_fast_path(False)
