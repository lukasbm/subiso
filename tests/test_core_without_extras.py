"""The core must stay importable and functional with only NetworkX installed.

This guards the packaging split: igraph (GPL) and the compiled engines are
opt-in extras, so nothing on the import path of ``subiso`` may need them.
"""

from __future__ import annotations

import subprocess
import sys

import networkx as nx

from subiso import Portfolio, contains_subgraph, count_subgraph_positions, get_subgraph_mappings


def test_core_answers_without_optional_engines():
    host, pattern = nx.cycle_graph(6), nx.path_graph(3)
    assert contains_subgraph(host, pattern, induced=True)
    assert count_subgraph_positions(host, pattern, induced=True) == 6
    assert len(get_subgraph_mappings(host, pattern, induced=True)) == 12


def test_import_does_not_pull_optional_graph_libraries():
    script = "import sys, subiso; print(sorted(name for name in ('igraph', 'rdkit') if name in sys.modules))"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", result.stdout


def test_portfolio_degrades_to_python_engines():
    names = [matcher.name for matcher in Portfolio()._matchers]
    assert "vf2" in names
    assert all(name != "portfolio" for name in names)
