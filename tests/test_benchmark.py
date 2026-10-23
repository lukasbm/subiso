"""Benchmark harness: sweep construction, cross-engine agreement, output files."""

from __future__ import annotations

import csv
import json

from subiso.backends.lad import LadMatcher
from subiso.backends.vf2 import Vf2Matcher
from subiso.benchmark.grid import synthetic_cells
from subiso.benchmark.harness import Cell, benchmark_cell, run_sweep
from subiso.datasets.synthetic import gnp_host, planted_pattern
from subiso.matcher import MatcherCrash, Query
from subiso.portfolio import Portfolio
from subiso.semantics import Semantics


def test_synthetic_cells_cover_axes():
    cells = synthetic_cells(
        (12,),
        (0.2,),
        (3,),
        induced_relations=(True, False),
        label_counts=(0, 2),
        pattern_sources=("planted", "random"),
        seed=0,
    )
    assert len(cells) == 8
    names = [cell.name for cell in cells]
    assert any("induced" in name for name in names)
    assert any("ordinary" in name for name in names)
    assert any("_l2" in name for name in names)


def test_run_sweep_writes_outputs_and_agrees(tmp_path):
    host = gnp_host(15, 0.3, seed=0)
    pattern = planted_pattern(host, 3, seed=1)
    cell = Cell("toy", Query(host, pattern, Semantics.INDUCED))
    engines = [Vf2Matcher(), LadMatcher(), Portfolio(matchers=[Vf2Matcher(), LadMatcher()])]
    output = run_sweep(
        [cell],
        engines,
        operations=("contains", "count", "first_mapping"),
        repeats=2,
        warmups=1,
        output_directory=tmp_path / "bench",
        parameters={"seed": 0},
    )
    assert (output / "results.csv").exists()
    assert (output / "summary.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["parameters"] == {"seed": 0}
    assert "subiso" in manifest["dependencies"]
    rows = list(csv.DictReader((output / "results.csv").open()))
    assert {"vf2", "lad", "portfolio"} <= {row["engine"] for row in rows}
    assert all(row["operation"] in {"contains", "count", "first_mapping"} for row in rows)


class _CrashingVf2(Vf2Matcher):
    name = "crasher"

    def contains(self, query, *, domains=None, time_limit=None):
        raise MatcherCrash("simulated native crash")


def test_benchmark_skips_crashing_engines():
    host = gnp_host(10, 0.4, seed=0)
    pattern = planted_pattern(host, 3, seed=1)
    cell = Cell("crashy", Query(host, pattern, Semantics.INDUCED))
    rows = benchmark_cell(cell, [Vf2Matcher(), _CrashingVf2()], operations=("contains",), repeats=1, warmups=0)
    assert {row["engine"] for row in rows} == {"vf2"}  # the crasher is skipped, not fatal


def test_run_sweep_amortized_protocol_records_prepare_rows(tmp_path):
    host = gnp_host(20, 0.4, seed=2)
    pattern = planted_pattern(host, 3, seed=3)
    cell = Cell("toy", Query(host, pattern, Semantics.INDUCED))
    engines = [Vf2Matcher(), LadMatcher()]
    output = run_sweep(
        [cell],
        engines,
        operations=("contains", "count"),
        repeats=1,
        warmups=1,
        protocol="amortized",
        output_directory=tmp_path / "bench",
    )
    rows = list(csv.DictReader((output / "results.csv").open()))
    assert {row["operation"] for row in rows} == {"prepare", "contains", "count"}
    assert {row["engine"] for row in rows} == {"vf2", "lad"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"] == "amortized"
