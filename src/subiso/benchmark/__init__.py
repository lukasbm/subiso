"""Benchmark layer: engine-agnostic timing with cross-engine exactness checks."""

from .grid import synthetic_cells
from .harness import Cell, benchmark_cell, run_sweep, time_call

__all__ = ["Cell", "benchmark_cell", "run_sweep", "synthetic_cells", "time_call"]
