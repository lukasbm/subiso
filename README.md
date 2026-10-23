# subiso

[![PyPI version](https://img.shields.io/pypi/v/subiso.svg)](https://pypi.org/project/subiso/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Python toolkit and learned dispatch framework for 15+ subgraph isomorphism engines.

Every backend implements the same `Matcher` interface, declares its `Capabilities` explicitly, and the `Portfolio` router is itself a `Matcher`, so user code never branches on solver backends.

---

## Installation

```bash
# Core package (pure Python: NetworkX VF2, VF2++)
pip install subiso

# Optional backend extras
pip install 'subiso[lad]'       # + igraph LAD (C, GPL-2.0+)
pip install 'subiso[engines]'   # + VF3/VF3L (C++) and FastISO (C++)
pip install 'subiso[chem]'      # + RDKit SMARTS matching
```

To install the in-process Glasgow Subgraph Solver wrapper:
```bash
pip install ./packages/subiso-glasgow
```

---

## Quickstart

### Function-Level Convenience API
```python
import networkx as nx
from subiso import contains_subgraph, count_subgraph_positions, count_embeddings

host = nx.cycle_graph(6)
pattern = nx.path_graph(3)

# One-liners routed through the portfolio automatically:
assert contains_subgraph(host, pattern, induced=True)
assert count_subgraph_positions(host, pattern, induced=True) == 6
assert count_embeddings(host, pattern, induced=True) == 12
```

### Portfolio Object & Query API
```python
from subiso import Query, Semantics, Portfolio

portfolio = Portfolio()
query = Query(host, pattern, Semantics.INDUCED)

assert portfolio.contains(query)
assert portfolio.count(query) == 6
```

---

## Amortized Queries & Target Preparation

When querying multiple patterns against the same target graph (e.g. motif mining or substructure screening), converting the host graph on every query creates significant overhead. `subiso` supports target preparation:

```python
# Precomputes O(1) host features and converts to engine-native structures lazily on first use
prepared_target = portfolio.prepare(host)

# Re-use prepared target across arbitrary pattern queries
q1 = Query(prepared_target, nx.path_graph(3), Semantics.INDUCED)
q2 = Query(prepared_target, nx.cycle_graph(4), Semantics.INDUCED)

assert portfolio.contains(q1)
assert not portfolio.contains(q2)
```

---

## Supported Engines

| Module | Engine | Type / Backend | Availability |
|---|---|---|---|
| `backends.vf2` | NetworkX VF2 | Pure Python | Always (core) |
| `backends.vf2pp` | NetworkX VF2++ | Pure Python | Always (core) |
| `backends.lad` | igraph LAD | C / igraph | Extra `[lad]` |
| `backends.vf3` | VF3 / VF3L / VF3P | C++ (`vf3py`) | Extra `[engines]` |
| `backends.fastiso` | FastISO | C++ | Extra `[engines]` |
| `backends.smarts` | RDKit SMARTS | C++ / Chem | Extra `[chem]` |
| `subiso_glasgow` | Glasgow Subgraph Solver | C++ / Bitset CP | `packages/subiso-glasgow` |
| `backends.subgraphmatching` | QuickSI, GraphQL, CFL, CECI, DP-iso, RI, LFTJ | C++ (MIT Benchmark) | `vendor/subgraphmatching` |

---

## Learned Dispatch & Fast-Path

`Portfolio` includes a lightweight, pre-trained decision-tree router that maps $O(1)$ query features (host/pattern node count, edge count, density ratio, label status) to solver rankings.

- **Tree Router (Default):** Evaluates a depth-6 decision tree ($1.07\times$ selection regret, $\sim 6\text{--}10\,\mu\text{s}$ dispatch overhead).
- **$O(1)$ Fast-Path Router:** Bypasses feature calculation and tree evaluation for microsecond-scale queries:
  - CLI: `python -m subiso check --fast-path`
  - Environment variable: `export SUBISO_FAST_PATH=1`
  - Python API: `from subiso import set_fast_path; set_fast_path(True)`

---

## Verification & Tests

Inspect available solvers and capability flags:
```bash
python -m subiso capabilities
```

Verify portfolio outputs against NetworkX brute force across the small graph atlas:
```bash
python -m subiso check
```

Run test suite:
```bash
pytest
```

---

## Reproducing Empirical Sweeps

The empirical results and methodology are documented in [`docs/writeup.tex`](docs/writeup.tex) (compiled: [`docs/writeup.pdf`](docs/writeup.pdf)).

1. **Benchmark Sweeps:**
   ```bash
   # Synthetic grid sweep
   python scripts/dataset_sweep.py --dataset=synthetic --protocol=one-shot

   # Labelled real collections (MUTAG, AIDS, ENZYMES, PROTEINS)
   python scripts/dataset_sweep.py --dataset=MUTAG --labelled=True --protocol=prepared
   ```

2. **Evaluate Learned Dispatcher:**
   ```bash
   python scripts/evaluate_policy.py
   ```

3. **Generate Figures:**
   ```bash
   python scripts/make_figures.py
   ```

---

## Citation

If you use `subiso` in your research, please cite:

```bibtex
@article{boehm2026subiso,
  author  = {B{\"o}hm, Lukas},
  title   = {Same Queries, Different Winners: Execution Protocol and Engine Choice in Subgraph Isomorphism},
  journal = {Technical Report, Leipzig University},
  year    = {2026}
}
```

---

## License

`subiso` is licensed under the [MIT License](LICENSE). Third-party optional backends retain their respective licenses (`igraph` is GPL-2.0+; RDKit and Glasgow are MIT/BSD).
