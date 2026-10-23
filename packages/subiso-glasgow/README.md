# subiso-glasgow

Glasgow subgraph solver backend for [subiso](../..), built in-process with
pybind11.  The solver source is the pinned `vendor/glasgow` git submodule
(MIT, `ciaranm/glasgow-subgraph-solver`); GMP is the only system dependency.

Build prerequisites: CMake ≥ 3.22, a C++20 compiler, GMP (`~/.local`, then
`/usr/local`, then `/usr` are searched).

```bash
uv pip install packages/subiso-glasgow
```

```python
import networkx as nx
from subiso.matcher import Query, Semantics
from subiso_glasgow import GlasgowMatcher

matcher = GlasgowMatcher()
host = nx.cycle_graph(8)
pattern = nx.path_graph(3)
assert matcher.contains(Query(host, pattern, Semantics.INDUCED))
assert matcher.count(Query(host, pattern, Semantics.INDUCED)) == 8
```

v0 scope: simple undirected graphs; vertex labels as equality classes (no
wildcards); induced and ordinary matching; existence, first mapping,
enumeration, counts, and a one-second-resolution timeout on the search calls.
