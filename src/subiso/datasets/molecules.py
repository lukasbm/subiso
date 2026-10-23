"""Molecular graph helpers (optional RDKit dependency).

Molecules become labelled NetworkX graphs over heavy atoms: nodes carry the
element symbol under ``element``, edges the bond order under ``order``.
Fragment patterns are connected induced subgraphs, the natural SMARTS-free
pattern distribution for chemical hosts.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem


def _require_rdkit():
    from rdkit import Chem

    return Chem


def mol_to_networkx(mol: Chem.Mol) -> nx.Graph:
    """Label heavy atoms by element and bonds by order."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            order=str(int(bond.GetBondTypeAsDouble())),
        )
    return graph


def smiles_to_networkx(smiles: str) -> nx.Graph:
    """Parse a SMILES string and convert it; hydrogen atoms stay implicit."""
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}.")
    return mol_to_networkx(mol)


def fragment_pattern(graph: nx.Graph, size: int, *, seed: int) -> nx.Graph:
    """Connected induced subgraph of exactly ``size`` vertices, grown by random BFS.

    Only components with at least ``size`` vertices are eligible: an earlier
    version silently returned a smaller pattern (including single isolated
    atoms in molecular hosts) when BFS exhausted its component, which corrupted
    the labelled cell names.
    """
    assert 1 <= size <= graph.number_of_nodes(), f"pattern size {size} exceeds host order {graph.number_of_nodes()}"
    rng = random.Random(seed)
    eligible = [component for component in nx.connected_components(graph) if len(component) >= size]
    assert eligible, f"no connected component of order {size} in a host of order {graph.number_of_nodes()}"
    start = rng.choice(sorted(rng.choice(eligible)))
    nodes = {start}
    frontier = list(graph.neighbors(start))
    while len(nodes) < size:
        choice = rng.choice(frontier)
        frontier.remove(choice)
        nodes.add(choice)
        frontier.extend(neighbor for neighbor in graph.neighbors(choice) if neighbor not in nodes)
    assert len(nodes) == size
    pattern = graph.subgraph(nodes).copy()
    assert pattern.number_of_nodes() == size
    pattern.graph["pattern_source"] = "fragment"
    return pattern
