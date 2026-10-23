"""RDKit substructure backend for SMARTS patterns over molecules.

This is a relation of its own rather than an isomorphism over NetworkX graphs:
targets and patterns are :class:`rdkit.Chem.Mol` objects.
"""

from __future__ import annotations

from typing import cast

from rdkit import Chem

from ..capabilities import Capabilities
from ..matcher import BaseMatcher, PreparedTarget, Query, UnsupportedOperation, unpack_query


def contains(target: Chem.Mol, query: Chem.Mol) -> bool:
    return bool(target.HasSubstructMatch(query, useChirality=False))


def mappings(target: Chem.Mol, query: Chem.Mol, *, cap: int | None = None) -> list[dict[int, int]]:
    """Query-atom-index -> target-atom-index mappings, optionally capped.

    RDKit aligns each match with the query's atom order, so a mapping is read
    directly off the match tuple.
    """
    max_matches = 0 if cap is None else cap
    return [
        {query_index: int(target_index) for query_index, target_index in enumerate(match)}
        for match in target.GetSubstructMatches(query, uniquify=True, useChirality=False, maxMatches=max_matches)
    ]


def occurrences(target: Chem.Mol, query: Chem.Mol, *, cap: int | None = None) -> list[frozenset[int]]:
    return [frozenset(mapping.values()) for mapping in mappings(target, query, cap=cap)]


def count(target: Chem.Mol, query: Chem.Mol, *, cap: int | None = None) -> int:
    return len(mappings(target, query, cap=cap))


class SmartsMatcher(BaseMatcher):
    """The :class:`~subiso.matcher.Matcher` view of RDKit SMARTS matching."""

    name = "smarts"
    capabilities = Capabilities(
        ordinary=True,
        contains=True,
        first_mapping=True,
        mappings=True,
        occurrences=True,
        count=True,
        count_capped=True,
    )

    def available(self) -> bool:
        return True

    def prepare(self, target, *, node_attr=None, edge_attr=None) -> PreparedTarget:
        mol = self._ensure_mol(target)
        return PreparedTarget(self.name, mol, target)

    def _ensure_mol(self, obj: object) -> Chem.Mol:
        if isinstance(obj, Chem.Mol):
            return obj
        if isinstance(obj, str):
            mol = Chem.MolFromSmarts(obj) or Chem.MolFromSmiles(obj)
            if mol is not None:
                return mol
        raise UnsupportedOperation("SmartsMatcher expects an RDKit Chem.Mol object or SMILES/SMARTS string.")

    def contains(self, query, *, domains=None, time_limit=None) -> bool:
        query = unpack_query(self.name, query)
        target_mol = self._ensure_mol(query.target)
        pattern_mol = self._ensure_mol(query.pattern)
        return contains(target_mol, pattern_mol)

    def first_mapping(self, query, *, domains=None, time_limit=None):
        query = unpack_query(self.name, query)
        target_mol = self._ensure_mol(query.target)
        pattern_mol = self._ensure_mol(query.pattern)
        maps = mappings(target_mol, pattern_mol, cap=1)
        return maps[0] if maps else None

    def mappings(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        target_mol = self._ensure_mol(query.target)
        pattern_mol = self._ensure_mol(query.pattern)
        return mappings(target_mol, pattern_mol, cap=cap)

    def count(self, query, *, cap=None, domains=None) -> int:
        query = unpack_query(self.name, query)
        target_mol = self._ensure_mol(query.target)
        pattern_mol = self._ensure_mol(query.pattern)
        return count(target_mol, pattern_mol, cap=cap)

    def occurrences(self, query, *, cap=None, domains=None):
        query = unpack_query(self.name, query)
        target_mol = self._ensure_mol(query.target)
        pattern_mol = self._ensure_mol(query.pattern)
        return occurrences(target_mol, pattern_mol, cap=cap)
