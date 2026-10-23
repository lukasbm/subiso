"""RDKit SMARTS backend: query-atom-index mappings and caps."""

from __future__ import annotations

import pytest

Chem = pytest.importorskip("rdkit.Chem")

from subiso.backends import smarts  # noqa: E402


def test_smarts_mappings_expose_query_atom_indices():
    target = Chem.MolFromSmiles("CCO")
    query = Chem.MolFromSmarts("CO")
    assert target is not None and query is not None
    mappings = smarts.mappings(target, query)
    assert mappings
    assert all(set(mapping) == set(range(query.GetNumAtoms())) for mapping in mappings)
    assert {frozenset(mapping.values()) for mapping in mappings} == set(smarts.occurrences(target, query))
    assert len(smarts.mappings(target, query, cap=1)) == 1


def test_smarts_contains_and_count():
    target = Chem.MolFromSmiles("c1ccccc1O")  # phenol
    assert smarts.contains(target, Chem.MolFromSmarts("c1ccccc1"))
    assert not smarts.contains(target, Chem.MolFromSmarts("n1ccccc1"))
    assert smarts.count(target, Chem.MolFromSmarts("[OH]")) == 1


def test_smarts_matcher_subclass():
    from subiso.matcher import Query
    from subiso.semantics import Semantics

    matcher = smarts.SmartsMatcher()
    assert matcher.available()
    target = Chem.MolFromSmiles("c1ccccc1O")
    pattern = Chem.MolFromSmarts("c1ccccc1")
    q = Query(target, pattern, Semantics.ORDINARY)

    assert matcher.contains(q)
    assert matcher.count(q) == 1
    first = matcher.first_mapping(q)
    assert first is not None
    assert len(first) == 6
    assert matcher.occurrences(q) == [frozenset(range(6))]
