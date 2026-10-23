"""Dataset layer: synthetic determinism, format round-trip, checksum-verified fetch."""

from __future__ import annotations

import networkx as nx
import pytest

from subiso.datasets.synthetic import gnp_host, planted_pattern, random_pattern, random_regular_host


def test_synthetic_generators_are_seeded_and_structured():
    first = gnp_host(20, 0.2, seed=1, label_count=3)
    same = gnp_host(20, 0.2, seed=1, label_count=3)
    other = gnp_host(20, 0.2, seed=2, label_count=3)
    assert nx.utils.graphs_equal(first, same)
    assert not nx.utils.graphs_equal(first, other)
    assert all("label" in data for _, data in first.nodes(data=True))

    host = random_regular_host(12, 3, seed=0)
    assert all(degree == 3 for _, degree in host.degree())
    pattern = planted_pattern(host, 4, seed=0)
    assert pattern.number_of_nodes() == 4
    for left, right in pattern.edges():
        assert host.has_edge(left, right)

    random = random_pattern(4, 0.1, seed=0, label_count=2)
    assert random.number_of_nodes() == 4


def test_molecule_conversion_and_fragment_pattern():
    pytest.importorskip("rdkit")
    from subiso.datasets.molecules import fragment_pattern, smiles_to_networkx

    benzene = smiles_to_networkx("c1ccccc1")
    assert benzene.number_of_nodes() == 6
    assert all(benzene.nodes[node]["element"] == "C" for node in benzene)
    fragment = fragment_pattern(benzene, 4, seed=0)
    assert fragment.number_of_nodes() == 4
    assert nx.is_connected(fragment)


def test_tudataset_parser(tmp_path):
    from subiso.datasets.tudataset import _parse

    directory = tmp_path / "G"
    directory.mkdir()
    (directory / "G_A.txt").write_text("1, 2\n2, 3\n4, 5\n")
    (directory / "G_graph_indicator.txt").write_text("1\n1\n1\n2\n2\n")
    (directory / "G_node_labels.txt").write_text("5\n6\n5\n0\n1\n")
    (directory / "G_graph_labels.txt").write_text("0\n1\n")
    graphs = _parse(directory, "G")
    assert len(graphs) == 2
    assert sorted(dict(graphs[0].nodes(data="label")).values()) == [5, 5, 6]
    assert graphs[1].number_of_nodes() == 2 and graphs[1].number_of_edges() == 1


def test_ogb_mol_zip_parser(tmp_path):
    import gzip
    import zipfile

    from subiso.datasets.ogb_mol import load_smiles

    archive = tmp_path / "ogbg-molhiv.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        payload = gzip.compress(b"smiles,mol_id\nCCO,1\nc1ccccc1,2\n")
        bundle.writestr("ogbg-molhiv/mapping/mol.csv.gz", payload)
    assert load_smiles(archive) == ["CCO", "c1ccccc1"]
    assert load_smiles(archive, limit=1) == ["CCO"]


def test_tudataset_limit_is_deterministic():
    import networkx as nx

    from subiso.datasets.tudataset import sample

    graphs = [nx.path_graph(index + 2) for index in range(10)]
    first = sample(graphs, 4, seed=0)
    second = sample(graphs, 4, seed=0)
    assert len(first) == 4 and first == second
    assert sample(graphs, 100) is graphs
