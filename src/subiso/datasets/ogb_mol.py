"""OGB molecule collections from their CSV distribution.

OGB's ``csv_mol_download`` zips contain ``<name>/mapping/mol.csv.gz`` with one
SMILES per row; this module extracts those strings so they can be parsed into
graphs with :mod:`subiso.datasets.molecules`.  Download the zip from the OGB
site or the ogb package's data server and pass its path; nothing is bundled
and no torch dependency is needed.  Licence: OGB datasets carry their own
terms (MIT for the code, dataset-specific for the data).
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from pathlib import Path


def load_smiles(zip_path: str | Path, *, limit: int | None = None) -> list[str]:
    """SMILES strings from an OGB csv_mol zip (first column named ``smiles``)."""
    with zipfile.ZipFile(zip_path) as bundle:
        members = [name for name in bundle.namelist() if name.endswith("mapping/mol.csv.gz")]
        if not members:
            raise ValueError(f"{zip_path}: no mapping/mol.csv.gz inside; is this an OGB csv_mol zip?")
        with bundle.open(members[0]) as raw:
            text = gzip.decompress(raw.read()).decode("utf-8")
    smiles = [row["smiles"] for row in csv.DictReader(io.StringIO(text))]
    return smiles[:limit] if limit is not None else smiles
