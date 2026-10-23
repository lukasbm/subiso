"""Type stubs for the pybind11 extension module."""

from collections.abc import Sequence

class PreparedGraph:
    @property
    def num_vertices(self) -> int: ...

def prepare_graph(
    n: int,
    edges: Sequence[tuple[int, int]],
    labels: Sequence[str | None],
) -> PreparedGraph: ...
def find_mapping_prepared(
    target: PreparedGraph,
    pattern_n: int,
    pattern_edges: Sequence[tuple[int, int]],
    pattern_labels: Sequence[str | None],
    induced: bool,
    timeout_seconds: float = ...,
) -> dict[int, int] | None: ...
def count_embeddings_prepared(
    target: PreparedGraph,
    pattern_n: int,
    pattern_edges: Sequence[tuple[int, int]],
    pattern_labels: Sequence[str | None],
    induced: bool,
    timeout_seconds: float = ...,
) -> int: ...
def count_images_prepared(
    target: PreparedGraph,
    pattern_n: int,
    pattern_edges: Sequence[tuple[int, int]],
    pattern_labels: Sequence[str | None],
    induced: bool,
    timeout_seconds: float = ...,
    limit: int = ...,
) -> int: ...
def enumerate_mappings_prepared(
    target: PreparedGraph,
    pattern_n: int,
    pattern_edges: Sequence[tuple[int, int]],
    pattern_labels: Sequence[str | None],
    induced: bool,
    timeout_seconds: float = ...,
    limit: int = ...,
) -> list[dict[int, int]]: ...
