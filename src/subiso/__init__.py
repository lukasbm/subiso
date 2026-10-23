"""subiso: one API for many subgraph isomorphism engines.

The router lives in :mod:`subiso.portfolio` and is itself a matcher, so callers
either use the :class:`~subiso.portfolio.Portfolio` or the function-level API in
:mod:`subiso.matching`.
"""

from .capabilities import Capabilities
from .dispatch import is_fast_path_enabled, set_fast_path
from .matcher import BaseMatcher, Matcher, Query, UnsupportedOperation
from .matching import (
    are_isomorphic,
    automorphism_orbits,
    contains_subgraph,
    count_automorphisms,
    count_embeddings,
    count_induced_occurrences_capped,
    count_subgraph_positions,
    find_induced_occurrences,
    find_subgraph_occurrences,
    get_first_subgraph_mapping,
    get_subgraph_mappings,
)
from .portfolio import Portfolio
from .semantics import Semantics

__all__ = [
    "BaseMatcher",
    "Capabilities",
    "Matcher",
    "Portfolio",
    "Query",
    "Semantics",
    "UnsupportedOperation",
    "are_isomorphic",
    "automorphism_orbits",
    "contains_subgraph",
    "count_automorphisms",
    "count_embeddings",
    "count_induced_occurrences_capped",
    "count_subgraph_positions",
    "find_induced_occurrences",
    "find_subgraph_occurrences",
    "get_first_subgraph_mapping",
    "get_subgraph_mappings",
    "is_fast_path_enabled",
    "set_fast_path",
]
