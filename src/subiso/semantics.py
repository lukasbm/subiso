"""Containment semantics a matcher can be asked for."""

from __future__ import annotations

from enum import Enum


class Semantics(Enum):
    """How a pattern must sit inside a target graph.

    ``INDUCED``
        Edge-preserving and edge-reflecting: ``(u, v) in pattern`` iff
        ``(f(u), f(v)) in target``.
    ``ORDINARY``
        Edge-preserving only: an ordinary subgraph isomorphism (monomorphism).

    Non-injective matching (homomorphism) is not a containment relation any
    wrapped engine implements, so it is deliberately not part of this enum.
    """

    INDUCED = "induced"
    ORDINARY = "ordinary"
