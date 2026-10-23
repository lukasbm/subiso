"""What a matcher can do, declared explicitly instead of guessed."""

from __future__ import annotations

from dataclasses import dataclass, fields

from .semantics import Semantics

#: Field of :class:`Capabilities` that answers for each :class:`Semantics`.
_SEMANTICS_FIELD = {
    Semantics.INDUCED: "induced",
    Semantics.ORDINARY: "ordinary",
}


@dataclass(frozen=True)
class Capabilities:
    """Feature flags of one matcher.

    The router only ever asks a matcher for operations it declares here, and
    raises :class:`subiso.matcher.UnsupportedOperation` instead of silently
    falling back to different semantics.
    """

    induced: bool = False
    ordinary: bool = False

    node_labels: bool = False
    edge_labels: bool = False
    wildcards: bool = False

    contains: bool = False
    first_mapping: bool = False
    mappings: bool = False
    occurrences: bool = False
    count: bool = False
    count_capped: bool = False
    count_embeddings: bool = False

    anchors: bool = False
    time_limit: bool = False

    def supports(self, semantics: Semantics) -> bool:
        return bool(getattr(self, _SEMANTICS_FIELD[semantics]))

    def combine(self, other: Capabilities) -> Capabilities:
        """Union of two capability sets, used by the portfolio."""
        return Capabilities(
            **{field.name: getattr(self, field.name) or getattr(other, field.name) for field in fields(self)}
        )
