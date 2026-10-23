"""Dataset layer.

Licensing policy: **nothing is bundled**.  Synthetic generators need no data
at all; the TU Dortmund and OGB loaders download into a caller-provided
directory and never re-host archives.
"""

from .synthetic import gnp_host, planted_pattern, random_pattern, random_regular_host

__all__ = [
    "gnp_host",
    "planted_pattern",
    "random_pattern",
    "random_regular_host",
]
