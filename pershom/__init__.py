"""pershom: persistence homology pairing with GF(2) chain certificates.

Public interface (fixed contract):

    compute_persistence(simplices) -> Persistence

Input: an iterable of pairs ``(vertices, filtration)`` where ``vertices`` is a
sequence of 1..3 distinct, strictly increasing, non-negative ``int`` vertex
ids and ``filtration`` is a finite ``int`` or ``float`` (``bool`` and
NaN/+-infinity are rejected). At most 300 simplices are accepted.

Output: a :class:`Persistence` with the filtration-ordered cells, H0/H1 bars
and GF(2) representative chains for every finite H1 bar.

Dynamic updates (this iteration):

    DynamicComplex(simplices=None)  -- mutable face-closed complex
        .add(vertices, filtration)  / .remove(vertices)   (atomic)
        .betti() -> (beta0, beta1)  (full recompute per step)
        .simplices() / .persistence() / len() / ``in``
"""

from .dynamic import DynamicComplex
from .persistence import (
    Bar,
    Cell,
    INF,
    Persistence,
    CertificateError,
    PersistenceError,
    PersistenceInputError,
    check_certificate,
    compute_persistence,
)

__all__ = [
    "Bar",
    "Cell",
    "DynamicComplex",
    "INF",
    "Persistence",
    "CertificateError",
    "PersistenceError",
    "PersistenceInputError",
    "check_certificate",
    "compute_persistence",
]

__version__ = "1.1.0"
