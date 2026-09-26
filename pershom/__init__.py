"""pershom: persistence homology pairing with GF(2) chain certificates.

Public interface (fixed contract):

    compute_persistence(simplices) -> Persistence

Input: an iterable of pairs ``(vertices, filtration)`` where ``vertices`` is a
sequence of 1..3 distinct, strictly increasing, non-negative ``int`` vertex
ids and ``filtration`` is a finite ``int`` or ``float`` (``bool`` and
NaN/+-infinity are rejected). At most 300 simplices are accepted.

Output: a :class:`Persistence` with the filtration-ordered cells, H0/H1 bars
and GF(2) representative chains for every finite H1 bar.
"""

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
from .cohomology import (
    CohomologySlice,
    CupProductResult,
    cohomology_slice,
    cup_product,
)

__all__ = [
    "Bar",
    "Cell",
    "CohomologySlice",
    "CupProductResult",
    "INF",
    "Persistence",
    "CertificateError",
    "PersistenceError",
    "PersistenceInputError",
    "check_certificate",
    "cohomology_slice",
    "compute_persistence",
    "cup_product",
]

__version__ = "1.1.0"
