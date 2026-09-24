"""pershom: persistence homology pairing with GF(2) chain certificates,
plus a dynamic (add/remove) GF(2) complex with per-step Betti numbers.

Public interface (fixed contract):

    compute_persistence(simplices) -> Persistence
    DynamicComplex()  -- .add/.remove/.apply(events), .betti(), .simplices()

Static input: an iterable of pairs ``(vertices, filtration)`` where
``vertices`` is a sequence of 1..3 distinct, strictly increasing,
non-negative ``int`` vertex ids and ``filtration`` is a finite ``int`` or
``float`` (``bool`` and NaN/+-infinity are rejected). At most 300 simplices
are accepted.

Static output: a :class:`Persistence` with the filtration-ordered cells,
H0/H1 bars and GF(2) representative chains for every finite H1 bar.

Dynamic updates: :class:`DynamicComplex` holds an unfiltered complex of
dimension <= 2 (same 300-simplex cap). ``add`` requires faces to be
present; ``remove`` rejects a simplex that still has a coface
(``coface_present``) atomically; ``apply`` commits a whole event batch or
nothing. Every applied event returns a :class:`DynamicStep` carrying the
recomputed ``(beta0, beta1)``.
"""

from .dynamic import ADD, REMOVE, DynamicComplex, DynamicStep
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
    "ADD",
    "REMOVE",
    "Bar",
    "Cell",
    "DynamicComplex",
    "DynamicStep",
    "INF",
    "Persistence",
    "CertificateError",
    "PersistenceError",
    "PersistenceInputError",
    "check_certificate",
    "compute_persistence",
]

__version__ = "1.1.0"
