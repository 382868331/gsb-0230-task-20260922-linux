"""Dynamic insert/delete GF(2) complex with per-step H0/H1 Betti numbers.

This module adds the incremental-update capability on top of the static
structures in :mod:`pershom.persistence`.  A :class:`DynamicComplex` keeps a
mutable, always face-closed complex (dimensions 0..2) whose simplices carry
filtration values; after every accepted update the H0/H1 Betti numbers are
recomputed **from scratch** using the existing boundary-matrix construction
(:func:`pershom.persistence._build_boundary`).  Full recompute is a deliberate
design choice: no zigzag interval decomposition and no chain certificates are
maintained across updates.

Update rules (all atomic -- validation completes before any mutation):

* ``add``: vertices/filtration validated exactly as for
  :func:`compute_persistence`; the simplex must not already be present
  (``duplicate_simplex``); every codimension-1 face must already be present
  (``missing_face``) with filtration ``<=`` the new simplex
  (``face_filtration_order``); the 300-simplex bound still applies
  (``too_many_simplices``).
* ``remove``: the simplex must be present (``missing_simplex``) and must have
  no coface still in the complex (``has_cofaces``) -- removing a simplex
  whose cofaces survive would break face closure, so it is rejected as a
  whole and the complex is left untouched.
"""

from __future__ import annotations

from .persistence import (
    MAX_SIMPLICES,
    Cell,
    PersistenceInputError,
    _build_boundary,
    _faces,
    _parse,
    _validate_filtration,
    _validate_vertices,
    compute_persistence,
)

__all__ = ["DynamicComplex"]


def _column_rank(columns) -> int:
    """Rank over GF(2) of integer-bitset columns (plain Gaussian elimination)."""
    pivots: dict[int, int] = {}
    for col in columns:
        while col:
            p = col.bit_length() - 1
            owner = pivots.get(p)
            if owner is None:
                pivots[p] = col
                break
            col ^= owner
    return len(pivots)


class DynamicComplex:
    """A mutable face-closed GF(2) complex (dimensions 0..2).

    ``DynamicComplex()`` starts empty; ``DynamicComplex(simplices)`` validates
    the iterable with exactly the static ``compute_persistence`` contract
    (atomically: a rejected constructor argument leaves no object behind).

    The complex is stored as a mapping ``(dimension, vertices) -> filtration``
    and is only ever mutated after full validation, so a rejected ``add`` /
    ``remove`` never leaves a half-updated state.
    """

    def __init__(self, simplices=None):
        self._filtration: dict[tuple[int, tuple[int, ...]], float | int] = {}
        if simplices is None:
            return
        # _parse performs the full static validation up front; the mapping
        # is only populated after it returns, so construction is atomic.
        for rec in _parse(simplices):
            self._filtration[(rec.dimension, rec.vertices)] = rec.filtration

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._filtration)

    def __contains__(self, vertices) -> bool:
        try:
            verts = tuple(vertices)
        except TypeError:
            return False
        return (len(verts) - 1, verts) in self._filtration

    def simplices(self) -> list[tuple[tuple[int, ...], float | int]]:
        """Current complex as ``(vertices, filtration)`` pairs, sorted by the
        usual total order ``(filtration, dimension, vertices)``."""
        return [
            (verts, f)
            for (dim, verts), f in sorted(
                self._filtration.items(),
                key=lambda item: (item[1], item[0][0], item[0][1]),
            )
        ]

    # -- updates (atomic) --------------------------------------------------

    def add(self, vertices, filtration) -> None:
        """Insert one simplex.  Raises :class:`PersistenceInputError` and
        changes nothing when the insertion would break the contract."""
        raw = (vertices, filtration)
        verts = _validate_vertices(vertices, None, raw)
        fval = _validate_filtration(filtration, None, raw)
        dim = len(verts) - 1
        key = (dim, verts)
        if key in self._filtration:
            raise PersistenceInputError(
                "duplicate_simplex",
                f"simplex d={dim} {verts!r} is already present",
                simplex=raw,
            )
        if len(self._filtration) >= MAX_SIMPLICES:
            raise PersistenceInputError(
                "too_many_simplices",
                f"at most {MAX_SIMPLICES} simplices are accepted; the complex "
                f"is full",
                simplex=raw,
            )
        if dim >= 1:
            for face in _faces(verts):
                face_f = self._filtration.get((dim - 1, face))
                if face_f is None:
                    raise PersistenceInputError(
                        "missing_face",
                        f"simplex d={dim} {verts!r} requires face {face!r} "
                        f"(d={dim - 1}), which is absent",
                        simplex=raw,
                    )
                if face_f > fval:
                    raise PersistenceInputError(
                        "face_filtration_order",
                        f"face {face!r} has filtration {face_f!r} > simplex "
                        f"filtration {fval!r}; faces must be <= the simplex",
                        simplex=raw,
                    )
        self._filtration[key] = fval

    def remove(self, vertices) -> None:
        """Delete one simplex.  Atomically rejected (``has_cofaces``) while
        any coface is still present; absent simplices raise
        ``missing_simplex``.  On rejection the complex is unchanged."""
        raw = (vertices, None)
        verts = _validate_vertices(vertices, None, raw)
        dim = len(verts) - 1
        key = (dim, verts)
        if key not in self._filtration:
            raise PersistenceInputError(
                "missing_simplex",
                f"simplex d={dim} {verts!r} is not in the complex",
                simplex=raw,
            )
        cofaces = self._cofaces(verts, dim)
        if cofaces:
            raise PersistenceInputError(
                "has_cofaces",
                f"simplex d={dim} {verts!r} still has coface(s) "
                f"{cofaces!r} in the complex; remove them first",
                simplex=raw,
            )
        del self._filtration[key]

    def _cofaces(self, verts: tuple[int, ...], dim: int) -> list[tuple[int, ...]]:
        """Present (dim+1)-simplices containing ``verts`` (direct cofaces).

        Direct cofaces suffice: face closure is an invariant of the complex,
        so a simplex with no direct coface has no coface of any dimension.
        """
        if dim >= 2:
            return []
        target = set(verts)
        return sorted(
            v
            for (d, v) in self._filtration
            if d == dim + 1 and target.issubset(v)
        )

    # -- measurements ------------------------------------------------------

    def betti(self) -> tuple[int, int]:
        """``(beta0, beta1)`` of the current complex, recomputed in full.

        Uses the existing boundary-matrix construction; ranks of the
        boundary maps give ``beta0 = n0 - rank(d1)`` and
        ``beta1 = n1 - rank(d1) - rank(d2)`` over GF(2).  The empty complex
        yields ``(0, 0)``.
        """
        cells = [
            Cell(index=i, dimension=len(verts) - 1, vertices=verts,
                 filtration=f)
            for i, (verts, f) in enumerate(self.simplices())
        ]
        columns = _build_boundary(cells)
        rank_d1 = _column_rank(
            [columns[c.index] for c in cells if c.dimension == 1]
        )
        rank_d2 = _column_rank(
            [columns[c.index] for c in cells if c.dimension == 2]
        )
        n0 = sum(1 for c in cells if c.dimension == 0)
        n1 = sum(1 for c in cells if c.dimension == 1)
        return n0 - rank_d1, n1 - rank_d1 - rank_d2

    def persistence(self):
        """Full static persistence of the current state.

        Convenience bridge to the original entry point: equivalent to
        ``compute_persistence(dyn.simplices())``.  Raises
        ``PersistenceInputError`` (``empty_input``) on the empty complex,
        matching the static contract.
        """
        return compute_persistence(self.simplices())

    def __repr__(self) -> str:
        dims = [0, 0, 0]
        for dim, _verts in self._filtration:
            dims[dim] += 1
        return (
            f"DynamicComplex(vertices={dims[0]}, edges={dims[1]}, "
            f"triangles={dims[2]})"
        )
