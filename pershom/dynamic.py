"""Dynamic GF(2) complex: insertion/deletion events with per-step Betti numbers.

This module adds *update* capability on top of the static filtration
machinery in :mod:`pershom.persistence`.  A :class:`DynamicComplex` holds a
plain (unfiltered) simplicial complex of dimension at most 2 and accepts
``add`` / ``remove`` events one at a time or as a transactional batch.
After every applied event the H0/H1 Betti numbers are recomputed from
scratch with the existing boundary-matrix reduction (full recompute is
deliberate; at the 300-simplex cap it costs milliseconds).

Atomicity contract:

* a single ``add``/``remove`` validates everything *before* mutating, so a
  rejected event never changes the complex;
* removing a simplex that still has a coface (a vertex on a present
  edge/triangle, an edge of a present triangle) is rejected atomically with
  reason ``coface_present``;
* :meth:`DynamicComplex.apply` is transactional: the whole batch is
  simulated on a private copy and committed only if every event is legal,
  so a failing batch leaves the complex exactly as it was.

Everything is deterministic and uses the Python standard library only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .persistence import (
    MAX_SIMPLICES,
    Cell,
    PersistenceInputError,
    _build_boundary,
    _faces,
    _reduce,
    _validate_vertices,
)

__all__ = ["ADD", "REMOVE", "DynamicComplex", "DynamicStep"]

ADD = "add"
REMOVE = "remove"
_OPS = (ADD, REMOVE)


@dataclass(frozen=True)
class DynamicStep:
    """Outcome of one applied event.

    Attributes:
        event: ``"add"`` or ``"remove"``.
        vertices: the canonical (strictly increasing) vertex tuple.
        dimension: 0, 1 or 2.
        betti: ``(beta0, beta1)`` of the complex *after* this event.
    """

    event: str
    vertices: tuple[int, ...]
    dimension: int
    betti: tuple[int, int]

    def __repr__(self) -> str:
        return (
            f"DynamicStep({self.event} d{self.dimension} {self.vertices!r} "
            f"-> betti={self.betti})"
        )


class DynamicComplex:
    """A mutable simplicial complex (dimensions 0..2) over GF(2).

    The complex starts empty.  ``add`` requires every codimension-1 face to
    be present (which inductively keeps the complex closed); ``remove``
    requires the simplex to be present and to have no coface left.  At most
    ``MAX_SIMPLICES`` (300) simplices are held, the same cap as the static
    interface.
    """

    def __init__(self):
        # Stored as (dimension, vertices) keys; vertices are canonical
        # strictly-increasing tuples produced by _validate_vertices.
        self._cells: set[tuple[int, tuple[int, ...]]] = set()

    # -- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, vertices) -> bool:
        try:
            verts = tuple(vertices)
        except TypeError:
            return False
        return (len(verts) - 1, verts) in self._cells

    def simplices(self) -> tuple[tuple[int, ...], ...]:
        """All simplices as vertex tuples, ordered by (dimension, vertices)."""
        return tuple(v for _d, v in sorted(self._cells))

    def betti(self) -> tuple[int, int]:
        """``(beta0, beta1)`` of the current complex, recomputed in full.

        Cells are ordered by ``(dimension, vertices)`` with a dummy
        filtration of 0; the existing GF(2) boundary reduction then gives
        ``rank(d1)`` (nonzero reduced edge columns) and ``rank(d2)``
        (nonzero reduced triangle columns), and
        ``beta0 = n0 - rank(d1)``, ``beta1 = n1 - rank(d1) - rank(d2)``.
        """
        cells = [
            Cell(index=i, dimension=d, vertices=v, filtration=0)
            for i, (d, v) in enumerate(sorted(self._cells))
        ]
        boundary = _build_boundary(cells)
        reduced, _combo, _owner = _reduce(boundary)
        count = [0, 0, 0]
        rank = [0, 0, 0]
        for cell, col in zip(cells, reduced):
            count[cell.dimension] += 1
            if col:
                rank[cell.dimension] += 1
        beta0 = count[0] - rank[1]
        beta1 = count[1] - rank[1] - rank[2]
        return beta0, beta1

    # -- event validation (pure: never touches self._cells) ----------------

    @staticmethod
    def _validate_new(vertices, cells, index=None):
        """Return the key to insert, or raise before any mutation."""
        verts = _validate_vertices(vertices, index, vertices)
        dim = len(verts) - 1
        key = (dim, verts)
        if key in cells:
            raise PersistenceInputError(
                "duplicate_simplex",
                f"simplex d={dim} {verts!r} is already present",
                index=index,
                simplex=vertices,
            )
        if len(cells) >= MAX_SIMPLICES:
            raise PersistenceInputError(
                "too_many_simplices",
                f"the complex already holds {MAX_SIMPLICES} simplices; "
                f"adding {verts!r} would exceed the bound",
                index=index,
                simplex=vertices,
            )
        if dim >= 1:
            for face in _faces(verts):
                if (dim - 1, face) not in cells:
                    raise PersistenceInputError(
                        "missing_face",
                        f"simplex d={dim} {verts!r} requires face {face!r} "
                        f"(d={dim - 1}), which is absent",
                        index=index,
                        simplex=vertices,
                    )
        return key

    @staticmethod
    def _validate_removable(vertices, cells, index=None):
        """Return the key to delete, or raise before any mutation."""
        verts = _validate_vertices(vertices, index, vertices)
        dim = len(verts) - 1
        key = (dim, verts)
        if key not in cells:
            raise PersistenceInputError(
                "not_present",
                f"simplex d={dim} {verts!r} is not in the complex",
                index=index,
                simplex=vertices,
            )
        target = set(verts)
        for other_dim, other_verts in cells:
            if other_dim > dim and target < set(other_verts):
                raise PersistenceInputError(
                    "coface_present",
                    f"cannot remove d={dim} {verts!r}: coface d={other_dim} "
                    f"{other_verts!r} is still present; remove it first",
                    index=index,
                    simplex=vertices,
                )
        return key

    # -- single events -----------------------------------------------------

    def add(self, vertices) -> DynamicStep:
        """Add one simplex; faces must already be present.

        Returns the :class:`DynamicStep` with the Betti numbers after the
        insertion.  Raises :class:`PersistenceInputError` (reasons
        ``bad_vertices``/``bad_vertex_id``/``duplicate_vertex_in_simplex``/
        ``unsorted_vertices``/``bad_dimension``/``duplicate_simplex``/
        ``too_many_simplices``/``missing_face``) without changing anything.
        """
        key = self._validate_new(vertices, self._cells)
        self._cells.add(key)
        return DynamicStep(ADD, key[1], key[0], self.betti())

    def remove(self, vertices) -> DynamicStep:
        """Remove one simplex; it must be present and have no coface left.

        Returns the :class:`DynamicStep` with the Betti numbers after the
        removal.  A simplex with a surviving coface is rejected atomically
        with reason ``coface_present``; an absent simplex with
        ``not_present``; in both cases the complex is unchanged.
        """
        key = self._validate_removable(vertices, self._cells)
        self._cells.discard(key)
        return DynamicStep(REMOVE, key[1], key[0], self.betti())

    # -- transactional batches --------------------------------------------

    @staticmethod
    def _parse_event(raw, index):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise PersistenceInputError(
                "bad_event",
                "each event must be an (op, vertices) pair with op in "
                "('add', 'remove')",
                index=index,
                simplex=raw,
            )
        op, vertices = raw
        if op not in _OPS:
            raise PersistenceInputError(
                "bad_event",
                f"event op must be 'add' or 'remove', got {op!r}",
                index=index,
                simplex=raw,
            )
        return op, vertices

    def apply(self, events) -> list[DynamicStep]:
        """Apply a batch of ``("add"|"remove", vertices)`` events atomically.

        The batch is first validated end-to-end on a private copy of the
        complex; only if *every* event is legal is the real complex
        updated.  A failing batch raises :class:`PersistenceInputError`
        (``index`` locates the offending event, 0-based) and leaves the
        complex exactly as it was — there is no partial application.

        Returns one :class:`DynamicStep` per event, each carrying the Betti
        numbers after that event.  An empty batch is a no-op returning
        ``[]``.
        """
        if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
            raise PersistenceInputError(
                "bad_input",
                "expected an iterable of (op, vertices) events",
            )
        parsed = []
        for index, raw in enumerate(events):
            parsed.append(self._parse_event(raw, index))

        # Phase 1: simulate on a scratch copy; nothing here can mutate self.
        work = set(self._cells)
        for index, (op, vertices) in enumerate(parsed):
            if op == ADD:
                key = self._validate_new(vertices, work, index=index)
                work.add(key)
            else:
                key = self._validate_removable(vertices, work, index=index)
                work.discard(key)

        # Phase 2: commit.  Validation already succeeded on the same
        # deterministic sequence, so these calls cannot raise.
        steps = []
        for op, vertices in parsed:
            steps.append(self.add(vertices) if op == ADD else self.remove(vertices))
        return steps

    def __repr__(self) -> str:
        per_dim = [0, 0, 0]
        for d, _v in self._cells:
            per_dim[d] += 1
        return (
            f"DynamicComplex(vertices={per_dim[0]}, edges={per_dim[1]}, "
            f"triangles={per_dim[2]}, betti={self.betti()})"
        )
