"""Core implementation: filtration validation, boundary matrix, low-pivot
reduction over GF(2), H0/H1 pairing and chain certificates.

Everything in this module is deterministic and uses the Python standard
library only.  No arithmetic is performed on filtration *values*: they are
only compared, so results are exact (see README "误差口径").
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "INF",
    "MAX_SIMPLICES",
    "Cell",
    "Bar",
    "SliceCycle",
    "Persistence",
    "PersistenceError",
    "PersistenceInputError",
    "CertificateError",
    "compute_persistence",
    "check_certificate",
]

INF = float("inf")
MAX_SIMPLICES = 300


class PersistenceError(ValueError):
    """Base class for all errors raised by :mod:`pershom`."""


class PersistenceInputError(PersistenceError):
    """A single offending input simplex.

    Attributes:
        index: 0-based position of the simplex in the input iterable
            (``None`` when the error is about the iterable as a whole).
        simplex: the offending ``(vertices, filtration)`` object when known.
        reason: short machine-stable code, e.g. ``"missing_face"``.
    """

    def __init__(self, reason: str, detail: str, *, index=None, simplex=None):
        self.reason = reason
        self.index = index
        self.simplex = simplex
        if index is None:
            msg = f"{reason}: {detail}"
        else:
            msg = f"input[{index}] {reason}: {detail}"
        super().__init__(msg)


class CertificateError(PersistenceError):
    """A birth/death chain certificate failed independent verification."""


@dataclass(frozen=True)
class Cell:
    """One simplex in total filtration order (``index`` is its rank)."""

    index: int
    dimension: int
    vertices: tuple[int, ...]
    filtration: float | int  # passed through exactly as supplied

    @property
    def key(self) -> tuple[int, tuple[int, ...]]:
        return (self.dimension, self.vertices)

    def __repr__(self) -> str:
        return (
            f"Cell(idx={self.index}, d={self.dimension}, "
            f"v={self.vertices}, f={self.filtration!r})"
        )


@dataclass(frozen=True)
class Bar:
    """A persistence interval ``[birth, death)``.

    ``death`` is :data:`INF` for an essential class.  ``birth_chain`` /
    ``death_chain`` are sorted tuples of ``Persistence.cells`` indices; they
    are GF(2) chains (symmetric difference of the listed cells).  Chains are
    present exactly on *finite H1* bars; otherwise they are ``None``.
    """

    dimension: int
    birth_index: int
    death_index: int | None
    birth: float | int
    death: float | int
    zero_length: bool
    birth_chain: tuple[int, ...] | None = None
    death_chain: tuple[int, ...] | None = None

    @property
    def finite(self) -> bool:
        return self.death_index is not None

    def alive_at(self, threshold) -> bool:
        return self.birth <= threshold and (
            self.death_index is None or threshold < self.death
        )

    def __repr__(self) -> str:
        d = self.death if self.death_index is not None else "inf"
        tag = " [zero-length]" if self.zero_length else ""
        return f"Bar(H{self.dimension}, [{self.birth!r}, {d!r}){tag})"


class _Record(NamedTuple):
    input_index: int
    dimension: int
    vertices: tuple[int, ...]
    filtration: float | int


@dataclass(frozen=True)
class SliceCycle:
    """One H1 generator cycle of a threshold subcomplex.

    Returned by :meth:`Persistence.slice_cycles`.  ``cycle`` is a GF(2)
    1-chain (sorted ``Persistence.cells`` indices of edges) that lies
    entirely inside the subcomplex of filtration ``<= threshold`` and
    generates one H1 class alive at that moment.

    ``bar`` is the original persistence interval this generator
    corresponds to (matched by birth cell).  For a *finite* bar the
    slice representative may differ from ``bar.birth_chain`` (the birth
    cycle can be rewritten by later boundary reductions and may even use
    edges that only appear after the threshold).  The replacement is
    kept fully trackable over GF(2) by two chains:

        boundary(homology_chain)
            == cycle  XOR  bar.birth_chain  XOR  residual_chain

    ``homology_chain`` is a 2-chain of triangles and ``residual_chain``
    a 1-cycle supported on edges strictly earlier than the birth edge
    (earlier-born classes -- the change-of-basis freedom of the
    persistence reduction).  ``residual_chain`` is empty exactly when
    the slice cycle and the birth cycle are homologous in the full
    complex.  For an essential bar (``death == INF``) there is no
    ``birth_chain`` and both witness fields are ``None``.  Zero-length
    bars never appear here: their birth edge and killing triangle share
    one filtration value and enter as a single batch, so after the
    batch the class no longer occupies the dimension.
    """

    threshold: float | int
    bar: Bar
    cycle: tuple[int, ...]
    homology_chain: tuple[int, ...] | None = None
    residual_chain: tuple[int, ...] | None = None

    @property
    def essential(self) -> bool:
        """True when the matched bar is infinite (never dies)."""
        return self.bar.death_index is None

    @property
    def birth(self):
        return self.bar.birth

    @property
    def death(self):
        return self.bar.death

    def __repr__(self) -> str:
        kind = "essential" if self.essential else f"finite [{self.birth!r}, {self.death!r})"
        return f"SliceCycle(t={self.threshold!r}, {kind}, cycle={self.cycle})"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _finite_number(x) -> bool:
    """True for ints (all finite) and finite non-NaN floats."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return isinstance(x, int) or math.isfinite(x)


def _validate_filtration(value, input_index, raw):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersistenceInputError(
            "bad_filtration_type",
            f"filtration must be int or float (bool rejected), got {type(value).__name__}",
            index=input_index,
            simplex=raw,
        )
    # Every Python int is finite; only floats can carry NaN/Infinity.
    # (Converting a huge int to float would raise OverflowError.)
    if isinstance(value, float) and not math.isfinite(value):
        raise PersistenceInputError(
            "non_finite_filtration",
            f"filtration must be finite (NaN/Infinity rejected), got {value!r}",
            index=input_index,
            simplex=raw,
        )
    return value


def _validate_vertices(verts, input_index, raw) -> tuple[int, ...]:
    if not isinstance(verts, (tuple, list)):
        raise PersistenceInputError(
            "bad_vertices",
            "vertices must be a tuple/list of 1..3 ints",
            index=input_index,
            simplex=raw,
        )
    out: list[int] = []
    for v in verts:
        if isinstance(v, bool) or not isinstance(v, int):
            raise PersistenceInputError(
                "bad_vertex_id",
                f"vertex ids must be non-negative ints (bool rejected), got {v!r}",
                index=input_index,
                simplex=raw,
            )
        if v < 0:
            raise PersistenceInputError(
                "bad_vertex_id",
                f"vertex ids must be non-negative, got {v}",
                index=input_index,
                simplex=raw,
            )
        out.append(v)
    if not 1 <= len(out) <= 3:
        raise PersistenceInputError(
            "bad_dimension",
            f"only dimensions 0..2 are supported, got {len(out) - 1} "
            f"({len(out)} vertices)",
            index=input_index,
            simplex=raw,
        )
    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            raise PersistenceInputError(
                "duplicate_vertex_in_simplex",
                f"simplex has a repeated vertex: {tuple(out)!r}",
                index=input_index,
                simplex=raw,
            )
        if out[i] < out[i - 1]:
            raise PersistenceInputError(
                "unsorted_vertices",
                f"vertices must be given strictly increasing; reordering is "
                f"never done silently, got {tuple(out)!r}",
                index=input_index,
                simplex=raw,
            )
    return tuple(out)


def _faces(verts: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Codimension-1 faces in canonical (sorted) form."""
    if len(verts) == 2:
        return [(verts[0],), (verts[1],)]
    # triangle (a,b,c): edges (b,c), (a,c), (a,b)
    a, b, c = verts
    return [(b, c), (a, c), (a, b)]


def _parse(simplices) -> list[_Record]:
    if isinstance(simplices, (str, bytes)) or not isinstance(simplices, Iterable):
        raise PersistenceInputError(
            "bad_input",
            "expected an iterable of (vertices, filtration) pairs",
        )
    records: list[_Record] = []
    seen: dict[tuple[int, tuple[int, ...]], int] = {}
    for input_index, raw in enumerate(simplices):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise PersistenceInputError(
                "bad_simplex",
                "each simplex must be a (vertices, filtration) pair",
                index=input_index,
                simplex=raw,
            )
        verts_obj, filt = raw
        if len(records) >= MAX_SIMPLICES:
            raise PersistenceInputError(
                "too_many_simplices",
                f"at most {MAX_SIMPLICES} simplices are accepted; "
                f"item {input_index} exceeds the bound",
                index=input_index,
                simplex=raw,
            )
        verts = _validate_vertices(verts_obj, input_index, raw)
        fval = _validate_filtration(filt, input_index, raw)
        dim = len(verts) - 1
        key = (dim, verts)
        if key in seen:
            raise PersistenceInputError(
                "duplicate_simplex",
                f"simplex d={dim} {verts!r} already given at input[{seen[key]}]",
                index=input_index,
                simplex=raw,
            )
        seen[key] = input_index
        records.append(
            _Record(input_index, dim, verts, fval)
        )
    if not records:
        raise PersistenceInputError("empty_input", "at least one simplex is required")

    # Face closure with non-decreasing filtration.  Checked after parsing so
    # the message can name both locations.
    by_key = {(r.dimension, r.vertices): r for r in records}
    for rec in records:
        if rec.dimension == 0:
            continue
        for face_verts in _faces(rec.vertices):
            face_dim = rec.dimension - 1
            face = by_key.get((face_dim, face_verts))
            if face is None:
                raise PersistenceInputError(
                    "missing_face",
                    f"simplex d={rec.dimension} {rec.vertices!r} requires face "
                    f"{face_verts!r} (d={face_dim}), which is absent",
                    index=rec.input_index,
                )
            if face.filtration > rec.filtration:
                raise PersistenceInputError(
                    "face_filtration_order",
                    f"face {face_verts!r} has filtration {face.filtration!r} > "
                    f"simplex filtration {rec.filtration!r}; faces must be "
                    f"<= the simplex",
                    index=rec.input_index,
                )
    return records


# ---------------------------------------------------------------------------
# Boundary matrix and reduction
# ---------------------------------------------------------------------------

def _build_boundary(cells: list[Cell]) -> list[int]:
    """Boundary columns over GF(2) as Python-int bitsets over cell indices."""
    idx: dict[tuple[int, tuple[int, ...]], int] = {
        c.key: c.index for c in cells
    }
    columns = [0] * len(cells)
    for cell in cells:
        col = 0
        if cell.dimension >= 1:
            for fv in _faces(cell.vertices):
                col ^= 1 << idx[(cell.dimension - 1, fv)]
        columns[cell.index] = col
    return columns


def _reduce(columns: list[int]) -> tuple[list[int], list[int], dict[int, int]]:
    """Classical left-to-right persistence reduction.

    Pivot convention: the *lowest* non-zero entry, i.e. the bottom-most 1
    of the column, which is the largest row index (highest set bit of the
    GF(2) bitset).  Returns reduced columns ``R``, change-of-basis masks
    ``V`` (``R[j] == XOR_{k in V[j]} D[k]``) and the map
    ``pivot row -> owning column``.
    """
    n = len(columns)
    reduced = [0] * n
    combo = [0] * n
    owner: dict[int, int] = {}
    for j in range(n):
        col = columns[j]
        comb = 1 << j
        while col:
            pivot = col.bit_length() - 1
            k = owner.get(pivot)
            if k is None:
                break
            col ^= reduced[k]
            comb ^= combo[k]
        reduced[j] = col
        combo[j] = comb
        if col:
            owner[col.bit_length() - 1] = j
    return reduced, combo, owner


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------

class Persistence:
    """Result of :func:`compute_persistence`.

    Attributes:
        cells: simplices in total filtration order; the list index is the
            matrix column / row index used everywhere else.
        bars: all H0 and H1 bars, ordered by ``(dimension, birth index,
            death index)``.
        h0_bars / h1_bars: filtered views.
    """

    def __init__(self, cells: list[Cell], bars: list[Bar], *,
                 _boundary=None, _reduced=None, _combo=None, _owner=None):
        self.cells: tuple[Cell, ...] = tuple(cells)
        self.bars: tuple[Bar, ...] = tuple(bars)
        self.h0_bars: tuple[Bar, ...] = tuple(b for b in bars if b.dimension == 0)
        self.h1_bars: tuple[Bar, ...] = tuple(b for b in bars if b.dimension == 1)
        # Reduction by-products kept for slice_cycles(); recomputed lazily
        # when a Persistence was assembled directly by user code.
        self._boundary = _boundary
        self._reduced = _reduced
        self._combo = _combo
        self._owner = _owner

    def _reduction(self):
        if self._boundary is None:
            boundary = _build_boundary(list(self.cells))
            reduced, combo, owner = _reduce(boundary)
            self._boundary, self._reduced, self._combo, self._owner = (
                boundary, reduced, combo, owner)
        return self._boundary, self._reduced, self._combo, self._owner

    # -- queries -----------------------------------------------------------

    def betti_at(self, threshold) -> tuple[int, int]:
        """Betti numbers ``(beta0, beta1)`` of the threshold subcomplex.

        The subcomplex consists of every cell with ``filtration <=
        threshold``; a bar ``[b, d)`` is alive iff ``b <= t < d``.
        ``threshold`` must be a finite int/float (no NaN/Infinity).
        """
        if not _finite_number(threshold):
            raise PersistenceInputError(
                "bad_threshold",
                "threshold must be a finite int or float",
            )
        b0 = sum(1 for b in self.h0_bars if b.alive_at(threshold))
        b1 = sum(1 for b in self.h1_bars if b.alive_at(threshold))
        return b0, b1

    def betti_profile(self, thresholds=None) -> list[tuple[float | int, tuple[int, int]]]:
        """``[(threshold, (b0, b1)), ...]``.

        Defaults to one sample per distinct filtration value.  Explicit
        thresholds must be finite numbers and are de-duplicated while
        preserving order.
        """
        if thresholds is None:
            points = sorted({c.filtration for c in self.cells})
        else:
            points = []
            seen_vals = set()
            for t in thresholds:
                if not _finite_number(t):
                    raise PersistenceInputError(
                        "bad_threshold",
                        "thresholds must be finite ints or floats",
                    )
                if t not in seen_vals:
                    seen_vals.add(t)
                    points.append(t)
        return [(t, self.betti_at(t)) for t in points]

    def chain_vertices(self, chain: Iterable[int]) -> list[tuple[int, ...]]:
        """Resolve a chain of cell indices to the listed vertex tuples."""
        return [self.cells[i].vertices for i in chain]

    def slice_cycles(self, threshold) -> tuple[SliceCycle, ...]:
        """H1 generator cycles of the subcomplex with filtration <= t.

        Every cell with ``filtration <= threshold`` is included; cells
        sharing one filtration value enter as a single batch (the slice
        is a prefix of the total filtration order), so no ordering
        artifact can expose half of a moment.  Returns one
        :class:`SliceCycle` per H1 class alive at ``threshold``
        (``len(result) == self.betti_at(threshold)[1]``), each matched
        to its original persistence interval via ``SliceCycle.bar``.

        Every returned cycle consists solely of edges already present at
        ``threshold`` -- it is the change-of-basis cycle of the birth
        edge column in the prefix reduction, never a future death
        filling chain.  For finite bars whose original birth cycle was
        rewritten by later reductions, ``homology_chain`` certifies the
        homology-equivalent transformation.  Essential (infinite) bars
        and finite bars are both returned, distinguished by
        ``SliceCycle.essential``; zero-length bars contribute nothing
        because their birth and death happen in the same batch.
        """
        if not _finite_number(threshold):
            raise PersistenceInputError(
                "bad_threshold",
                "threshold must be a finite int or float",
            )
        cells = self.cells
        boundary, reduced, combo, owner = self._reduction()
        # Prefix of the total order = the whole batch of every
        # filtration value <= threshold.
        m = 0
        for c in cells:
            if c.filtration <= threshold:
                m += 1
            else:
                break
        # Column j only ever combines columns < j, so restricting the
        # reduction to the prefix is exact; boundary columns of the
        # prefix only reference faces inside the prefix (closure).
        p_reduced, p_combo, p_owner = _reduce(boundary[:m])
        bar_by_birth = {b.birth_index: b for b in self.h1_bars}
        out: list[SliceCycle] = []
        for j in range(m):
            if cells[j].dimension != 1 or p_reduced[j] != 0:
                continue  # not an edge, or a negative (H0-killing) edge
            if p_owner.get(j) is not None:
                continue  # class already dead inside the slice
            mask = p_combo[j]
            cycle = tuple(i for i in range(j + 1) if (mask >> i) & 1)
            bar = bar_by_birth[j]
            homology_chain = residual_chain = None
            if bar.finite:
                homology_chain, residual_chain = self._homology_witness(
                    cycle, bar, reduced, combo, owner)
            sc = SliceCycle(
                threshold, bar, cycle, homology_chain, residual_chain)
            self._check_slice_cycle(sc, m)
            out.append(sc)
        return tuple(out)

    def _homology_witness(self, cycle, bar, reduced, combo, owner):
        """Track the representative swap birth_chain -> cycle.

        Reduces ``x = cycle XOR bar.birth_chain`` against the triangle
        columns of the full-complex reduction.  Returns ``(S, r)`` with
        ``dS = x XOR r``; the leftover ``r`` is a 1-cycle whose rows no
        triangle owns, i.e. a combination of classes born strictly
        before the bar (empty iff the two representatives are
        homologous in the full complex).
        """
        x = 0
        for i in cycle:
            x ^= 1 << i
        for i in bar.birth_chain:
            x ^= 1 << i
        s = 0
        while x:
            k = owner.get(x.bit_length() - 1)
            if k is None:
                break
            x ^= reduced[k]
            s ^= combo[k]
        triangles = tuple(i for i in range(s.bit_length()) if (s >> i) & 1)
        residual = tuple(i for i in range(x.bit_length()) if (x >> i) & 1)
        return triangles, residual

    def _check_slice_cycle(self, sc: SliceCycle, m: int):
        """Self-check: closure, slice membership, witness boundary."""
        cells = self.cells
        boundary, _, _, _ = self._reduction()
        acc = 0
        for i in sc.cycle:
            if i >= m or cells[i].dimension != 1:
                raise CertificateError(
                    f"slice cycle uses cell {i} outside the current edge set"
                )
            acc ^= boundary[i]
        if acc:
            raise CertificateError("slice cycle is not a closed 1-cycle")
        if sc.homology_chain is not None:
            target = 0
            for i in sc.cycle:
                target ^= 1 << i
            for i in sc.bar.birth_chain:
                target ^= 1 << i
            res_acc = 0
            for i in sc.residual_chain:
                if i >= sc.bar.birth_index or cells[i].dimension != 1:
                    raise CertificateError(
                        f"residual chain uses cell {i} not earlier than "
                        f"birth cell {sc.bar.birth_index}")
                target ^= 1 << i
                res_acc ^= boundary[i]
            if res_acc:
                raise CertificateError("residual chain is not a cycle")
            acc = 0
            for i in sc.homology_chain:
                if cells[i].dimension != 2:
                    raise CertificateError(
                        f"homology witness uses non-triangle cell {i}")
                acc ^= boundary[i]
            if acc != target:
                raise CertificateError(
                    "homology witness boundary != "
                    "cycle XOR birth_chain XOR residual_chain")

    def __repr__(self) -> str:
        return (
            f"Persistence(cells={len(self.cells)}, "
            f"h0={len(self.h0_bars)}, h1={len(self.h1_bars)})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_persistence(simplices) -> Persistence:
    """Compute H0/H1 persistence pairings with GF(2) chain certificates.

    See module docstring / README for the full contract.  The function is
    atomic: it builds the whole result in local state and either returns a
    complete :class:`Persistence` or raises :class:`PersistenceInputError`;
    no observable state is partially modified.
    """
    records = _parse(simplices)
    ordered = sorted(
        records,
        key=lambda r: (r.filtration, r.dimension, r.vertices),
    )
    cells = [
        Cell(index=i, dimension=r.dimension, vertices=r.vertices,
             filtration=r.filtration)
        for i, r in enumerate(ordered)
    ]
    boundary = _build_boundary(cells)
    reduced, combo, owner = _reduce(boundary)

    bars: list[Bar] = []

    # H0: one bar per vertex column.
    for cell in cells:
        if cell.dimension != 0:
            continue
        killer = owner.get(cell.index)
        if killer is None:
            bars.append(Bar(0, cell.index, None, cell.filtration, INF, False))
        else:
            death_cell = cells[killer]
            bars.append(
                Bar(
                    dimension=0,
                    birth_index=cell.index,
                    death_index=killer,
                    birth=cell.filtration,
                    death=death_cell.filtration,
                    zero_length=cell.filtration == death_cell.filtration,
                )
            )

    # H1: positive edge columns (reduced column zero).  Killed by a triangle
    # -> finite bar with chains; otherwise essential.
    for cell in cells:
        if cell.dimension != 1:
            continue
        killer = owner.get(cell.index)
        if reduced[cell.index] != 0:
            continue  # negative edge: it kills an H0 class, carries no H1 bar
        if killer is None:
            bars.append(Bar(1, cell.index, None, cell.filtration, INF, False))
            continue
        death_cell = cells[killer]
        birth_mask = reduced[killer]
        death_mask = combo[killer]
        birth_chain = tuple(
            i for i in range(killer) if (birth_mask >> i) & 1
        )
        death_chain = tuple(
            i for i in range(killer + 1) if (death_mask >> i) & 1
        )
        bars.append(
            Bar(
                dimension=1,
                birth_index=cell.index,
                death_index=killer,
                birth=cell.filtration,
                death=death_cell.filtration,
                zero_length=cell.filtration == death_cell.filtration,
                birth_chain=birth_chain,
                death_chain=death_chain,
            )
        )

    n = len(cells)
    bars.sort(
        key=lambda b: (
            b.dimension,
            b.birth_index,
            n if b.death_index is None else b.death_index,
        )
    )
    result = Persistence(
        cells, bars,
        _boundary=boundary, _reduced=reduced, _combo=combo, _owner=owner,
    )
    # Self-check: every emitted finite H1 certificate must verify.  This
    # costs nothing at the supported scale and turns any internal bug into
    # a loud failure instead of a bad certificate.
    for bar in result.h1_bars:
        if bar.finite:
            check_certificate(result, bar)
    return result


# ---------------------------------------------------------------------------
# Certificate verification (uses only cell vertex data, never the reduction)
# ---------------------------------------------------------------------------

def _edge_endpoints(cells: tuple[Cell, ...], edge_index: int) -> tuple[int, int]:
    e = cells[edge_index]
    if e.dimension != 1:
        raise CertificateError(
            f"birth chain contains cell {edge_index} of dimension {e.dimension}, "
            f"expected an edge"
        )
    return e.vertices[0], e.vertices[1]


def _triangle_edges(tri_verts: tuple[int, ...]) -> list[tuple[int, int]]:
    a, b, c = tri_verts
    return [(b, c), (a, c), (a, b)]


def check_certificate(persistence: Persistence, bar: Bar) -> bool:
    """Independently verify a finite H1 bar's chain certificate.

    Checks, over GF(2):

    1. ``birth_chain`` is a non-empty 1-chain and its boundary is zero
       (parity check on edge endpoints);
    2. the bar's birth edge is the latest (maximum filtration-order) edge
       in the birth chain, and every chain edge exists at the death value;
    3. ``death_chain`` is a non-empty 2-chain containing the death triangle,
       every triangle exists at the death value, and the symmetric
       difference of the triangle boundaries equals ``birth_chain`` exactly
       (equality is bit-exact; no tolerance);
    4. the bar's endpoint values and ``zero_length`` flag agree with the
       cells.

    Returns ``True`` or raises :class:`CertificateError`.
    """
    cells = persistence.cells

    if bar.dimension != 1 or bar.death_index is None:
        raise CertificateError("certificates exist only on finite H1 bars")
    if bar.birth_chain is None or bar.death_chain is None:
        raise CertificateError("bar carries no chain certificate")

    birth_edge = cells[bar.birth_index]
    death_tri = cells[bar.death_index]
    if birth_edge.dimension != 1 or death_tri.dimension != 2:
        raise CertificateError("bar endpoints have the wrong dimensions")

    # ---- endpoint value consistency (exact) ------------------------------
    if bar.birth != birth_edge.filtration or bar.death != death_tri.filtration:
        raise CertificateError("bar endpoint values disagree with cells")
    if bar.zero_length != (bar.birth == bar.death):
        raise CertificateError("zero_length flag disagrees with endpoint values")
    if bar.death < bar.birth:
        raise CertificateError("death must be >= birth")

    # ---- birth chain: a cycle over GF(2) ---------------------------------
    birth_set = set(bar.birth_chain)
    if len(birth_set) != len(bar.birth_chain):
        raise CertificateError("birth_chain lists an edge twice")
    if not birth_set:
        raise CertificateError("birth_chain is empty")
    parity: dict[int, int] = {}
    for ei in bar.birth_chain:
        if not 0 <= ei < len(cells):
            raise CertificateError(f"birth chain index out of range: {ei}")
        a, b = _edge_endpoints(cells, ei)
        if cells[ei].filtration > death_tri.filtration:
            raise CertificateError(
                f"birth edge {cells[ei].vertices!r} appears only after death "
                f"({cells[ei].filtration!r} > {death_tri.filtration!r})"
            )
        parity[a] = parity.get(a, 0) ^ 1
        parity[b] = parity.get(b, 0) ^ 1
    odd = [v for v, p in parity.items() if p]
    if odd:
        raise CertificateError(
            f"birth chain is not a cycle: vertices with odd boundary degree: {odd}"
        )
    # The pivot row is the *bottom* edge of the reduced death column, i.e.
    # the edge with the greatest filtration-order index: that is exactly
    # where the H1 class was born.
    latest = max(bar.birth_chain)
    if latest != bar.birth_index:
        raise CertificateError(
            f"latest birth-chain edge is cell {latest}, "
            f"expected birth cell {bar.birth_index}"
        )

    # ---- death chain: 2-chain whose boundary == birth chain --------------
    death_set = set(bar.death_chain)
    if len(death_set) != len(bar.death_chain):
        raise CertificateError("death_chain lists a triangle twice")
    if not death_set:
        raise CertificateError("death_chain is empty")
    if bar.death_index not in death_set:
        raise CertificateError("death_chain does not contain the death triangle")

    edge_index: dict[tuple[int, int], int] = {}
    for c in cells:
        if c.dimension == 1:
            edge_index[c.vertices] = c.index
    produced: set[int] = set()
    for ti in bar.death_chain:
        if not 0 <= ti < len(cells):
            raise CertificateError(f"death chain index out of range: {ti}")
        tri = cells[ti]
        if tri.dimension != 2:
            raise CertificateError(
                f"death chain contains cell {ti} of dimension {tri.dimension}, "
                f"expected a triangle"
            )
        if ti > bar.death_index:
            raise CertificateError(
                f"death chain uses cell {ti} later than death column "
                f"{bar.death_index}"
            )
        if tri.filtration > death_tri.filtration:
            raise CertificateError(
                f"triangle {tri.vertices!r} has filtration {tri.filtration!r} "
                f"> death {death_tri.filtration!r}"
            )
        for ev in _triangle_edges(tri.vertices):
            ei = edge_index.get(ev)
            if ei is None:  # pragma: no cover - closure validated on input
                raise CertificateError(f"missing edge face {ev!r}")
            if ei in produced:
                produced.discard(ei)
            else:
                produced.add(ei)

    if produced != birth_set:
        only_chain = birth_set - produced
        only_boundary = produced - birth_set
        raise CertificateError(
            "boundary of death chain != birth chain "
            f"(only in birth_chain: {sorted(only_chain)}, "
            f"only in triangle boundary: {sorted(only_boundary)})"
        )

    return True
