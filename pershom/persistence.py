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
    "SlicedCycle",
    "SliceCycles",
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
class SlicedCycle:
    """One H1 generator of the threshold subcomplex, tied to its bar.

    Attributes:
        bar: the original persistence Bar this generator corresponds to
            (its [birth, death) interval contains the slice threshold).
        cycle: sorted tuple of Persistence.cells edge indices; a GF(2)
            1-cycle built only from edges already present at the slice
            threshold (never from the future death-filling chain).
        essential: True when bar is an infinite (essential) bar; the
            cycle is then the reduced representative tracked through the
            change-of-basis mask of the column reduction (a homology-
            preserving transformation of the original birth cycle).
    """

    bar: Bar
    cycle: tuple[int, ...]
    essential: bool

    def __repr__(self) -> str:
        kind = 'essential' if self.essential else 'finite'
        return (
            f'SlicedCycle({kind}, bar=[{self.bar.birth!r}, '
            f'{self.bar.death!r}), edges={list(self.cycle)})'
        )


class SliceCycles:
    """Result of Persistence.slice_cycles.

    Attributes:
        threshold: the slice value t.
        generators: tuple of SlicedCycle, one per H1 class alive at t
            (its length equals betti_at(t)[1]).
        zero_length_bars: zero-length H1 bars with birth <= t.  They are
            listed separately for traceability and never occupy a
            generator slot: the whole same-value batch is in the
            subcomplex, so a class born and killed inside one batch
            contributes nothing.
    """

    def __init__(self, threshold, generators, zero_length_bars):
        self.threshold = threshold
        self.generators: tuple[SlicedCycle, ...] = tuple(generators)
        self.zero_length_bars: tuple[Bar, ...] = tuple(zero_length_bars)

    def __iter__(self):
        return iter(self.generators)

    def __len__(self) -> int:
        return len(self.generators)

    def __repr__(self) -> str:
        return (
            f'SliceCycles(t={self.threshold!r}, '
            f'generators={len(self.generators)}, '
            f'zero_length={len(self.zero_length_bars)})'
        )


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


def _gf2_rank(vectors: Iterable[int]) -> int:
    """Rank of GF(2) vectors given as Python-int bitsets."""
    pivots: dict[int, int] = {}
    for vec in vectors:
        v = vec
        while v:
            pivot = v.bit_length() - 1
            if pivot in pivots:
                v ^= pivots[pivot]
            else:
                pivots[pivot] = v
                break
    return len(pivots)


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

    def __init__(self, cells: list[Cell], bars: list[Bar]):
        self.cells: tuple[Cell, ...] = tuple(cells)
        self.bars: tuple[Bar, ...] = tuple(bars)
        self.h0_bars: tuple[Bar, ...] = tuple(b for b in bars if b.dimension == 0)
        self.h1_bars: tuple[Bar, ...] = tuple(b for b in bars if b.dimension == 1)
        # Reduction by-products kept for slice_cycles: combo[j] is the
        # change-of-basis mask of column j (a 1-cycle when column j is a
        # positive edge column).  Not part of the public contract.
        self._combo: tuple[int, ...] | None = None

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

    def slice_cycles(self, threshold) -> SliceCycles:
        """H1 generators of the subcomplex with filtration <= threshold.

        The subcomplex takes every cell with filtration <= threshold as
        one atomic batch: all simplices sharing a filtration value are in
        or out together, regardless of their internal tie-break order, so
        a slice never observes half of a moment.

        Returns a SliceCycles whose generators correspond one-to-one to
        the H1 bars alive at the threshold (birth <= t < death).  Each
        generator cycle uses only edges already present at the threshold:

        * finite bars contribute their birth cycle (every edge of it has
          filtration <= birth <= t by the filtration total order);
        * essential (infinite) bars contribute the reduced representative
          recovered from the change-of-basis mask of the birth column --
          the homology-equivalent transformation accumulated while later
          columns were reduced is exactly what that mask tracks;
        * zero-length bars are reported separately in zero_length_bars
          and never occupy a generator slot.

        Before returning, the slice is checked against a boundary matrix
        rebuilt from scratch for the threshold subcomplex: every cycle
        must be closed, the cycles must be linearly independent modulo
        boundaries, and their number must equal beta1 of the subcomplex.
        A failure raises CertificateError instead of returning a bad
        slice.  threshold must be a finite int/float.
        """
        if not _finite_number(threshold):
            raise PersistenceInputError(
                'bad_threshold',
                'threshold must be a finite int or float',
            )
        generators: list[SlicedCycle] = []
        for bar in self.h1_bars:
            if bar.zero_length or not bar.alive_at(threshold):
                continue
            if bar.death_index is None:
                mask = self._combo[bar.birth_index]
                cycle = tuple(
                    i for i in range(bar.birth_index + 1) if (mask >> i) & 1
                )
                generators.append(SlicedCycle(bar=bar, cycle=cycle, essential=True))
            else:
                generators.append(
                    SlicedCycle(bar=bar, cycle=bar.birth_chain, essential=False)
                )
        zero_length = tuple(
            b for b in self.h1_bars if b.zero_length and b.birth <= threshold
        )
        sliced = SliceCycles(threshold, generators, zero_length)
        self._verify_slice(sliced)
        return sliced

    def _verify_slice(self, sliced: SliceCycles) -> None:
        """Independently re-check a slice against a rebuilt boundary matrix.

        Uses only cell vertex/filtration data and fresh GF(2) rank
        computations -- never the reduction that produced the cycles.
        """
        t = sliced.threshold
        cells = self.cells
        verts = [c for c in cells if c.dimension == 0 and c.filtration <= t]
        edges = [c for c in cells if c.dimension == 1 and c.filtration <= t]
        tris = [c for c in cells if c.dimension == 2 and c.filtration <= t]
        vpos = {c.vertices[0]: k for k, c in enumerate(verts)}
        epos = {c.index: k for k, c in enumerate(edges)}
        edge_by_verts = {c.vertices: c for c in edges}

        cycle_vecs: list[int] = []
        for gen in sliced.generators:
            vec = 0
            parity: dict[int, int] = {}
            seen: set[int] = set()
            for ei in gen.cycle:
                cell = cells[ei]
                if cell.dimension != 1:
                    raise CertificateError(
                        f'slice cycle contains cell {ei} of dimension '
                        f'{cell.dimension}, expected an edge'
                    )
                if ei in seen:
                    raise CertificateError(
                        f'slice cycle lists edge {ei} twice'
                    )
                seen.add(ei)
                if cell.filtration > t:
                    raise CertificateError(
                        f'slice cycle uses edge {cell.vertices!r} with '
                        f'filtration {cell.filtration!r} > slice {t!r}'
                    )
                vec ^= 1 << epos[ei]
                a, b = cell.vertices
                parity[a] = parity.get(a, 0) ^ 1
                parity[b] = parity.get(b, 0) ^ 1
            odd = sorted(v for v, p in parity.items() if p)
            if odd:
                raise CertificateError(
                    f'slice cycle is not closed: odd boundary at {odd}'
                )
            if not seen:
                raise CertificateError('slice cycle is empty')
            cycle_vecs.append(vec)

        d1 = [
            (1 << vpos[c.vertices[0]]) | (1 << vpos[c.vertices[1]])
            for c in edges
        ]
        d2 = []
        for c in tris:
            vec = 0
            for fv in _faces(c.vertices):
                vec ^= 1 << epos[edge_by_verts[fv].index]
            d2.append(vec)

        rank_bounds = _gf2_rank(d2)
        if _gf2_rank(d2 + cycle_vecs) != rank_bounds + len(cycle_vecs):
            raise CertificateError(
                'slice cycles are linearly dependent modulo boundaries'
            )
        beta1 = (len(edges) - _gf2_rank(d1)) - rank_bounds
        if beta1 != len(sliced.generators):
            raise CertificateError(
                f'slice has {len(sliced.generators)} generators but the '
                f'rebuilt boundary matrices give beta1={beta1}'
            )

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
    result = Persistence(cells, bars)
    result._combo = tuple(combo)
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
