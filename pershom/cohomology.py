"""Cohomology of a fixed filtration slice and the H1 x H1 -> H2 cup product.

Everything here lives over GF(2) and uses the Python standard library only.
The cup product is evaluated with the simplicial Alexander-Whitney formula
under the vertex total order (vertex ids are required strictly increasing,
which fixes the order): for a triangle (u, v, w) with u < v < w,

    (a ~ b)([u, v, w]) = a([u, v]) * b([v, w])   (mod 2).

The pipeline is: build the slice subcomplex, compute cocycle/coboundary
spaces and H^1 / H^2 bases by GF(2) elimination, evaluate the raw product
2-cochain, then reduce it modulo coboundaries to H^2 coordinates.  Betti
numbers alone are never compared; the product is always pinned down as a
concrete cohomology class.
"""

from __future__ import annotations

from dataclasses import dataclass

from .persistence import (
    PersistenceInputError,
    _finite_number,
    _parse,
)

__all__ = [
    "CohomologySlice",
    "CupProductResult",
    "cohomology_slice",
    "cup_product",
]


# ---------------------------------------------------------------------------
# GF(2) linear algebra on int bitsets
# ---------------------------------------------------------------------------

def _pivot_table(vectors):
    """Insert bitset vectors into a pivot table {pivot_bit: vector}."""
    table = {}
    for vec in vectors:
        x = vec
        while x:
            p = x.bit_length() - 1
            if p in table:
                x ^= table[p]
            else:
                table[p] = x
                break
    return table


def _kernel_basis(columns):
    """Basis of ker of the GF(2) map whose columns are the given bitsets.

    Returns bitsets over domain (column) indices.
    """
    pivots = {}
    kernel = []
    for j, col in enumerate(columns):
        x, combo = col, 1 << j
        while x:
            p = x.bit_length() - 1
            if p in pivots:
                x ^= pivots[p][0]
                combo ^= pivots[p][1]
            else:
                pivots[p] = (x, combo)
                break
        if x == 0:
            kernel.append(combo)
    return kernel


def _quotient_basis(sub_basis, ambient_basis):
    """Basis of ambient / sub: independent residuals of ambient mod sub."""
    table = _pivot_table(sub_basis)
    out = []
    for vec in ambient_basis:
        x = vec
        while x:
            p = x.bit_length() - 1
            if p in table:
                x ^= table[p]
            else:
                table[p] = x
                out.append(x)
                break
    return out


def _coordinates(target, sub_basis, quotient_basis):
    """Write target = (subspace part) + sum coords[i] * quotient_basis[i].

    Returns the coordinate tuple over GF(2), or None when target does not
    lie in sub + quotient.
    """
    combined = list(sub_basis) + list(quotient_basis)
    n_sub = len(sub_basis)
    pivots = {}
    for j, vec in enumerate(combined):
        x, combo = vec, 1 << j
        while x:
            p = x.bit_length() - 1
            if p in pivots:
                x ^= pivots[p][0]
                combo ^= pivots[p][1]
            else:
                pivots[p] = (x, combo)
                break
    x, combo = target, 0
    while x:
        p = x.bit_length() - 1
        if p in pivots:
            x ^= pivots[p][0]
            combo ^= pivots[p][1]
        else:
            return None
    return tuple((combo >> (n_sub + i)) & 1 for i in range(len(quotient_basis)))


def _bits(mask):
    i = 0
    while mask:
        if mask & 1:
            yield i
        i += 1
        mask >>= 1


# ---------------------------------------------------------------------------
# Cohomology of one filtration slice
# ---------------------------------------------------------------------------

class CohomologySlice:
    """GF(2) cohomology spaces of the subcomplex at one filtration value.

    Attributes:
        threshold: the slice parameter t (cells with filtration <= t).
        vertices / edges / triangles: cells of the slice, sorted by vertex
            tuple; cochains are subsets of these (GF(2)).
        h1_basis: basis of H^1; each entry is a sorted tuple of edges
            (vertex pairs) forming a 1-cocycle class representative.
        h2_basis: basis of H^2; each entry is a sorted tuple of triangles
            forming a 2-cocycle class representative.
        b0 / b1 / b2: Betti numbers of the slice.
    """

    def __init__(self, threshold, vertices, edges, triangles,
                 h1_basis, h2_basis, b0):
        self.threshold = threshold
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles
        self.h1_basis = h1_basis
        self.h2_basis = h2_basis
        self.b0 = b0
        self.b1 = len(h1_basis)
        self.b2 = len(h2_basis)
        # Internal bitset data (positions in the tuples above).
        self._edge_pos = {e: i for i, e in enumerate(edges)}
        self._b2_coboundaries = []   # B^2 = im delta_1 (bitsets over triangles)
        self._h2_masks = []

    def __repr__(self):
        return (
            "CohomologySlice(t=%r, cells=(%d, %d, %d), betti=(%d, %d, %d))"
            % (self.threshold, len(self.vertices), len(self.edges),
               len(self.triangles), self.b0, self.b1, self.b2)
        )


def cohomology_slice(simplices, threshold):
    """Build the GF(2) cohomology spaces of the slice at threshold.

    'simplices' uses exactly the same contract as
    'pershom.compute_persistence' (validated by the same parser);
    'threshold' must be a finite int/float.  The slice is the subcomplex
    of cells with filtration <= threshold.
    """
    if not _finite_number(threshold):
        raise PersistenceInputError(
            "bad_threshold", "threshold must be a finite int or float"
        )
    records = _parse(simplices)
    cells = sorted(
        (r for r in records if r.filtration <= threshold),
        key=lambda r: (r.dimension, r.vertices),
    )
    vertices = tuple(r.vertices for r in cells if r.dimension == 0)
    edges = tuple(r.vertices for r in cells if r.dimension == 1)
    triangles = tuple(r.vertices for r in cells if r.dimension == 2)

    vert_pos = {v[0]: i for i, v in enumerate(vertices)}
    edge_pos = {e: i for i, e in enumerate(edges)}

    # delta_0: C^0 -> C^1; column per vertex = its incident edges.
    d0_cols = [0] * len(vertices)
    for j, (a, b) in enumerate(edges):
        d0_cols[vert_pos[a]] |= 1 << j
        d0_cols[vert_pos[b]] |= 1 << j

    # delta_1: C^1 -> C^2; column per edge = triangles containing it.
    d1_cols = [0] * len(edges)
    for t, (a, b, c) in enumerate(triangles):
        bit = 1 << t
        d1_cols[edge_pos[(b, c)]] |= bit
        d1_cols[edge_pos[(a, c)]] |= bit
        d1_cols[edge_pos[(a, b)]] |= bit

    # B^1 = im delta_0, Z^1 = ker delta_1, H^1 = Z^1 / B^1.
    b1_coboundary_masks = list(_pivot_table(d0_cols).values())
    z1_masks = _kernel_basis(d1_cols)
    h1_masks = _quotient_basis(b1_coboundary_masks, z1_masks)

    # B^2 = im delta_1, Z^2 = C^2 (no 3-simplices), H^2 = C^2 / B^2.
    b2_coboundary_masks = list(_pivot_table(d1_cols).values())
    all_tri_units = [1 << i for i in range(len(triangles))]
    h2_masks = _quotient_basis(b2_coboundary_masks, all_tri_units)

    def to_edges(mask):
        return tuple(edges[i] for i in _bits(mask))

    def to_tris(mask):
        return tuple(triangles[i] for i in _bits(mask))

    result = CohomologySlice(
        threshold=threshold,
        vertices=vertices,
        edges=edges,
        triangles=triangles,
        h1_basis=tuple(to_edges(m) for m in h1_masks),
        h2_basis=tuple(to_tris(m) for m in h2_masks),
        b0=len(vertices) - len(b1_coboundary_masks),
    )
    result._b2_coboundaries = b2_coboundary_masks
    result._h2_masks = h2_masks
    return result


# ---------------------------------------------------------------------------
# Alexander-Whitney cup product
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CupProductResult:
    """Result of one H^1 x H^1 -> H^2 cup product query.

    Attributes:
        left / right: the two input 1-cocycles, normalised to sorted tuples
            of vertex pairs.
        raw: the raw Alexander-Whitney 2-cochain: sorted tuple of triangles
            (u, v, w) with left(u, v) == right(v, w) == 1.
        coordinates: GF(2) coordinates of the product class in
            CohomologySlice.h2_basis.
        is_zero: whether the product class vanishes in H^2.
    """

    left: tuple
    right: tuple
    raw: tuple
    coordinates: tuple
    is_zero: bool


def _normalise_cocycle(cocycle, slice_, name):
    if isinstance(cocycle, (str, bytes)):
        raise PersistenceInputError(
            "bad_cocycle", name + " must be an iterable of vertex pairs"
        )
    try:
        items = list(cocycle)
    except TypeError:
        raise PersistenceInputError(
            "bad_cocycle", name + " must be an iterable of vertex pairs"
        ) from None
    edges = set()
    for item in items:
        ok = (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        )
        if not ok:
            raise PersistenceInputError(
                "bad_cocycle",
                "%s entries must be pairs of int vertex ids, got %r" % (name, item),
            )
        a, b = item
        edge = (a, b) if a < b else (b, a)
        if a == b or edge not in slice_._edge_pos:
            raise PersistenceInputError(
                "edge_not_in_slice",
                "%s uses edge %r, which is not an edge of the slice at t=%r"
                % (name, tuple(item), slice_.threshold),
            )
        edges.add(edge)
    mask = 0
    for e in edges:
        mask |= 1 << slice_._edge_pos[e]
    # Closedness: the coboundary must vanish on every triangle (GF(2)).
    for (a, b, c) in slice_.triangles:
        total = (
            ((mask >> slice_._edge_pos[(a, b)]) & 1)
            ^ ((mask >> slice_._edge_pos[(a, c)]) & 1)
            ^ ((mask >> slice_._edge_pos[(b, c)]) & 1)
        )
        if total:
            raise PersistenceInputError(
                "cocycle_not_closed",
                "%s is not closed: its coboundary is non-zero on triangle %r"
                % (name, (a, b, c)),
            )
    return tuple(sorted(edges)), mask


def cup_product(slice_, left, right):
    """Cup product of two 1-cocycles on a fixed slice, reduced to H^2.

    'left' / 'right' are iterables of vertex pairs (order within a pair
    is ignored; both must be edges of the slice and GF(2)-closed, otherwise
    'PersistenceInputError' is raised).  The raw cochain is evaluated with
    the Alexander-Whitney formula under the vertex total order and then
    reduced modulo coboundaries to coordinates in 'slice_.h2_basis'.
    """
    if not isinstance(slice_, CohomologySlice):
        raise PersistenceInputError(
            "bad_input", "slice must come from pershom.cohomology_slice"
        )
    left_edges, left_mask = _normalise_cocycle(left, slice_, "left")
    right_edges, right_mask = _normalise_cocycle(right, slice_, "right")

    # Alexander-Whitney: (left ~ right)([u,v,w]) = left([u,v]) * right([v,w]).
    raw_mask = 0
    for t_idx, (u, v, w) in enumerate(slice_.triangles):
        front = (left_mask >> slice_._edge_pos[(u, v)]) & 1
        back = (right_mask >> slice_._edge_pos[(v, w)]) & 1
        if front & back:
            raw_mask |= 1 << t_idx

    coords = _coordinates(raw_mask, slice_._b2_coboundaries, slice_._h2_masks)
    if coords is None:  # pragma: no cover - product of cocycles is closed
        raise AssertionError("cup product is not a cocycle; internal error")
    raw = tuple(slice_.triangles[i] for i in _bits(raw_mask))
    return CupProductResult(
        left=left_edges,
        right=right_edges,
        raw=raw,
        coordinates=coords,
        is_zero=not any(coords),
    )

