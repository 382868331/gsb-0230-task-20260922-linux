"""Cup product H^1 x H^1 -> H^2 on a fixed filtration slice, over GF(2).

Given a finite simplicial complex of dimension at most 2 (the same input
contract as :func:`pershom.compute_persistence`) and an optional filtration
threshold, this module builds the cochain spaces of the slice complex,
computes a basis of H^1 (cocycles modulo coboundaries) and of H^2, and
evaluates the cup product of two 1-cocycles with the simplicial
Alexander-Whitney formula under the global vertex order:

    (alpha ~ beta)(a, b, c) = alpha((a, b)) * beta((b, c))      (a < b < c)

The raw 2-cochain is then reduced to coordinates in the computed H^2 basis
by exact GF(2) elimination.  Everything is deterministic, standard-library
only, and uses the same bitset-over-GF(2) style as persistence.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .persistence import (
    CertificateError,
    PersistenceInputError,
    _finite_number,
    _parse,
)

__all__ = [
    "CupProduct",
    "CohomologySlice",
    "compute_cohomology",
    "check_cup_product",
]


# ---------------------------------------------------------------------------
# GF(2) linear algebra on Python-int bitsets
# ---------------------------------------------------------------------------

def _echelon(masks) -> dict[int, int]:
    """Reduced row echelon basis of the span of masks.

    Returns a pivot -> mask dict where each pivot (highest set bit)
    occurs in exactly one row.
    """
    pivots: dict[int, int] = {}
    for m in masks:
        while m:
            p = m.bit_length() - 1
            q = pivots.get(p)
            if q is None:
                for r, row in list(pivots.items()):
                    if (row >> p) & 1:
                        pivots[r] = row ^ m
                pivots[p] = m
                break
            m ^= q
    return pivots


def _reduce(mask: int, pivots: dict[int, int]) -> int:
    """Reduce mask by a pivot basis; returns the remainder."""
    while mask:
        p = mask.bit_length() - 1
        q = pivots.get(p)
        if q is None:
            return mask
        mask ^= q
    return 0


def _nullspace(equations, nvars: int) -> list[int]:
    """Basis of the GF(2) nullspace of equation masks over nvars bits."""
    pivots = _echelon(equations)
    basis: list[int] = []
    for v in range(nvars):
        if v in pivots:
            continue
        x = 1 << v
        for p, eq in pivots.items():
            # RREF: non-pivot entries of eq are free variables only.
            if bin(eq & ~(1 << p) & x).count("1") & 1:
                x |= 1 << p
        basis.append(x)
    return basis


def _as_mask(cocycle, n: int, what: str) -> int:
    """Accept an int bitset or an iterable of indices; return the bitset."""
    if isinstance(cocycle, bool):
        raise PersistenceInputError(
            "bad_cochain", f"{what} must be an int bitset or an iterable of indices"
        )
    if isinstance(cocycle, int):
        if cocycle < 0 or cocycle >= (1 << n):
            raise PersistenceInputError(
                "bad_cochain", f"{what} bitset out of range for {n} cells"
            )
        return cocycle
    mask = 0
    try:
        iterator = iter(cocycle)
    except TypeError:
        raise PersistenceInputError(
            "bad_cochain", f"{what} must be an int bitset or an iterable of indices"
        ) from None
    for i in iterator:
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
            raise PersistenceInputError(
                "bad_cochain", f"{what} contains invalid index {i!r} (need 0..{n - 1})"
            )
        mask ^= 1 << i
    return mask


def _bits(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(mask.bit_length()) if (mask >> i) & 1)


# ---------------------------------------------------------------------------
# Result of one cup-product query
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CupProduct:
    """One H^1 x H^1 -> H^2 cup-product evaluation.

    cocycle_a / cocycle_b are the two input 1-cocycles as sorted tuples of
    edge indices (into CohomologySlice.edges); raw_product is the
    Alexander-Whitney 2-cochain as sorted triangle indices (into
    CohomologySlice.triangles); h2_coordinates are the coordinates of the
    product class in CohomologySlice.h2_basis; is_zero is True iff the
    product class vanishes in H^2 (i.e. the raw cochain is a coboundary).
    """

    cocycle_a: tuple[int, ...]
    cocycle_b: tuple[int, ...]
    raw_product: tuple[int, ...]
    h2_coordinates: tuple[int, ...]
    is_zero: bool


# ---------------------------------------------------------------------------
# Cohomology of one filtration slice
# ---------------------------------------------------------------------------

class CohomologySlice:
    """Cohomology spaces of a fixed slice complex (dimension <= 2, GF(2)).

    Attributes:
        vertices / edges / triangles: vertex tuples of the slice cells,
            each sorted lexicographically; cocycles and cochains are
            bitsets (or index tuples) over edges / triangles.
        b1_basis: basis of the 1-coboundary space B^1 = im(delta_0).
        h1_basis: 1-cocycle representatives of an H^1 basis.
        h2_basis: 2-cochain representatives of an H^2 basis.
    """

    def __init__(self, vertices, edges, triangles):
        self.vertices: tuple[tuple[int, ...], ...] = tuple(vertices)
        self.edges: tuple[tuple[int, int], ...] = tuple(edges)
        self.triangles: tuple[tuple[int, int, int], ...] = tuple(triangles)

        n_e = len(self.edges)
        n_t = len(self.triangles)
        edge_index = {e: i for i, e in enumerate(self.edges)}
        self._edge_index = edge_index

        # delta_0: C^0 -> C^1, one row per vertex (mask of incident edges).
        delta0_rows = [0] * len(self.vertices)
        vertex_index = {v: i for i, v in enumerate(self.vertices)}
        for ei, (a, b) in enumerate(self.edges):
            delta0_rows[vertex_index[(a,)]] |= 1 << ei
            delta0_rows[vertex_index[(b,)]] |= 1 << ei
        self._b1 = _echelon(delta0_rows)

        # Cocycle equations: for each triangle, the edge mask of its boundary.
        tri_boundary = []
        # delta_1 columns: C^1 -> C^2, one mask per edge (incident triangles).
        delta1_cols = [0] * n_e
        for ti, (a, b, c) in enumerate(self.triangles):
            mask = (
                (1 << edge_index[(a, b)])
                | (1 << edge_index[(a, c)])
                | (1 << edge_index[(b, c)])
            )
            tri_boundary.append(mask)
            for e in ((a, b), (a, c), (b, c)):
                delta1_cols[edge_index[e]] |= 1 << ti
        self._b2 = _echelon(delta1_cols)

        # Z^1 = ker delta_1, then H^1 = Z^1 / B^1.
        z1_basis = _nullspace(tri_boundary, n_e)
        combined = dict(self._b1)
        h1: list[int] = []
        for z in z1_basis:
            r = _reduce(z, combined)
            if r:
                combined[r.bit_length() - 1] = r
                h1.append(r)
        self._h1 = h1

        # H^2 = C^2 / B^2 (no 3-simplices, so Z^2 = C^2).
        combined2 = dict(self._b2)
        h2: list[int] = []
        for t in range(n_t):
            r = _reduce(1 << t, combined2)
            if r:
                combined2[r.bit_length() - 1] = r
                h2.append(r)
        self._h2 = h2
        self._h2_pivots = {r.bit_length() - 1: r for r in h2}

    # -- bases as index tuples ---------------------------------------------

    @property
    def b1_basis(self) -> tuple[tuple[int, ...], ...]:
        return tuple(_bits(m) for m in self._b1.values())

    @property
    def h1_basis(self) -> tuple[tuple[int, ...], ...]:
        return tuple(_bits(m) for m in self._h1)

    @property
    def h2_basis(self) -> tuple[tuple[int, ...], ...]:
        return tuple(_bits(m) for m in self._h2)

    @property
    def betti1(self) -> int:
        return len(self._h1)

    @property
    def betti2(self) -> int:
        return len(self._h2)

    # -- queries -----------------------------------------------------------

    def check_closed(self, cocycle) -> bool:
        """True iff the given 1-cochain is a cocycle (delta zero on every
        triangle); raises PersistenceInputError otherwise."""
        mask = _as_mask(cocycle, len(self.edges), "cocycle")
        for a, b, c in self.triangles:
            parity = (
                ((mask >> self._edge_index[(a, b)]) & 1)
                ^ ((mask >> self._edge_index[(a, c)]) & 1)
                ^ ((mask >> self._edge_index[(b, c)]) & 1)
            )
            if parity:
                raise PersistenceInputError(
                    "not_a_cocycle",
                    f"cochain is not closed on triangle {(a, b, c)!r}",
                )
        return True

    def reduce_h2(self, cochain) -> tuple[int, ...]:
        """H^2 coordinates of a 2-cochain in h2_basis."""
        mask = _as_mask(cochain, len(self.triangles), "2-cochain")
        r = _reduce(mask, self._b2)
        coords = [0] * len(self._h2)
        while r:
            p = r.bit_length() - 1
            q = self._h2_pivots.get(p)
            if q is None:  # pragma: no cover - H^2 basis spans C^2/B^2
                raise CertificateError("2-cochain not spanned by the H^2 basis")
            coords[self._h2.index(q)] = 1
            r ^= q
        return tuple(coords)

    def cup_product(self, i: int, j: int) -> CupProduct:
        """Cup product of H^1 basis cocycles i and j."""
        for k in (i, j):
            if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < len(self._h1):
                raise PersistenceInputError(
                    "bad_basis_index",
                    f"basis index {k!r} out of range (H^1 dimension {len(self._h1)})",
                )
        return self.cup_product_of(self._h1[i], self._h1[j])

    def cup_product_of(self, cocycle_a, cocycle_b) -> CupProduct:
        """Cup product of two explicit 1-cocycles (bitsets or edge indices)."""
        am = _as_mask(cocycle_a, len(self.edges), "cocycle_a")
        bm = _as_mask(cocycle_b, len(self.edges), "cocycle_b")
        self.check_closed(am)
        self.check_closed(bm)
        raw = 0
        for ti, (a, b, c) in enumerate(self.triangles):
            if ((am >> self._edge_index[(a, b)]) & 1) and (
                (bm >> self._edge_index[(b, c)]) & 1
            ):
                raw |= 1 << ti
        coords = self.reduce_h2(raw)
        return CupProduct(
            cocycle_a=_bits(am),
            cocycle_b=_bits(bm),
            raw_product=_bits(raw),
            h2_coordinates=coords,
            is_zero=not any(coords),
        )

    def __repr__(self) -> str:
        return (
            f"CohomologySlice(V={len(self.vertices)}, E={len(self.edges)}, "
            f"T={len(self.triangles)}, beta1={self.betti1}, beta2={self.betti2})"
        )


# ---------------------------------------------------------------------------
# Independent verification of a cup-product result
# ---------------------------------------------------------------------------

def check_cup_product(slice_: CohomologySlice, cp: CupProduct) -> bool:
    """Independently verify a CupProduct against the slice.

    Recomputes, using only the vertex tuples of the slice cells:

    1. both cocycles are closed (even parity on every triangle boundary);
    2. the raw Alexander-Whitney 2-cochain from the two cocycles equals
       cp.raw_product exactly (set equality, no tolerance);
    3. cp.h2_coordinates are the H^2 coordinates of that raw cochain and
       cp.is_zero agrees with them.

    Returns True or raises CertificateError.
    """
    edges = slice_.edges
    triangles = slice_.triangles

    def edge_set(indices, field):
        out = set()
        for ei in indices:
            if isinstance(ei, bool) or not isinstance(ei, int) or not 0 <= ei < len(edges):
                raise CertificateError(f"{field} contains invalid edge index {ei!r}")
            out.add(edges[ei])
        return out

    a_edges = edge_set(cp.cocycle_a, "cocycle_a")
    b_edges = edge_set(cp.cocycle_b, "cocycle_b")

    for name, es in (("cocycle_a", a_edges), ("cocycle_b", b_edges)):
        for (x, y, z) in triangles:
            parity = ((x, y) in es) + ((x, z) in es) + ((y, z) in es)
            if parity % 2:
                raise CertificateError(
                    f"{name} is not closed on triangle {(x, y, z)!r}"
                )

    raw = set()
    for ti, (x, y, z) in enumerate(triangles):
        if (x, y) in a_edges and (y, z) in b_edges:
            raw.add(ti)
    if raw != set(cp.raw_product):
        raise CertificateError(
            "raw_product does not match the Alexander-Whitney recomputation "
            f"(only in result: {sorted(set(cp.raw_product) - raw)}, "
            f"only recomputed: {sorted(raw - set(cp.raw_product))})"
        )
    for ti in cp.raw_product:
        if isinstance(ti, bool) or not isinstance(ti, int) or not 0 <= ti < len(triangles):
            raise CertificateError(f"raw_product contains invalid triangle index {ti!r}")

    coords = slice_.reduce_h2(tuple(sorted(raw)))
    if tuple(coords) != tuple(cp.h2_coordinates):
        raise CertificateError(
            f"h2_coordinates {cp.h2_coordinates!r} disagree with the "
            f"recomputed reduction {coords!r}"
        )
    if cp.is_zero != (not any(coords)):
        raise CertificateError("is_zero flag disagrees with h2_coordinates")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_cohomology(simplices, threshold=None) -> CohomologySlice:
    """Build the cohomology slice of a filtered complex at threshold.

    simplices follows the same contract as pershom.compute_persistence
    (validated identically, including the 300-simplex bound).  The slice
    consists of every simplex with filtration <= threshold; with
    threshold=None all simplices are included.  The slice is automatically
    a closed subcomplex because face filtration values never exceed
    simplex values.
    """
    records = _parse(simplices)
    if threshold is not None and not _finite_number(threshold):
        raise PersistenceInputError(
            "bad_threshold", "threshold must be a finite int or float"
        )
    vertices, edges, triangles = [], [], []
    for r in records:
        if threshold is not None and r.filtration > threshold:
            continue
        if r.dimension == 0:
            vertices.append(r.vertices)
        elif r.dimension == 1:
            edges.append(r.vertices)
        else:
            triangles.append(r.vertices)
    return CohomologySlice(sorted(vertices), sorted(edges), sorted(triangles))
