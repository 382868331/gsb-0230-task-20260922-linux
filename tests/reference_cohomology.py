"""Independent GF(2) reference for the cohomology cup product.

This module deliberately does NOT import pershom.  It re-implements, with
plain Gaussian elimination over GF(2) on Python-int bitsets, everything the
tests cross-check against pershom.cohomology:

- coboundary ranks and Betti numbers of a <=2-dimensional complex,
- closedness of 1-cocycles,
- the Alexander-Whitney cup product of two 1-cocycles,
- reduction of a 2-cochain to H^2 coordinates.

The library under test is only fed the same vertex-tuple complex; all
expected values below come from this file.
"""


def echelon(masks):
    """Pivot -> row dict, reduced row echelon over GF(2)."""
    pivots = {}
    for m in masks:
        while m:
            p = m.bit_length() - 1
            q = pivots.get(p)
            if q is None:
                for r in list(pivots):
                    if (pivots[r] >> p) & 1:
                        pivots[r] ^= m
                pivots[p] = m
                break
            m ^= q
    return pivots


def reduce_by(mask, pivots):
    while mask:
        p = mask.bit_length() - 1
        q = pivots.get(p)
        if q is None:
            return mask
        mask ^= q
    return 0


def nullspace(equations, nvars):
    pivots = echelon(equations)
    basis = []
    for v in range(nvars):
        if v in pivots:
            continue
        x = 1 << v
        for p, eq in pivots.items():
            if bin(eq & ~(1 << p) & x).count("1") & 1:
                x |= 1 << p
        basis.append(x)
    return basis


class ReferenceSlice:
    """Cohomology of a complex given as vertex-tuple lists."""

    def __init__(self, vertices, edges, triangles):
        self.vertices = sorted(vertices)
        self.edges = sorted(edges)
        self.triangles = sorted(triangles)
        eidx = {e: i for i, e in enumerate(self.edges)}
        vidx = {v: i for i, v in enumerate(self.vertices)}

        d0 = [0] * len(self.vertices)
        for ei, (a, b) in enumerate(self.edges):
            d0[vidx[(a,)]] |= 1 << ei
            d0[vidx[(b,)]] |= 1 << ei
        self.b1 = echelon(d0)

        tri_eq = []
        d1 = [0] * len(self.edges)
        for ti, (a, b, c) in enumerate(self.triangles):
            tri_eq.append(
                (1 << eidx[(a, b)]) | (1 << eidx[(a, c)]) | (1 << eidx[(b, c)])
            )
            for e in ((a, b), (a, c), (b, c)):
                d1[eidx[e]] |= 1 << ti
        self.b2 = echelon(d1)

        z1 = nullspace(tri_eq, len(self.edges))
        combined = dict(self.b1)
        self.h1 = []
        for z in z1:
            r = reduce_by(z, combined)
            if r:
                combined[r.bit_length() - 1] = r
                self.h1.append(r)

        combined2 = dict(self.b2)
        self.h2 = []
        for t in range(len(self.triangles)):
            r = reduce_by(1 << t, combined2)
            if r:
                combined2[r.bit_length() - 1] = r
                self.h2.append(r)
        self.h2_pivots = {r.bit_length() - 1: r for r in self.h2}

    @property
    def betti1(self):
        return len(self.h1)

    @property
    def betti2(self):
        return len(self.h2)

    def is_closed(self, mask):
        eidx = {e: i for i, e in enumerate(self.edges)}
        for a, b, c in self.triangles:
            if (
                ((mask >> eidx[(a, b)]) & 1)
                ^ ((mask >> eidx[(a, c)]) & 1)
                ^ ((mask >> eidx[(b, c)]) & 1)
            ):
                return False
        return True

    def cup_raw(self, am, bm):
        eidx = {e: i for i, e in enumerate(self.edges)}
        raw = 0
        for ti, (a, b, c) in enumerate(self.triangles):
            if ((am >> eidx[(a, b)]) & 1) and ((bm >> eidx[(b, c)]) & 1):
                raw |= 1 << ti
        return raw

    def h2_coords(self, mask):
        r = reduce_by(mask, self.b2)
        coords = []
        for h in self.h2:
            p = h.bit_length() - 1
            if (r >> p) & 1:
                coords.append(1)
                r ^= h
            else:
                coords.append(0)
        assert r == 0, "H^2 basis failed to span"
        return tuple(coords)


def pairing_matrix(ref):
    """Full H^1 x H^1 -> H^2 pairing matrix from the reference basis."""
    return [
        [ref.h2_coords(ref.cup_raw(ref.h1[i], ref.h1[j])) for j in range(ref.betti1)]
        for i in range(ref.betti1)
    ]


def gf2_rank(rows):
    """Rank of a 0/1 matrix (list of tuples) over GF(2)."""
    piv = echelon(
        [sum(bit << k for k, bit in enumerate(row)) for row in rows if any(row)]
    )
    return len(piv)
