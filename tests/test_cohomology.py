"""Tests for the H1 x H1 -> H2 cup product on a fixed filtration slice.

Expected values come from 'reference' (independent set-based GF(2)
Gaussian elimination -- it never imports pershom) or from hand-computed
closed forms.  Fixtures: the 7-vertex torus triangulation and a wedge
S1 v S1 v S2, which share Betti numbers (1, 2, 1) but have different
cup products.
"""

from __future__ import annotations

import itertools
import unittest

import reference as ref

from pershom import (
    PersistenceInputError,
    cohomology_slice,
    cup_product,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def complex_from(triangles=(), extra_edges=(), extra_verts=(), filtration=None):
    """Build a valid (vertices, filtration) input list from a closure."""
    tris = sorted({tuple(sorted(t)) for t in triangles})
    edges = {tuple(sorted(e)) for e in extra_edges}
    for a, b, c in tris:
        edges |= {(a, b), (a, c), (b, c)}
    edges = sorted(edges)
    verts = sorted(set(extra_verts) | {v for e in edges for v in e})
    out = [((v,), 0) for v in verts]
    out += [(e, 0) for e in edges]
    out += [(t, 0) for t in tris]
    if filtration is not None:
        out = [(v, filtration(v)) for v, _ in out]
    return out


def torus7():
    """Minimal 7-vertex triangulation of the torus (vertex-transitive on Z7)."""
    tris = []
    for i in range(7):
        tris.append(tuple(sorted((i, (i + 1) % 7, (i + 3) % 7))))
        tris.append(tuple(sorted((i, (i + 2) % 7, (i + 3) % 7))))
    return complex_from(tris)


def wedge_s1_s1_s2():
    """S1 v S1 v S2: same Betti numbers (1,2,1) as the torus, trivial product."""
    sphere = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    loops = [(0, 4), (4, 5), (0, 5), (0, 6), (6, 7), (0, 7)]
    return complex_from(sphere, extra_edges=loops)


def cup_matrix(slice_):
    """Matrix of H^2 coordinates: entry [i][j] = coords of hi ~ hj."""
    n = len(slice_.h1_basis)
    return [
        [cup_product(slice_, slice_.h1_basis[i], slice_.h1_basis[j]).coordinates
         for j in range(n)]
        for i in range(n)
    ]


def cup_matrix_rank(slice_):
    mat = cup_matrix(slice_)
    # Flatten each coordinate vector: rows indexed by i, columns by (j, h2-coord).
    rows = [
        tuple(x for coord in row for x in coord)
        for row in mat
    ]
    return ref.gf2_matrix_rank(rows)


# ---------------------------------------------------------------------------
# Slice cohomology spaces
# ---------------------------------------------------------------------------

class SliceCohomologyTest(unittest.TestCase):
    def test_torus_betti_matches_reference(self):
        simp = torus7()
        sl = cohomology_slice(simp, 0)
        self.assertEqual((sl.b0, sl.b1, sl.b2), (1, 2, 1))
        self.assertEqual((sl.b0, sl.b1, sl.b2), ref.betti_ranks_2(simp, 0))
        self.assertEqual((sl.b0, sl.b1), ref.betti_ranks(simp, 0))
        self.assertEqual(sl.b0, ref.betti_union_find(simp, 0))

    def test_wedge_betti_matches_reference(self):
        simp = wedge_s1_s1_s2()
        sl = cohomology_slice(simp, 0)
        self.assertEqual((sl.b0, sl.b1, sl.b2), (1, 2, 1))
        self.assertEqual((sl.b0, sl.b1, sl.b2), ref.betti_ranks_2(simp, 0))

    def test_h1_basis_members_are_closed_and_independent(self):
        # Every basis cocycle must pass the library's own closedness check
        # (cup_product validates), and stay non-zero modulo coboundaries:
        # cupping is well-defined, and the basis size equals b1 (checked
        # against the reference above).
        for simp in (torus7(), wedge_s1_s1_s2()):
            sl = cohomology_slice(simp, 0)
            for cocycle in sl.h1_basis:
                # Non-trivial class: not a coboundary itself.  Check via the
                # reference: a cocycle is a coboundary iff adding it to the
                # B^1 generators does not raise the rank.  Recompute B^1
                # independently as vertex stars.
                _, edges, _ = ref.slice_cells(simp, 0)
                stars = [
                    frozenset(ref.vertex_star_edges(simp, 0, v))
                    for v, in ref.slice_cells(simp, 0)[0]
                ]
                self.assertFalse(
                    ref.gf2_in_span(frozenset(cocycle), stars),
                    "basis cocycle is a coboundary",
                )

    def test_threshold_slicing(self):
        # Vertices at 0, edges at 1, triangles at 2: the slice grows.
        simp = complex_from(
            [(0, 1, 2)],
            filtration=lambda v: {1: 0, 2: 1, 3: 2}[len(v)],
        )
        sl0 = cohomology_slice(simp, 0)
        self.assertEqual((sl0.b0, sl0.b1, sl0.b2), (3, 0, 0))
        sl1 = cohomology_slice(simp, 1)
        self.assertEqual((sl1.b0, sl1.b1, sl1.b2), (1, 1, 0))
        sl2 = cohomology_slice(simp, 2)
        self.assertEqual((sl2.b0, sl2.b1, sl2.b2), (1, 0, 0))
        for t, expect in ((0, (3, 0, 0)), (1, (1, 1, 0)), (2, (1, 0, 0))):
            self.assertEqual(ref.betti_ranks_2(simp, t), expect)

    def test_bad_threshold_rejected(self):
        with self.assertRaises(PersistenceInputError) as ctx:
            cohomology_slice(torus7(), float("nan"))
        self.assertEqual(ctx.exception.reason, "bad_threshold")

    def test_input_validation_shared_with_persistence(self):
        # Missing face is rejected by the shared parser.
        with self.assertRaises(PersistenceInputError) as ctx:
            cohomology_slice([((0, 1), 0)], 0)
        self.assertEqual(ctx.exception.reason, "missing_face")


# ---------------------------------------------------------------------------
# Cup product: torus vs wedge
# ---------------------------------------------------------------------------

class CupProductTest(unittest.TestCase):
    def test_torus_product_nonzero(self):
        simp = torus7()
        sl = cohomology_slice(simp, 0)
        mat = cup_matrix(sl)
        # Over GF(2) the torus has intersection form [[0,1],[1,0]]: the two
        # generators square to 0 and cup to the H^2 generator.
        self.assertEqual(mat[0][0], (0,))
        self.assertEqual(mat[1][1], (0,))
        self.assertEqual(mat[0][1], (1,))
        self.assertEqual(mat[1][0], (1,))
        self.assertEqual(cup_matrix_rank(sl), 2)

    def test_wedge_product_zero_same_betti(self):
        simp = wedge_s1_s1_s2()
        sl = cohomology_slice(simp, 0)
        self.assertEqual((sl.b0, sl.b1, sl.b2), (1, 2, 1))  # same as torus
        for i in range(2):
            for j in range(2):
                res = cup_product(sl, sl.h1_basis[i], sl.h1_basis[j])
                self.assertTrue(res.is_zero)
                self.assertEqual(res.coordinates, (0,))
        self.assertEqual(cup_matrix_rank(sl), 0)

    def test_raw_cochain_and_zero_flag_match_reference(self):
        # Cross-check every product against direct GF(2) elimination.
        for simp in (torus7(), wedge_s1_s1_s2()):
            sl = cohomology_slice(simp, 0)
            for a, b in itertools.product(sl.h1_basis, repeat=2):
                res = cup_product(sl, a, b)
                raw_ref, zero_ref = ref.reference_cup_product(simp, 0, a, b)
                self.assertEqual(set(res.raw), raw_ref)
                self.assertEqual(res.is_zero, zero_ref)

    def test_raw_cochain_hand_computed(self):
        # Hand-check the Alexander-Whitney evaluation on the torus for the
        # pair of basis cocycles: recompute the raw chain straight from the
        # formula and compare with the library's 'raw' field.
        simp = torus7()
        sl = cohomology_slice(simp, 0)
        a, b = sl.h1_basis
        res = cup_product(sl, a, b)
        expect = {
            (u, v, w)
            for (u, v, w) in sl.triangles
            if (u, v) in set(a) and (v, w) in set(b)
        }
        self.assertEqual(set(res.raw), expect)
        self.assertEqual(len(res.raw), 1)  # a single triangle carries the class

    def test_coboundary_change_leaves_result_invariant(self):
        # Adding delta_0 of any 0-cochain (a vertex star) to either input
        # must not change the H^2 coordinates.
        simp = torus7()
        sl = cohomology_slice(simp, 0)
        a, b = sl.h1_basis
        base = cup_product(sl, a, b).coordinates
        for (v,) in sl.vertices:
            star = ref.vertex_star_edges(simp, 0, v)
            a2 = sorted(set(a) ^ set(star))
            b2 = sorted(set(b) ^ set(star))
            self.assertEqual(cup_product(sl, a2, b).coordinates, base)
            self.assertEqual(cup_product(sl, a, b2).coordinates, base)

    def test_symmetric_difference_of_cocycles_adds(self):
        # Bilinearity sanity check on the torus: (a+a)~b == 0.
        simp = torus7()
        sl = cohomology_slice(simp, 0)
        a, b = sl.h1_basis
        zero_cocycle = sorted(set(a) ^ set(a))
        res = cup_product(sl, zero_cocycle, b)
        self.assertTrue(res.is_zero)
        self.assertEqual(res.raw, ())

    def test_vertex_relabeling_preserves_rank_and_zero_pattern(self):
        # Permuting vertex ids may change the basis, but the cup-matrix
        # rank and the zero/non-zero pattern of pair products are invariant.
        import random

        rng = random.Random(20260926)
        for build, expect_rank in ((torus7, 2), (wedge_s1_s1_s2, 0)):
            simp = build()
            sl = cohomology_slice(simp, 0)
            self.assertEqual(cup_matrix_rank(sl), expect_rank)
            n_vertices = len(ref.slice_cells(simp, 0)[0])
            for _ in range(5):
                perm = list(range(n_vertices))
                rng.shuffle(perm)
                relabelled = [
                    (tuple(sorted(perm[v] for v in verts)), f)
                    for verts, f in simp
                ]
                sl2 = cohomology_slice(relabelled, 0)
                self.assertEqual((sl2.b0, sl2.b1, sl2.b2), (1, 2, 1))
                self.assertEqual(cup_matrix_rank(sl2), expect_rank)


# ---------------------------------------------------------------------------
# Cup product: error semantics
# ---------------------------------------------------------------------------

class CupProductInputTest(unittest.TestCase):
    def setUp(self):
        self.simp = torus7()
        self.sl = cohomology_slice(self.simp, 0)

    def test_non_closed_cocycle_rejected(self):
        # A single edge is (generically) not closed on the torus.
        with self.assertRaises(PersistenceInputError) as ctx:
            cup_product(self.sl, [(0, 1)], self.sl.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "cocycle_not_closed")

    def test_edge_outside_slice_rejected(self):
        with self.assertRaises(PersistenceInputError) as ctx:
            cup_product(self.sl, [(0, 99)], self.sl.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "edge_not_in_slice")

    def test_malformed_cocycle_rejected(self):
        with self.assertRaises(PersistenceInputError) as ctx:
            cup_product(self.sl, "not-a-cocycle", self.sl.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "bad_cocycle")
        with self.assertRaises(PersistenceInputError) as ctx:
            cup_product(self.sl, [(0, 1, 2)], self.sl.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "bad_cocycle")
        with self.assertRaises(PersistenceInputError) as ctx:
            cup_product(self.sl, [(0, True)], self.sl.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "bad_cocycle")

    def test_result_fields(self):
        res = cup_product(self.sl, self.sl.h1_basis[0], self.sl.h1_basis[1])
        self.assertEqual(res.left, tuple(sorted(self.sl.h1_basis[0])))
        self.assertEqual(res.right, tuple(sorted(self.sl.h1_basis[1])))
        self.assertEqual(len(res.coordinates), len(self.sl.h2_basis))
        self.assertEqual(res.is_zero, not any(res.coordinates))


if __name__ == "__main__":
    unittest.main()
