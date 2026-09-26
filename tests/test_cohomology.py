"""Tests for the H^1 x H^1 -> H^2 cup product on a fixed filtration slice.

Expected values come from reference_cohomology (an independent GF(2)
Gaussian-elimination reimplementation that never imports pershom) or from
hand-computed examples (3x3 grid torus; wedge of two circles and a
2-sphere, which has the same Betti numbers but a trivial product).
"""

from __future__ import annotations

import itertools
import random
import unittest

import reference_cohomology as ref

from pershom import (
    CertificateError,
    PersistenceInputError,
    check_cup_product,
    compute_cohomology,
    compute_persistence,
)


def torus_complex(filtration="by_dimension"):
    """3x3 grid triangulation of the torus: 9 vertices, 27 edges, 18 tris."""
    def vid(i, j):
        return i * 3 + j

    f = {"by_dimension": (0, 1, 2), "flat": (0, 0, 0)}[filtration]
    simps = [((vid(i, j),), f[0]) for i in range(3) for j in range(3)]
    edges = set()
    tris = set()
    for i in range(3):
        for j in range(3):
            a = vid(i, j)
            b = vid((i + 1) % 3, j)
            c = vid(i, (j + 1) % 3)
            d = vid((i + 1) % 3, (j + 1) % 3)
            for t in ((a, b, d), (a, c, d)):
                t = tuple(sorted(t))
                tris.add(t)
                edges.update(
                    {(t[0], t[1]), (t[0], t[2]), (t[1], t[2])}
                )
    simps += [(e, f[1]) for e in sorted(edges)]
    simps += [(t, f[2]) for t in sorted(tris)]
    return simps


def wedge_complex():
    """S^1 v S^1 v S^2 at one vertex: beta = (1, 2, 1), trivial cup product.

    Two triangle boundaries (figure eight) on vertices 0,1,2 and 0,3,4,
    plus a hollow tetrahedron on 0,5,6,7.
    """
    circles = [(0, 1, 2), (0, 3, 4)]
    sphere_tris = [(0, 5, 6), (0, 5, 7), (0, 6, 7), (5, 6, 7)]
    verts = sorted({v for t in circles + sphere_tris for v in t})
    edges = set()
    for t in circles + sphere_tris:
        edges.update({(t[0], t[1]), (t[0], t[2]), (t[1], t[2])})
    simps = [((v,), 0) for v in verts]
    simps += [(e, 0) for e in sorted(edges)]
    simps += [(t, 0) for t in sorted(sphere_tris)]
    return simps


def split_complex(simps):
    verts, edges, tris = [], [], []
    for v, _ in simps:
        [verts, edges, tris][len(v) - 1].append(v)
    return verts, edges, tris


def mask_of(indices):
    m = 0
    for i in indices:
        m ^= 1 << i
    return m


class TorusCupProduct(unittest.TestCase):
    def setUp(self):
        self.simps = torus_complex()
        self.cs = compute_cohomology(self.simps)
        self.ref = ref.ReferenceSlice(*split_complex(self.simps))

    def test_betti_numbers(self):
        self.assertEqual((self.cs.betti1, self.cs.betti2), (2, 1))
        self.assertEqual((self.ref.betti1, self.ref.betti2), (2, 1))
        # agrees with the persistent-homology Betti numbers of the slice
        p = compute_persistence(self.simps)
        self.assertEqual(p.betti_at(2), (1, 2))

    def test_product_is_nonzero(self):
        table = [
            [self.cs.cup_product(i, j).h2_coordinates for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(ref.gf2_rank([c for row in table for c in row]), 1)
        self.assertTrue(any(c == (1,) for row in table for c in row))

    def test_every_result_verifies_independently(self):
        for i, j in itertools.product(range(2), repeat=2):
            cp = self.cs.cup_product(i, j)
            self.assertTrue(check_cup_product(self.cs, cp))
            self.assertEqual(cp.is_zero, cp.h2_coordinates == (0,))

    def test_matches_gf2_reference_exactly(self):
        # Feed the library's own H^1 basis cocycles to the independent
        # reference and compare raw cochains and H^2 coordinates exactly.
        for i, j in itertools.product(range(2), repeat=2):
            cp = self.cs.cup_product(i, j)
            am = mask_of(cp.cocycle_a)
            bm = mask_of(cp.cocycle_b)
            self.assertTrue(self.ref.is_closed(am))
            self.assertTrue(self.ref.is_closed(bm))
            self.assertEqual(mask_of(cp.raw_product), self.ref.cup_raw(am, bm))
            self.assertEqual(cp.h2_coordinates, self.ref.h2_coords(mask_of(cp.raw_product)))

    def test_coboundary_invariance(self):
        # Adding any 1-coboundary to either factor must not change the class.
        for b in self.cs.b1_basis:
            bm = mask_of(b)
            for i, j in itertools.product(range(2), repeat=2):
                ai = mask_of(self.cs.h1_basis[i])
                bj = mask_of(self.cs.h1_basis[j])
                want = self.cs.cup_product(i, j).h2_coordinates
                self.assertEqual(
                    self.cs.cup_product_of(ai ^ bm, bj).h2_coordinates, want
                )
                self.assertEqual(
                    self.cs.cup_product_of(ai, bj ^ bm).h2_coordinates, want
                )
                self.assertEqual(
                    self.cs.cup_product_of(ai ^ bm, bj ^ bm).h2_coordinates, want
                )
        self.assertEqual(self.cs.cup_product(0, 1).h2_coordinates, (1,))

    def test_raw_product_changes_but_class_is_stable(self):
        # The raw 2-cochain may move by a coboundary while H^2 coordinates
        # stay fixed; verify both directions of that statement honestly.
        b0 = mask_of(self.cs.b1_basis[0])
        shifted = mask_of(self.cs.h1_basis[0]) ^ b0
        cp = self.cs.cup_product_of(shifted, self.cs.h1_basis[1])
        ref_cp = self.cs.cup_product(0, 1)
        self.assertEqual(cp.h2_coordinates, ref_cp.h2_coordinates)
        diff = mask_of(cp.raw_product) ^ mask_of(ref_cp.raw_product)
        self.assertEqual(self.cs.reduce_h2(diff), (0,))  # differs by coboundary

    def test_vertex_relabeling_preserves_rank(self):
        rng = random.Random(7)
        for _ in range(5):
            perm = list(range(9))
            rng.shuffle(perm)
            relabeled = [
                (tuple(sorted(perm[v] for v in verts)), f)
                for verts, f in self.simps
            ]
            cs2 = compute_cohomology(relabeled)
            self.assertEqual((cs2.betti1, cs2.betti2), (2, 1))
            table = [
                [cs2.cup_product(i, j).h2_coordinates for j in range(2)]
                for i in range(2)
            ]
            self.assertEqual(ref.gf2_rank([c for row in table for c in row]), 1)


class WedgeCupProduct(unittest.TestCase):
    """Same Betti numbers as the torus, but the product is identically zero."""

    def setUp(self):
        self.simps = wedge_complex()
        self.cs = compute_cohomology(self.simps)
        self.ref = ref.ReferenceSlice(*split_complex(self.simps))

    def test_same_betti_numbers_as_torus(self):
        self.assertEqual((self.cs.betti1, self.cs.betti2), (2, 1))
        self.assertEqual((self.ref.betti1, self.ref.betti2), (2, 1))

    def test_product_is_identically_zero(self):
        for i, j in itertools.product(range(2), repeat=2):
            cp = self.cs.cup_product(i, j)
            self.assertTrue(check_cup_product(self.cs, cp))
            self.assertTrue(cp.is_zero)
            self.assertEqual(cp.h2_coordinates, (0,))
            # even the raw cochain is empty here: the circle edges never
            # appear in any triangle
            self.assertEqual(cp.raw_product, ())

    def test_reference_agrees(self):
        self.assertEqual(
            ref.pairing_matrix(self.ref), [[(0,), (0,)], [(0,), (0,)]]
        )


class FiltrationSlices(unittest.TestCase):
    def test_slice_grows_with_threshold(self):
        simps = torus_complex()
        early = compute_cohomology(simps, threshold=1)
        self.assertEqual(len(early.triangles), 0)
        self.assertEqual(early.betti2, 0)
        # 27 edges, 9 vertices, one component: beta1 = 27 - 9 + 1
        self.assertEqual(early.betti1, 19)
        for i, j in [(0, 0), (3, 17)]:
            cp = early.cup_product(i, j)
            self.assertTrue(cp.is_zero)
            self.assertEqual(cp.h2_coordinates, ())
            self.assertTrue(check_cup_product(early, cp))
        full = compute_cohomology(simps, threshold=2)
        self.assertEqual((full.betti1, full.betti2), (2, 1))

    def test_flat_filtration_slice_equals_unthresholded(self):
        simps = torus_complex(filtration="flat")
        a = compute_cohomology(simps, threshold=0)
        b = compute_cohomology(simps)
        self.assertEqual(a.edges, b.edges)
        self.assertEqual(a.h1_basis, b.h1_basis)
        self.assertEqual(
            a.cup_product(0, 1).h2_coordinates, b.cup_product(0, 1).h2_coordinates
        )


class ValidationAndErrors(unittest.TestCase):
    def setUp(self):
        self.cs = compute_cohomology(torus_complex())

    def test_input_contract_shared_with_persistence(self):
        with self.assertRaises(PersistenceInputError):
            compute_cohomology([((1, 0), 0)])  # unsorted vertices
        with self.assertRaises(PersistenceInputError):
            compute_cohomology([((0, 1), 0)])  # missing vertex faces
        with self.assertRaises(PersistenceInputError):
            compute_cohomology([])

    def test_bad_threshold(self):
        with self.assertRaises(PersistenceInputError):
            compute_cohomology(torus_complex(), threshold=float("nan"))

    def test_bad_basis_index(self):
        for bad in (-1, 2, True, 1.5):
            with self.assertRaises(PersistenceInputError):
                self.cs.cup_product(bad, 0)

    def test_non_cocycle_rejected(self):
        open_chain = (0,)  # a single edge is (generically) not closed
        edge = self.cs.edges[0]
        # ensure the chosen edge actually borders a triangle (it does: torus)
        with self.assertRaises(PersistenceInputError) as ctx:
            self.cs.cup_product_of(open_chain, self.cs.h1_basis[0])
        self.assertEqual(ctx.exception.reason, "not_a_cocycle")
        self.assertIn(edge, [self.cs.edges[i] for i in self.cs.h1_basis[0]] + [edge])

    def test_bad_cochain_type(self):
        with self.assertRaises(PersistenceInputError):
            self.cs.cup_product_of("not a cochain", self.cs.h1_basis[0])
        with self.assertRaises(PersistenceInputError):
            self.cs.cup_product_of((10**6,), self.cs.h1_basis[0])

    def test_tampered_result_fails_verification(self):
        cp = self.cs.cup_product(0, 1)
        from dataclasses import replace

        fake = replace(cp, h2_coordinates=(0,), is_zero=True)
        with self.assertRaises(CertificateError):
            check_cup_product(self.cs, fake)
        fake2 = replace(cp, raw_product=())
        with self.assertRaises(CertificateError):
            check_cup_product(self.cs, fake2)
        fake3 = replace(cp, cocycle_a=(0,))
        with self.assertRaises(CertificateError):
            check_cup_product(self.cs, fake3)


if __name__ == "__main__":
    unittest.main()
