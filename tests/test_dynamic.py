"""Tests for the dynamic insert/delete complex (pershom.DynamicComplex).

Expected Betti numbers come from hand-computed sequences (documented inline)
and from the independent rank-nullity reference in ``tests/reference.py``
(``ref.betti_ranks``), never from the library under test.  The library's own
static entry point is only used as a supplementary consistency bridge.
"""

from __future__ import annotations

import random
import unittest

import reference as ref

from pershom import (
    DynamicComplex,
    PersistenceInputError,
    compute_persistence,
)


def mirror_betti(mirror: dict) -> tuple[int, int]:
    """Independent expectation: rank-nullity on the mirrored simplex list."""
    data = [(verts, f) for (dim, verts), f in mirror.items()]
    if not data:
        return (0, 0)
    top = max(f for _, f in data)
    return ref.betti_ranks(data, top)


class TestHandComputedSequence(unittest.TestCase):
    """Ring -> fill -> delete face -> delete edges, checked step by step.

    The expected (beta0, beta1) values below are derived by hand from the
    cell counts and cycle structure of each intermediate complex.
    """

    def test_full_lifecycle(self):
        dyn = DynamicComplex()
        self.assertEqual(dyn.betti(), (0, 0))  # empty complex

        events = [
            ("add", (0,), 0, (1, 0)),
            ("add", (1,), 0, (2, 0)),
            ("add", (2,), 0, (3, 0)),
            ("add", (0, 1), 1, (2, 0)),   # path of one edge
            ("add", (1, 2), 1, (1, 0)),   # path of two edges: connected
            ("add", (0, 2), 1, (1, 1)),   # ring closes -> one H1 class
            ("add", (0, 1, 2), 2, (1, 0)),  # hole filled
            ("remove", (0, 1, 2), None, (1, 1)),  # hole reopens
            ("remove", (0, 2), None, (1, 0)),     # ring breaks into a path
            ("remove", (1, 2), None, (2, 0)),     # one edge + isolated vertex
            ("remove", (0, 1), None, (3, 0)),     # three isolated vertices
            ("remove", (0,), None, (2, 0)),
            ("remove", (1,), None, (1, 0)),
            ("remove", (2,), None, (0, 0)),       # empty again
        ]
        for step, (op, verts, filt, want) in enumerate(events):
            with self.subTest(step=step, op=op, verts=verts):
                if op == "add":
                    dyn.add(verts, filt)
                else:
                    dyn.remove(verts)
                self.assertEqual(dyn.betti(), want)
        self.assertEqual(len(dyn), 0)
        self.assertEqual(dyn.simplices(), [])

    def test_figure_eight_two_holes(self):
        # Two triangle rings sharing vertex 2: beta0=1, beta1=2.
        dyn = DynamicComplex()
        for v in range(5):
            dyn.add((v,), 0)
        for e in ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)):
            dyn.add(e, 1)
        self.assertEqual(dyn.betti(), (1, 2))
        dyn.add((0, 1, 2), 2)   # fill the first ring
        self.assertEqual(dyn.betti(), (1, 1))
        dyn.add((2, 3, 4), 2)   # fill the second ring
        self.assertEqual(dyn.betti(), (1, 0))
        # Deleting the shared vertex is impossible while edges remain.
        with self.assertRaises(PersistenceInputError) as cm:
            dyn.remove((2,))
        self.assertEqual(cm.exception.reason, "has_cofaces")
        self.assertEqual(dyn.betti(), (1, 0))  # unchanged

    def test_two_components_merge_and_split(self):
        # Two separate edges, then a connecting edge merges the components.
        dyn = DynamicComplex()
        for v in range(4):
            dyn.add((v,), 0)
        dyn.add((0, 1), 0)
        dyn.add((2, 3), 0)
        self.assertEqual(dyn.betti(), (2, 0))
        dyn.add((1, 2), 0)   # zero-length filtration ties are fine here
        self.assertEqual(dyn.betti(), (1, 0))
        dyn.remove((1, 2))
        self.assertEqual(dyn.betti(), (2, 0))


class TestAtomicRejections(unittest.TestCase):
    """Every rejected update must leave the complex bit-for-bit unchanged."""

    def setUp(self):
        # A filled triangle: 3 vertices @0, 3 edges @1, 1 triangle @2.
        self.dyn = DynamicComplex()
        for v in range(3):
            self.dyn.add((v,), 0)
        for e in ((0, 1), (0, 2), (1, 2)):
            self.dyn.add(e, 1)
        self.dyn.add((0, 1, 2), 2)
        self.before = self.dyn.simplices()

    def assertRejectedUnchanged(self, reason, fn, *args):
        betti_before = self.dyn.betti()
        with self.assertRaises(PersistenceInputError) as cm:
            fn(*args)
        self.assertEqual(cm.exception.reason, reason)
        self.assertEqual(self.dyn.simplices(), self.before)
        self.assertEqual(self.dyn.betti(), betti_before)

    def test_remove_edge_with_triangle_coface(self):
        self.assertRejectedUnchanged("has_cofaces", self.dyn.remove, (0, 1))

    def test_remove_vertex_with_edge_cofaces(self):
        self.assertRejectedUnchanged("has_cofaces", self.dyn.remove, (0,))

    def test_remove_absent_simplex(self):
        self.assertRejectedUnchanged("missing_simplex", self.dyn.remove, (9,))
        self.assertRejectedUnchanged("missing_simplex", self.dyn.remove, (0, 3))
        self.assertRejectedUnchanged(
            "missing_simplex", self.dyn.remove, (0, 1, 3)
        )

    def test_add_duplicate(self):
        self.assertRejectedUnchanged("duplicate_simplex", self.dyn.add, (0,), 5)
        self.assertRejectedUnchanged(
            "duplicate_simplex", self.dyn.add, (0, 1), 5
        )
        self.assertRejectedUnchanged(
            "duplicate_simplex", self.dyn.add, (0, 1, 2), 5
        )

    def test_add_with_missing_face(self):
        self.assertRejectedUnchanged("missing_face", self.dyn.add, (0, 3), 1)
        self.assertRejectedUnchanged(
            "missing_face", self.dyn.add, (0, 1, 3), 2
        )

    def test_add_face_filtration_order(self):
        # Vertex 3 exists at filtration 9; an edge onto it at 1 is illegal.
        self.dyn.add((3,), 9)
        self.before = self.dyn.simplices()
        self.assertRejectedUnchanged(
            "face_filtration_order", self.dyn.add, (0, 3), 1
        )

    def test_malformed_vertices_and_filtration(self):
        self.assertRejectedUnchanged(
            "unsorted_vertices", self.dyn.add, (2, 0), 1
        )
        self.assertRejectedUnchanged("bad_vertex_id", self.dyn.add, (-1,), 0)
        self.assertRejectedUnchanged("bad_vertex_id", self.dyn.add, (True,), 0)
        self.assertRejectedUnchanged(
            "duplicate_vertex_in_simplex", self.dyn.add, (1, 1), 0
        )
        self.assertRejectedUnchanged(
            "bad_dimension", self.dyn.add, (0, 1, 2, 3), 0
        )
        self.assertRejectedUnchanged(
            "bad_filtration_type", self.dyn.add, (3,), "1"
        )
        self.assertRejectedUnchanged(
            "bad_filtration_type", self.dyn.add, (3,), True
        )
        self.assertRejectedUnchanged(
            "non_finite_filtration", self.dyn.add, (3,), float("nan")
        )
        self.assertRejectedUnchanged(
            "non_finite_filtration", self.dyn.add, (3,), float("inf")
        )
        # remove() validates the vertex format too
        self.assertRejectedUnchanged(
            "unsorted_vertices", self.dyn.remove, (1, 0)
        )
        self.assertRejectedUnchanged("bad_vertices", self.dyn.remove, "ab")

    def test_rejection_does_not_consume_capacity(self):
        # A rejected add must not count towards the 300-simplex bound.
        dyn = DynamicComplex(((i,), 0) for i in range(300))
        with self.assertRaises(PersistenceInputError) as cm:
            dyn.add((300,), 0)
        self.assertEqual(cm.exception.reason, "too_many_simplices")
        self.assertEqual(len(dyn), 300)
        # After a real removal there is room again.
        dyn.remove((0,))
        dyn.add((300,), 0)
        self.assertEqual(len(dyn), 300)
        self.assertEqual(dyn.betti(), (300, 0))


class TestConstructorAndBridge(unittest.TestCase):
    def test_constructor_validates_like_static_entry(self):
        with self.assertRaises(PersistenceInputError) as cm:
            DynamicComplex([((0,), 0), ((0, 1), 1)])  # missing vertex 1
        self.assertEqual(cm.exception.reason, "missing_face")
        with self.assertRaises(PersistenceInputError):
            DynamicComplex([((0,), 0), ((0,), 1)])  # duplicate
        with self.assertRaises(PersistenceInputError):
            DynamicComplex(42)  # not an iterable of pairs

    def test_constructor_accepts_valid_complex(self):
        dyn = DynamicComplex(ref.triangle_ring_example())
        self.assertEqual(len(dyn), 7)
        self.assertEqual(dyn.betti(), (1, 0))  # filled triangle
        self.assertIn((0, 2), dyn)
        self.assertNotIn((5,), dyn)

    def test_empty_constructor_then_build(self):
        dyn = DynamicComplex()
        self.assertEqual(len(dyn), 0)
        dyn.add((0,), 0)
        self.assertEqual(dyn.betti(), (1, 0))

    def test_persistence_bridge_matches_static_entry_point(self):
        dyn = DynamicComplex()
        for v in range(3):
            dyn.add((v,), 0)
        for e in ((0, 1), (0, 2), (1, 2)):
            dyn.add(e, 1)
        dyn.add((0, 1, 2), 2)
        # The snapshot feeds the original static entry point unchanged.
        via_dyn = dyn.persistence()
        via_static = compute_persistence(dyn.simplices())
        self.assertEqual(
            [(b.dimension, b.birth, b.death) for b in via_dyn.bars],
            [(b.dimension, b.birth, b.death) for b in via_static.bars],
        )
        # Hand-computed: one H1 bar [1, 2), two H0 bars [0, 1), one essential.
        self.assertEqual(len(via_dyn.h1_bars), 1)
        self.assertEqual((via_dyn.h1_bars[0].birth, via_dyn.h1_bars[0].death),
                         (1, 2))
        # Dynamic betti agrees with the static Betti query at the top value.
        self.assertEqual(dyn.betti(), via_static.betti_at(2))

    def test_persistence_on_empty_complex_raises_empty_input(self):
        with self.assertRaises(PersistenceInputError) as cm:
            DynamicComplex().persistence()
        self.assertEqual(cm.exception.reason, "empty_input")

    def test_readd_after_remove_with_new_filtration(self):
        dyn = DynamicComplex(ref.triangle_ring_example())
        dyn.remove((0, 1, 2))
        self.assertEqual(dyn.betti(), (1, 1))
        dyn.add((0, 1, 2), 7)  # refill at a different filtration value
        self.assertEqual(dyn.betti(), (1, 0))
        bar = dyn.persistence().h1_bars[0]
        self.assertEqual((bar.birth, bar.death), (1, 7))


class TestRandomEventCrossCheck(unittest.TestCase):
    """Deterministic random event streams vs. the independent reference.

    Every generated event (valid or invalid) is applied to the dynamic
    complex; the expected outcome is decided from a plain-dict mirror of the
    complex, and after every step the Betti numbers are compared against
    ``ref.betti_ranks`` (rank-nullity, independent of the library).
    """

    def test_random_events(self):
        rng = random.Random(20260924)
        nverts = 6
        universe = (
            [(v,) for v in range(nverts)]
            + [(a, b) for a in range(nverts) for b in range(a + 1, nverts)]
            + [
                (a, b, c)
                for a in range(nverts)
                for b in range(a + 1, nverts)
                for c in range(b + 1, nverts)
            ]
        )
        dyn = DynamicComplex()
        mirror: dict[tuple[int, tuple[int, ...]], int] = {}

        def faces_of(verts):
            if len(verts) == 2:
                return [(verts[0],), (verts[1],)]
            a, b, c = verts
            return [(b, c), (a, c), (a, b)]

        def has_coface(verts):
            d = len(verts) - 1
            target = set(verts)
            return any(
                dd == d + 1 and target.issubset(vv)
                for (dd, vv) in mirror
            )

        for step in range(400):
            verts = universe[rng.randrange(len(universe))]
            dim = len(verts) - 1
            key = (dim, verts)
            before = dyn.simplices()
            if key in mirror:
                # attempt a removal
                if has_coface(verts):
                    with self.assertRaises(PersistenceInputError) as cm:
                        dyn.remove(verts)
                    self.assertEqual(cm.exception.reason, "has_cofaces")
                    self.assertEqual(dyn.simplices(), before)
                else:
                    dyn.remove(verts)
                    del mirror[key]
            else:
                faces = faces_of(verts) if dim >= 1 else []
                if any((dim - 1, f) not in mirror for f in faces):
                    # Pick a filtration that satisfies every *present* face,
                    # so the missing face is the (only) reason to reject.
                    f_ok = max(
                        (mirror[(dim - 1, f)] for f in faces
                         if (dim - 1, f) in mirror),
                        default=0,
                    )
                    with self.assertRaises(PersistenceInputError) as cm:
                        dyn.add(verts, f_ok)
                    self.assertEqual(cm.exception.reason, "missing_face")
                    self.assertEqual(dyn.simplices(), before)
                else:
                    lo = max((mirror[(dim - 1, f)] for f in faces), default=0)
                    f = rng.randint(lo, lo + 2)
                    dyn.add(verts, f)
                    mirror[key] = f
            with self.subTest(step=step, size=len(mirror)):
                self.assertEqual(dyn.betti(), mirror_betti(mirror))
        # Supplementary: the static entry point agrees on the final state.
        if mirror:
            top = max(mirror.values())
            self.assertEqual(
                dyn.betti(), dyn.persistence().betti_at(top)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
