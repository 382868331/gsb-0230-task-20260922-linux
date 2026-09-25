"""End-to-end tests for the pershom library.

Expected values come from ``reference`` (a fully independent rank-based
reimplementation: Gaussian elimination + rank invariant + union-find) or
from closed-form hand calculations -- never from the library under test.
The library's own reduction is only consumed through its public API.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

import reference as ref

from pershom import (
    INF,
    CertificateError,
    PersistenceInputError,
    SliceCycle,
    check_certificate,
    compute_persistence,
)
from pershom.persistence import _build_boundary, _reduce


def cells_table(result):
    """Public cell data as plain (vertices, filtration) tuples."""
    return [(c.vertices, c.filtration) for c in result.cells]


def positive_bars(result, dim):
    out = []
    for b in result.bars:
        if b.dimension != dim:
            continue
        out.append((b.birth, None if b.death_index is None else b.death))
    return sorted(out, key=lambda x: (x[0], x[1] is None, x[1]))


class TestAnalyticTriangleRing(unittest.TestCase):
    """The headline analytic example: boundary triangle filled later."""

    def setUp(self):
        self.data = ref.triangle_ring_example()
        self.res = compute_persistence(iter(self.data))  # generator input
        self.r = self.res

    def test_exact_bars(self):
        h0 = positive_bars(self.r, 0)
        self.assertEqual(
            h0, [(0, 1), (0, 1), (0, None)]
        )
        h1 = positive_bars(self.r, 1)
        self.assertEqual(h1, [(1, 2)])

    def test_betti_per_threshold_closed_form(self):
        # Hand-computed: t=0 -> (3,0); t=1 -> (1,1); t=2 -> (1,0)
        self.assertEqual(self.r.betti_at(0), (3, 0))
        self.assertEqual(self.r.betti_at(0.5), (3, 0))
        self.assertEqual(self.r.betti_at(1), (1, 1))
        self.assertEqual(self.r.betti_at(1.999), (1, 1))
        self.assertEqual(self.r.betti_at(2), (1, 0))
        self.assertEqual(self.r.betti_at(99), (1, 0))

    def test_birth_and_death_chains(self):
        (bar,) = self.r.h1_bars
        self.assertEqual(
            sorted(self.r.chain_vertices(bar.birth_chain)),
            [(0, 1), (0, 2), (1, 2)],
        )
        self.assertEqual(
            self.r.chain_vertices(bar.death_chain), [(0, 1, 2)]
        )
        self.assertFalse(bar.zero_length)

    def test_independent_certificate_verification(self):
        table = cells_table(self.r)
        for bar in self.r.h1_bars:
            ok, reason = ref.verify_h1_certificate_cells(
                table,
                bar.birth_index,
                bar.death_index,
                bar.birth_chain,
                bar.death_chain,
            )
            self.assertTrue(ok, reason)
            self.assertTrue(check_certificate(self.r, bar))

    def test_independent_boundary_faces(self):
        # Every higher cell's faces exist, are earlier in filtration order
        # and carry filtration <= the cell (independent face enumeration).
        cells = self.r.cells
        by_verts = {(c.dimension, c.vertices): c for c in cells}
        for c in cells:
            if c.dimension == 1:
                faces = [((0, (c.vertices[0],))), ((0, (c.vertices[1],)))]
            elif c.dimension == 2:
                a, b, cc = c.vertices
                faces = [
                    (1, (b, cc)), (1, (a, cc)), (1, (a, b))
                ]
            else:
                continue
            for fk in faces:
                face = by_verts[fk]
                self.assertLess(face.index, c.index)
                self.assertLessEqual(face.filtration, c.filtration)


class TestIndependentCrossChecks(unittest.TestCase):
    """Library barcode/Betti vs. the independent rank-invariant reference."""

    EXAMPLES = {
        "ring": ref.triangle_ring_example,
        "two_rings": ref.two_rings_one_filled,
        "grid_with_hole": ref.filled_grid_with_hole,
    }

    def test_betti_numbers_at_every_threshold(self):
        for name, factory in self.EXAMPLES.items():
            with self.subTest(example=name):
                data = factory()
                res = compute_persistence(data)
                values = sorted({f for _, f in data})
                # sample at each value and halfway between values
                samples = list(values)
                for a, b in zip(values, values[1:]):
                    samples.append((a + b) / 2)
                for t in samples:
                    with self.subTest(t=t):
                        got = res.betti_at(t)
                        want = ref.betti_ranks(data, t)
                        self.assertEqual(got, want, (name, t))
                        self.assertEqual(
                            got[0], ref.betti_union_find(data, t)
                        )
                        self.assertGreaterEqual(got[1], 0)

    def test_betti_profile_default_points(self):
        data = ref.two_rings_one_filled()
        res = compute_persistence(data)
        profile = dict(res.betti_profile())
        self.assertEqual(profile[0], (6, 0))
        self.assertEqual(profile[1], (2, 2))
        self.assertEqual(profile[2], (2, 1))

    def test_positive_barcode_matches_rank_invariant(self):
        for name, factory in self.EXAMPLES.items():
            with self.subTest(example=name):
                data = factory()
                res = compute_persistence(data)
                for dim in (0, 1):
                    got = positive_bars(res, dim)
                    want = sorted(
                        ref.positive_barcode(data, dim),
                        key=lambda x: (x[0], x[1] is None, x[1]),
                    )
                    self.assertEqual(got, want, (name, dim))

    def test_every_finite_h1_certificate_verifies_independently(self):
        data = ref.filled_grid_with_hole()
        res = compute_persistence(data)
        table = cells_table(res)
        finite = [b for b in res.h1_bars if b.finite]
        # 5 filled squares: 5 bars [1,3) and 5 bars [2,3), one essential.
        self.assertEqual(len(finite), 10)
        for bar in finite:
            ok, reason = ref.verify_h1_certificate_cells(
                table,
                bar.birth_index,
                bar.death_index,
                bar.birth_chain,
                bar.death_chain,
            )
            self.assertTrue(ok, reason)


class TestZeroLengthAndTies(unittest.TestCase):
    def test_all_same_filtration_full_triangle(self):
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), 0), ((0, 2), 0), ((1, 2), 0),
            ((0, 1, 2), 0),
        ]
        res = compute_persistence(data)
        # Zero-length bars are retained, one per simplex not paired by a
        # strictly-later simplex: H0 x2, H1 x1; H0 x1 essential.
        flags = sorted((b.dimension, b.zero_length, b.finite) for b in res.bars)
        self.assertEqual(
            flags,
            [
                (0, False, False),
                (0, True, True),
                (0, True, True),
                (1, True, True),
            ],
        )
        # Half-open interval: at t=0 the subcomplex is the full triangle.
        self.assertEqual(res.betti_at(0), (1, 0))

    def test_ring_and_fill_same_value_zero_length_h1(self):
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), 1), ((0, 2), 1), ((1, 2), 1),
            ((0, 1, 2), 1),
        ]
        res = compute_persistence(data)
        h1 = [b for b in res.h1_bars]
        self.assertEqual(len(h1), 1)
        self.assertTrue(h1[0].zero_length)
        self.assertEqual(h1[0].birth, h1[0].death)
        # zero-length bar contributes to no threshold slice
        self.assertEqual(res.betti_at(1), (1, 0))

    def test_tie_breaks_by_dimension_then_vertices(self):
        data = [
            ((2,), 5), ((0,), 5), ((1,), 5),
            ((1, 2), 5), ((0, 2), 5), ((0, 1), 5),
            ((0, 1, 2), 5),
        ]  # deliberately shuffled input
        res = compute_persistence(data)
        ordered = [(c.dimension, c.vertices) for c in res.cells]
        self.assertEqual(
            ordered,
            [
                (0, (0,)), (0, (1,)), (0, (2,)),
                (1, (0, 1)), (1, (0, 2)), (1, (1, 2)),
                (2, (0, 1, 2)),
            ],
        )


class TestEssentialClassesAndComponents(unittest.TestCase):
    def test_bare_square_ring_is_essential_h1(self):
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0), ((3,), 0),
            ((0, 1), 1), ((1, 2), 1), ((2, 3), 1), ((0, 3), 1),
        ]
        res = compute_persistence(data)
        self.assertEqual(len(res.h1_bars), 1)
        bar = res.h1_bars[0]
        self.assertIs(bar.death, INF)
        self.assertIsNone(bar.death_index)
        self.assertIsNone(bar.birth_chain)
        self.assertIsNone(bar.death_chain)
        self.assertEqual(res.betti_at(1), (1, 1))
        self.assertEqual(res.betti_at(10**9), (1, 1))

    def test_two_separated_components(self):
        data = ref.two_rings_one_filled()
        res = compute_persistence(data)
        essential_h0 = [
            b for b in res.h0_bars if b.death_index is None
        ]
        self.assertEqual(len(essential_h0), 2)
        self.assertEqual(res.betti_at(2), (2, 1))

    def test_isolated_vertices(self):
        res = compute_persistence([((0,), 0), ((7,), 1), ((9,), 2)])
        self.assertEqual(len(res.h0_bars), 3)
        self.assertTrue(all(b.death_index is None for b in res.h0_bars))
        self.assertEqual(res.betti_at(2), (3, 0))


class TestCertificateTampering(unittest.TestCase):
    def setUp(self):
        self.res = compute_persistence(ref.filled_grid_with_hole())
        self.bar = next(b for b in self.res.h1_bars if b.finite)

    def _fails(self, bar):
        with self.assertRaises(CertificateError):
            check_certificate(self.res, bar)

    def test_drop_edge_from_birth_chain(self):
        chain = list(self.bar.birth_chain)
        self._fails(replace(self.bar, birth_chain=tuple(chain[:-1])))

    def test_add_unrelated_edge_to_birth_chain(self):
        used = set(self.bar.birth_chain)
        extra = next(
            c.index for c in self.res.cells
            if c.dimension == 1 and c.index not in used
        )
        self._fails(
            replace(self.bar, birth_chain=self.bar.birth_chain + (extra,))
        )

    def test_swap_birth_onto_wrong_edge(self):
        other = next(
            i for i in self.bar.birth_chain if i != self.bar.birth_index
        )
        self._fails(replace(self.bar, birth_index=other))

    def test_drop_triangle_from_death_chain(self):
        chain = tuple(i for i in self.bar.death_chain if i != self.bar.death_index)
        self._fails(replace(self.bar, death_chain=chain))

    def test_extra_triangle_changes_boundary(self):
        used = set(self.bar.death_chain)
        extra = next(
            c.index for c in self.res.cells
            if c.dimension == 2 and c.index not in used
        )
        self._fails(
            replace(self.bar, death_chain=self.bar.death_chain + (extra,))
        )

    def test_wrong_flag_fails(self):
        self._fails(replace(self.bar, zero_length=not self.bar.zero_length))


class TestErrorSemantics(unittest.TestCase):
    def assertReject(self, data, reason=None):
        with self.assertRaises(PersistenceInputError) as cm:
            compute_persistence(data)
        if reason is not None:
            self.assertEqual(cm.exception.reason, reason)
        return cm.exception

    def test_nan_infinity_rejected(self):
        self.assertReject([((0,), float("nan"))], "non_finite_filtration")
        self.assertReject([((0,), float("inf"))], "non_finite_filtration")
        self.assertReject([((0,), -float("inf"))], "non_finite_filtration")

    def test_bool_rejected(self):
        self.assertReject([((0,), True)], "bad_filtration_type")
        self.assertReject([((0,), False)], "bad_filtration_type")
        self.assertReject([((True,), 0)], "bad_vertex_id")

    def test_string_filtration_rejected(self):
        self.assertReject([((0,), "1")], "bad_filtration_type")

    def test_missing_vertex_face(self):
        err = self.assertReject(
            [((0,), 0), ((0, 1), 1)], "missing_face"
        )
        self.assertEqual(err.index, 1)

    def test_missing_edge_face_of_triangle(self):
        err = self.assertReject(
            [
                ((0,), 0), ((1,), 0), ((2,), 0),
                ((0, 1), 1), ((1, 2), 1),
                ((0, 1, 2), 2),
            ],
            "missing_face",
        )
        self.assertEqual(err.index, 5)

    def test_face_filtration_greater_than_simplex(self):
        err = self.assertReject(
            [((0,), 1), ((1,), 0), ((0, 1), 0)],
            "face_filtration_order",
        )
        self.assertEqual(err.index, 2)

    def test_duplicate_vertex_within_simplex(self):
        self.assertReject([((0, 0), 0)], "duplicate_vertex_in_simplex")
        self.assertReject(
            [((0,), 0), ((1,), 0), ((2,), 0),
             ((0, 1), 0), ((1, 2), 0), ((0, 2), 0),
             ((0, 0, 1), 0)],
            "duplicate_vertex_in_simplex",
        )

    def test_duplicate_simplex_rejected(self):
        err = self.assertReject(
            [((0,), 0), ((0,), 1)], "duplicate_simplex"
        )
        self.assertEqual(err.index, 1)

    def test_unsorted_vertices_rejected(self):
        self.assertReject(
            [((0,), 0), ((1,), 0), ((1, 0), 0)], "unsorted_vertices"
        )
        self.assertReject(
            [((0,), 0), ((1,), 0), ((2,), 0),
             ((0, 1), 0), ((0, 2), 0), ((1, 2), 0),
             ((0, 2, 1), 0)],
            "unsorted_vertices",
        )

    def test_bad_dimensions(self):
        self.assertReject([((), 0)], "bad_dimension")
        self.assertReject([((0, 1, 2, 3), 0)], "bad_dimension")

    def test_negative_vertex(self):
        self.assertReject([((-1,), 0)], "bad_vertex_id")

    def test_bad_shapes(self):
        self.assertReject(42, "bad_input")
        self.assertReject([42], "bad_simplex")
        self.assertReject([((0,), 0, 1)], "bad_simplex")
        self.assertReject([("ab", 0)], "bad_vertices")
        self.assertReject([((0,),)], "bad_simplex")
        self.assertReject([("x", 0)], "bad_vertices")
        self.assertReject([], "empty_input")

    def test_too_many_simplices(self):
        err = self.assertReject(
            [((i,), 0) for i in range(301)], "too_many_simplices"
        )
        self.assertEqual(err.index, 300)

    def test_limit_boundary_accepted(self):
        # Exactly 300 simplices: one filled triangle (7 cells) plus 293
        # isolated vertices.
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), 1), ((0, 2), 1), ((1, 2), 1),
            ((0, 1, 2), 2),
        ]
        nxt = 100
        while len(data) < 300:
            data.append(((nxt,), 0))
            nxt += 1
        res = compute_persistence(data)
        self.assertEqual(len(res.cells), 300)

    def test_bad_threshold(self):
        res = compute_persistence([((0,), 0)])
        for bad in (float("nan"), float("inf"), "1", True):
            with self.assertRaises(PersistenceInputError):
                res.betti_at(bad)

    def test_error_is_value_error_family(self):
        # Callers can catch ValueError broadly.
        self.assertTrue(issubclass(PersistenceInputError, ValueError))

    def test_failure_is_atomic_input_unchanged(self):
        data = [
            ((0,), 0), ((1,), 0),
            ((0, 1), 0),
            ((0, 1, 2), 1),  # missing vertex 2 and edges
        ]
        snapshot = [tuple([tuple(v), f]) for v, f in data]
        with self.assertRaises(PersistenceInputError):
            compute_persistence(data)
        self.assertEqual(
            [(list(v), f) for v, f in data],
            [(list(v), f) for v, f in snapshot],
        )


class TestValueSemantics(unittest.TestCase):
    def test_integer_filtrations_preserve_type(self):
        res = compute_persistence([((0,), 3)])
        self.assertEqual(res.cells[0].filtration, 3)
        self.assertIsInstance(res.cells[0].filtration, int)

    def test_float_filtrations_exact_passthrough(self):
        x = 0.1 + 0.2
        res = compute_persistence([((0,), 0), ((1,), 0), ((0, 1), x)])
        edge = next(c for c in res.cells if c.dimension == 1)
        self.assertEqual(edge.filtration, x)

    def test_half_open_boundary_exact(self):
        # birth included, death excluded, bit-exact float comparison
        res = compute_persistence(ref.triangle_ring_example())
        bar = res.h1_bars[0]
        self.assertTrue(bar.alive_at(1))
        self.assertTrue(bar.alive_at(1.0))
        self.assertFalse(bar.alive_at(2))
        self.assertFalse(bar.alive_at(2.0))

    def test_float_equality_gives_zero_length(self):
        x = 1 / 7
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), x), ((0, 2), x), ((1, 2), x),
            ((0, 1, 2), x),
        ]
        res = compute_persistence(data)
        self.assertTrue(res.h1_bars[0].zero_length)

    def test_repr_smoke(self):
        res = compute_persistence(ref.triangle_ring_example())
        self.assertIn("H1", repr(res.h1_bars[0]))
        self.assertIn("cells=7", repr(res))


class TestRandomizedCrossCheck(unittest.TestCase):
    """Deterministic-seed property tests against the independent reference.

    Generates many small face-closed 2-complexes with random integer
    filtrations (faces constrained to be <= simplices), shuffled input,
    and asserts the library's positive barcode, Betti numbers and every H1
    certificate agree with the rank-invariant / Gaussian-elimination
    reference.  Also checks structural invariants of the reduction.
    """

    @staticmethod
    def _random_complex(rng, nverts, maxf):
        edges = [
            (a, b)
            for a in range(nverts)
            for b in range(a + 1, nverts)
            if rng.random() < rng.uniform(0.3, 0.95)
        ]
        eset = set(edges)
        tris = [
            (a, b, c)
            for a in range(nverts)
            for b in range(a + 1, nverts)
            for c in range(b + 1, nverts)
            if {(a, b), (a, c), (b, c)} <= eset and rng.random() < 0.5
        ]
        fv = {v: rng.randint(0, maxf) for v in range(nverts)}
        data = [((v,), fv[v]) for v in range(nverts)]
        fe = {}
        for (a, b) in edges:
            f = rng.randint(max(fv[a], fv[b]), maxf)
            fe[(a, b)] = f
            data.append(((a, b), f))
        for tri in tris:
            lo = max(fe[(tri[0], tri[1])], fe[(tri[0], tri[2])],
                     fe[(tri[1], tri[2])])
            data.append((tri, rng.randint(lo, maxf)))
        rng.shuffle(data)
        return data

    def test_property_cross_check(self):
        rng = random.Random(20260923)
        for trial in range(300):
            with self.subTest(trial=trial):
                data = self._random_complex(
                    rng, rng.randint(1, 8), rng.randint(1, 4)
                )
                res = compute_persistence(data)

                # reduction invariants: each pivot row is a positive column,
                # pivot rows/owners are unique.
                reduced, _combo, owner = _reduce(
                    _build_boundary(list(res.cells))
                )
                for row in owner:
                    self.assertEqual(reduced[row], 0)
                self.assertEqual(len(set(owner)), len(owner))
                self.assertEqual(
                    len(set(owner.values())), len(owner)
                )

                for dim in (0, 1):
                    got = positive_bars(res, dim)
                    got = [x for x in got if x[0] != x[1]]
                    want = sorted(
                        ref.positive_barcode(data, dim),
                        key=lambda x: (x[0], x[1] is None, x[1]),
                    )
                    self.assertEqual(got, want)

                values = sorted({f for _, f in data})
                table = cells_table(res)
                for t in values:
                    self.assertEqual(
                        res.betti_at(t), ref.betti_ranks(data, t)
                    )
                    self.assertEqual(
                        res.betti_at(t)[0], ref.betti_union_find(data, t)
                    )
                for bar in res.h1_bars:
                    if bar.finite:
                        ok, reason = ref.verify_h1_certificate_cells(
                            table, bar.birth_index, bar.death_index,
                            bar.birth_chain, bar.death_chain,
                        )
                        self.assertTrue(ok, reason)

    def test_input_order_does_not_change_result(self):
        rng = random.Random(777)
        data = self._random_complex(rng, 6, 3)
        base = compute_persistence(data)
        base_sig = [
            (b.dimension, b.birth, None if b.death_index is None else b.death,
             b.zero_length)
            for b in base.bars
        ]
        for _ in range(5):
            shuffled = list(data)
            rng.shuffle(shuffled)
            other = compute_persistence(shuffled)
            sig = [
                (b.dimension, b.birth,
                 None if b.death_index is None else b.death, b.zero_length)
                for b in other.bars
            ]
            self.assertEqual(sig, base_sig)



def _shared_edge_two_cycles():
    """Two 1-cycles sharing edge (1, 2); all edges at t=1, vertices at 0.

    Cycle A = (0,1),(0,2),(1,2); cycle B = (1,2),(1,3),(2,3).
    Hand count: 5 edges, rank(d1) = 3 (a spanning tree on 4 vertices),
    no triangles -> beta1 = 5 - 3 = 2 at t=1.
    """
    return [
        ((0,), 0), ((1,), 0), ((2,), 0), ((3,), 0),
        ((0, 1), 1), ((0, 2), 1), ((1, 2), 1), ((1, 3), 1), ((2, 3), 1),
    ]


def _square_filled_at_two():
    """4-cycle at t=1; diagonal and both filling triangles at t=2."""
    return [
        ((0,), 0), ((1,), 0), ((2,), 0), ((3,), 0),
        ((0, 1), 1), ((1, 2), 1), ((2, 3), 1), ((0, 3), 1),
        ((0, 2), 2),
        ((0, 1, 2), 2), ((0, 2, 3), 2),
    ]


class TestSliceCyclesAnalytic(unittest.TestCase):
    """Hand-computed slice_cycles examples."""

    def test_triangle_ring_stepwise(self):
        res = compute_persistence(ref.triangle_ring_example())
        self.assertEqual(res.slice_cycles(0), ())
        (sc,) = res.slice_cycles(1)
        self.assertIsInstance(sc, SliceCycle)
        self.assertEqual(
            sorted(res.chain_vertices(sc.cycle)),
            [(0, 1), (0, 2), (1, 2)],
        )
        self.assertFalse(sc.essential)
        self.assertEqual((sc.birth, sc.death), (1, 2))
        self.assertIs(sc.bar, res.h1_bars[0])
        # The whole batch at t=2 (the triangle) closes the class.
        self.assertEqual(res.slice_cycles(2), ())
        self.assertEqual(res.slice_cycles(99), ())

    def test_two_cycles_sharing_an_edge(self):
        res = compute_persistence(_shared_edge_two_cycles())
        self.assertEqual(res.betti_at(1), (1, 2))
        cycles = res.slice_cycles(1)
        self.assertEqual(len(cycles), 2)
        edge_sets = [set(res.chain_vertices(sc.cycle)) for sc in cycles]
        # Every cycle is closed and uses only edges present at t=1.
        for es in edge_sets:
            parity = {}
            for a, b in es:
                parity[a] = parity.get(a, 0) ^ 1
                parity[b] = parity.get(b, 0) ^ 1
            self.assertFalse(any(parity.values()))
        # Mod boundaries (none here) the two generators are independent:
        # their symmetric difference is the non-empty third cycle.
        xor = edge_sets[0] ^ edge_sets[1]
        self.assertTrue(xor)
        # The two returned generators are closed and independent; their
        # symmetric difference is the third (dependent) cycle of the
        # hand count, one of {(0,1),(0,2),(1,2)}, {(1,2),(1,3),(2,3)},
        # {(0,1),(0,2),(1,3),(2,3)}.
        hand_cycles = [
            {(0, 1), (0, 2), (1, 2)},
            {(1, 2), (1, 3), (2, 3)},
            {(0, 1), (0, 2), (1, 3), (2, 3)},
        ]
        self.assertIn(xor, hand_cycles)
        self.assertTrue(all(es in hand_cycles for es in edge_sets))
        # Both classes are essential (never filled).
        self.assertTrue(all(sc.essential for sc in cycles))
        self.assertTrue(all(sc.homology_chain is None for sc in cycles))

    def test_shared_edge_cycles_then_filled(self):
        data = _shared_edge_two_cycles() + [
            ((0, 1, 2), 2), ((1, 2, 3), 2),
        ]
        res = compute_persistence(data)
        self.assertEqual(len(res.slice_cycles(1)), 2)
        self.assertEqual(res.slice_cycles(2), ())
        finite = [b for b in res.h1_bars if b.finite]
        self.assertEqual(len(finite), 2)

    def test_same_value_all_filled_batch(self):
        # Vertices, edges and the filling triangle all at t=0: the H1
        # class is born and dies inside one batch, so after the batch it
        # does not occupy the dimension.
        data = [
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), 0), ((0, 2), 0), ((1, 2), 0),
            ((0, 1, 2), 0),
        ]
        res = compute_persistence(data)
        (bar,) = res.h1_bars
        self.assertTrue(bar.zero_length)
        self.assertEqual(res.betti_at(0), (1, 0))
        self.assertEqual(res.slice_cycles(0), ())

    def test_square_batch_boundary_then_fill(self):
        res = compute_persistence(_square_filled_at_two())
        (sc,) = res.slice_cycles(1)
        # The generator at t=1 is exactly the 4 boundary edges; the
        # diagonal and triangles only exist from t=2 on.
        self.assertEqual(
            sorted(res.chain_vertices(sc.cycle)),
            [(0, 1), (0, 3), (1, 2), (2, 3)],
        )
        self.assertEqual(
            [sc.cycle for sc in res.slice_cycles(1.5)],
            [sc.cycle for sc in res.slice_cycles(1)],
        )
        self.assertEqual(res.slice_cycles(2), ())

    def test_essential_ring(self):
        data = ref.triangle_ring_example()[:-1]  # never filled
        res = compute_persistence(data)
        (sc,) = res.slice_cycles(7)
        self.assertTrue(sc.essential)
        self.assertEqual(sc.death, INF)
        self.assertIsNone(sc.homology_chain)
        self.assertIsNone(sc.bar.death_index)

    def test_correspondence_with_alive_bars(self):
        res = compute_persistence(ref.two_rings_one_filled())
        for t in (0, 1, 2, 10):
            alive = {
                b.birth_index for b in res.h1_bars if b.alive_at(t)
            }
            matched = {sc.bar.birth_index for sc in res.slice_cycles(t)}
            self.assertEqual(matched, alive)

    def test_bad_threshold_rejected(self):
        res = compute_persistence(ref.triangle_ring_example())
        for bad in (float("nan"), INF, -INF, True, "1", None):
            with self.assertRaises(PersistenceInputError) as ctx:
                res.slice_cycles(bad)
            self.assertEqual(ctx.exception.reason, "bad_threshold")


class TestSliceCyclesIndependentCheck(unittest.TestCase):
    """Cross-check slices against a boundary matrix rebuilt from scratch
    (vertex data only, GF(2) rank via tests/reference.py)."""

    def _check_slice(self, res, data, t):
        cells = res.cells
        cycles = res.slice_cycles(t)
        # Independent slice boundary data.
        edges = {}  # vertices -> filtration, only edges present at t
        tri_bounds = []
        for verts, f in data:
            if f > t:
                continue
            if len(verts) == 2:
                edges[tuple(verts)] = f
            elif len(verts) == 3:
                a, b, c = verts
                tri_bounds.append(
                    frozenset([(b, c), (a, c), (a, b)])
                )
        rank_b1 = ref.gf2_rank(tri_bounds)
        # 1. every returned cycle is closed and lies inside the slice
        # 2. generators are independent modulo boundaries
        # 3. their number equals the independently computed beta1
        cycle_vecs = []
        for sc in cycles:
            vec = set(res.chain_vertices(sc.cycle))
            self.assertTrue(vec)
            self.assertLessEqual(
                max(cells[i].filtration for i in sc.cycle), t,
                "cycle uses an edge from the future",
            )
            self.assertTrue(set(vec) <= set(edges))
            parity = {}
            for a, b in vec:
                parity[a] = parity.get(a, 0) ^ 1
                parity[b] = parity.get(b, 0) ^ 1
            self.assertFalse(any(parity.values()), "cycle not closed")
            cycle_vecs.append(frozenset(vec))
            # Correspondence: the matched bar is alive exactly now.
            self.assertTrue(sc.bar.alive_at(t))
            self.assertEqual(sc.essential, sc.bar.death_index is None)
            if sc.essential:
                self.assertIsNone(sc.homology_chain)
                self.assertIsNone(sc.residual_chain)
            else:
                # Trackable homology-equivalent transformation:
                # d(homology_chain)
                #   == cycle XOR birth_chain XOR residual_chain,
                # with residual a cycle of strictly earlier edges.
                residual = set(res.chain_vertices(sc.residual_chain))
                rparity = {}
                for a, b in residual:
                    rparity[a] = rparity.get(a, 0) ^ 1
                    rparity[b] = rparity.get(b, 0) ^ 1
                self.assertFalse(any(rparity.values()))
                self.assertTrue(
                    all(i < sc.bar.birth_index for i in sc.residual_chain)
                )
                produced = set()
                for ti in sc.homology_chain:
                    a, b, c = cells[ti].vertices
                    produced.symmetric_difference_update(
                        [(b, c), (a, c), (a, b)]
                    )
                expect = (
                    set(vec)
                    ^ set(res.chain_vertices(sc.bar.birth_chain))
                    ^ residual
                )
                self.assertEqual(produced, expect)
        indep = (
            ref.gf2_rank(tri_bounds + cycle_vecs) - rank_b1
        )
        self.assertEqual(indep, len(cycles), "cycles dependent mod boundaries")
        self.assertEqual(len(cycles), ref.betti_ranks(data, t)[1])

    def test_fixed_examples(self):
        datasets = [
            ref.triangle_ring_example(),
            ref.two_rings_one_filled(),
            ref.filled_grid_with_hole(),
            _shared_edge_two_cycles(),
            _shared_edge_two_cycles() + [((0, 1, 2), 2), ((1, 2, 3), 2)],
            _square_filled_at_two(),
        ]
        for data in datasets:
            res = compute_persistence(data)
            values = sorted({f for _, f in data})
            probes = sorted(
                set(values)
                | {values[0] - 1, values[-1] + 1}
                | {(a + b) / 2 for a, b in zip(values, values[1:])}
            )
            for t in probes:
                with self.subTest(t=t):
                    self._check_slice(res, data, t)

    def test_random_complexes(self):
        rng = random.Random(20260922)
        for trial in range(12):
            data = TestRandomizedCrossCheck._random_complex(
                rng, nverts=7, maxf=4
            )
            res = compute_persistence(data)
            values = sorted({f for _, f in data})
            for t in values + [values[-1] + 1]:
                with self.subTest(trial=trial, t=t):
                    self._check_slice(res, data, t)

    def test_original_pairing_untouched_by_slices(self):
        # slice_cycles is a read-only query: bars and certificates are
        # identical before and after slicing (no dynamic deletions).
        data = ref.filled_grid_with_hole()
        res = compute_persistence(data)
        before = list(res.bars)
        for t in (0, 1, 2, 3, 9):
            res.slice_cycles(t)
        self.assertEqual(list(res.bars), before)
        for bar in res.h1_bars:
            if bar.finite:
                self.assertTrue(check_certificate(res, bar))


if __name__ == "__main__":
    unittest.main(verbosity=2)
