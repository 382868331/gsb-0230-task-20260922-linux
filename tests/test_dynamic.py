"""Tests for the dynamic (add/remove) complex: pershom.DynamicComplex.

Expected Betti numbers come from closed-form hand calculations on fixed
scripts, or from ``reference.dynamic_betti`` (an independent rank-nullity
implementation that never imports pershom).  Rejection behaviour is checked
against a plain Python-set model maintained by the test itself.
"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

import reference as ref

from pershom import (
    ADD,
    REMOVE,
    DynamicComplex,
    PersistenceInputError,
    compute_persistence,
)
from pershom.persistence import MAX_SIMPLICES


def model_faces(verts):
    """Codimension-1 faces, plain test-side set logic."""
    if len(verts) == 2:
        return [(verts[0],), (verts[1],)]
    a, b, c = verts
    return [(b, c), (a, c), (a, b)]


def model_add_error(model, verts):
    """None if adding ``verts`` to the model set is legal, else the reason."""
    dim = len(verts) - 1
    if (dim, verts) in model:
        return "duplicate_simplex"
    if len(model) >= MAX_SIMPLICES:
        return "too_many_simplices"
    if dim >= 1:
        for face in model_faces(verts):
            if (dim - 1, face) not in model:
                return "missing_face"
    return None


def model_remove_error(model, verts):
    """None if removing ``verts`` from the model set is legal, else reason."""
    dim = len(verts) - 1
    if (dim, verts) not in model:
        return "not_present"
    target = set(verts)
    for other_dim, other_verts in model:
        if other_dim > dim and target < set(other_verts):
            return "coface_present"
    return None


class TestHandComputedScripts(unittest.TestCase):
    """Fixed event scripts whose Betti numbers are computed by hand."""

    def test_ring_forms_edge_by_edge(self):
        cx = DynamicComplex()
        self.assertEqual(cx.betti(), (0, 0))  # empty complex
        # Three isolated vertices.
        for v in range(3):
            step = cx.add((v,))
        self.assertEqual(step.betti, (3, 0))
        # First two edges: still a forest, components merge.
        self.assertEqual(cx.add((0, 1)).betti, (2, 0))
        self.assertEqual(cx.add((1, 2)).betti, (1, 0))
        # Third edge closes the ring: H1 is born.
        step = cx.add((0, 2))
        self.assertEqual(step.betti, (1, 1))
        self.assertEqual(step.event, ADD)
        self.assertEqual(step.vertices, (0, 2))
        self.assertEqual(step.dimension, 1)

    def test_fill_then_delete_face_reopens_hole(self):
        cx = DynamicComplex()
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
        ])
        self.assertEqual(cx.betti(), (1, 1))
        # Filling the triangle kills the H1 class.
        self.assertEqual(cx.add((0, 1, 2)).betti, (1, 0))
        # Deleting the face reopens exactly the same hole.
        step = cx.remove((0, 1, 2))
        self.assertEqual(step.betti, (1, 1))
        self.assertEqual(step.event, REMOVE)
        self.assertEqual(step.dimension, 2)

    def test_delete_edges_unravels_ring(self):
        cx = DynamicComplex()
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
        ])
        self.assertEqual(cx.betti(), (1, 1))
        # Remove one edge: ring becomes a path, H1 dies, still connected.
        self.assertEqual(cx.remove((0, 2)).betti, (1, 0))
        # Remove another: one edge plus an isolated vertex.
        self.assertEqual(cx.remove((1, 2)).betti, (2, 0))
        # Last edge: three isolated vertices.
        self.assertEqual(cx.remove((0, 1)).betti, (3, 0))
        # Remove the vertices down to the empty complex.
        for v in range(3):
            cx.remove((v,))
        self.assertEqual(cx.betti(), (0, 0))
        self.assertEqual(len(cx), 0)
        self.assertEqual(cx.simplices(), ())

    def test_square_with_diagonal_hand_computed(self):
        # Square ring 0-1-2-3: beta = (1, 1).
        cx = DynamicComplex()
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)), (ADD, (3,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (2, 3)), (ADD, (0, 3)),
        ])
        self.assertEqual(cx.betti(), (1, 1))
        # Diagonal edge splits the square: two independent cycles.
        self.assertEqual(cx.add((0, 2)).betti, (1, 2))
        # Fill one half: one cycle remains.
        self.assertEqual(cx.add((0, 1, 2)).betti, (1, 1))
        # Fill the other half: contractible.
        self.assertEqual(cx.add((0, 2, 3)).betti, (1, 0))
        # Remove both faces: back to two cycles.
        self.assertEqual(cx.remove((0, 1, 2)).betti, (1, 1))
        self.assertEqual(cx.remove((0, 2, 3)).betti, (1, 2))
        # Remove the diagonal: back to the single square ring.
        self.assertEqual(cx.remove((0, 2)).betti, (1, 1))

    def test_two_components_independent(self):
        cx = DynamicComplex()
        # Ring on {0,1,2}; separate edge on {7,8}.
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
            (ADD, (7,)), (ADD, (8,)), (ADD, (7, 8)),
        ])
        self.assertEqual(cx.betti(), (2, 1))
        # Deleting in one component does not touch the other.
        self.assertEqual(cx.remove((7, 8)).betti, (3, 1))
        self.assertEqual(cx.remove((0, 2)).betti, (3, 0))

    def test_step_sequence_returned_by_apply(self):
        cx = DynamicComplex()
        steps = cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
            (ADD, (0, 1, 2)),
        ])
        # Hand-computed Betti after each event.
        want = [(1, 0), (2, 0), (3, 0), (2, 0), (1, 0), (1, 1), (1, 0)]
        self.assertEqual([s.betti for s in steps], want)
        self.assertEqual(
            [s.event for s in steps], [ADD] * 7
        )
        self.assertEqual(steps[-1].vertices, (0, 1, 2))
        self.assertEqual(steps[-1].dimension, 2)

    def test_empty_batch_is_noop(self):
        cx = DynamicComplex()
        cx.add((5,))
        self.assertEqual(cx.apply([]), [])
        self.assertEqual(cx.betti(), (1, 0))

    def test_betti_matches_known_small_enumeration(self):
        # Every subcomplex of the filled triangle (2^7 = 128 subsets that
        # are face-closed): compare against the independent reference.
        cells = [
            (0,), (1,), (2,),
            (0, 1), (0, 2), (1, 2),
            (0, 1, 2),
        ]
        faces_of = {
            (0, 1): [(0,), (1,)],
            (0, 2): [(0,), (2,)],
            (1, 2): [(1,), (2,)],
            (0, 1, 2): [(0, 1), (0, 2), (1, 2)],
        }
        checked = 0
        for mask in range(1 << len(cells)):
            subset = [cells[i] for i in range(len(cells)) if mask >> i & 1]
            keys = {(len(s) - 1, s) for s in subset}
            if any(
                (len(f) - 1, f) not in keys
                for s in subset
                for f in faces_of.get(s, [])
            ):
                continue  # not a complex
            cx = DynamicComplex()
            cx.apply([(ADD, s) for s in subset])
            self.assertEqual(cx.betti(), ref.dynamic_betti(subset), subset)
            checked += 1
        # Hand count of face-closed subsets of the 7-cell filled triangle:
        # 1 (empty) + 3 (one vertex) + 3*2 (two vertices, edge optional)
        # + 8 (three vertices, any edge subset) + 1 (all edges + face) = 19.
        self.assertEqual(checked, 19)


class TestAtomicRejections(unittest.TestCase):
    """Illegal events raise a locatable error and change nothing."""

    def setUp(self):
        # Filled triangle: vertices+edges+face on {0,1,2}.
        self.cx = DynamicComplex()
        self.cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
            (ADD, (0, 1, 2)),
        ])
        self.before = self.cx.simplices()

    def assert_unchanged(self):
        self.assertEqual(self.cx.simplices(), self.before)
        self.assertEqual(self.cx.betti(), (1, 0))

    def test_remove_edge_with_triangle_coface_rejected(self):
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.remove((0, 1))
        self.assertEqual(cm.exception.reason, "coface_present")
        self.assert_unchanged()
        # The complex is still fully usable afterwards.
        self.assertEqual(self.cx.remove((0, 1, 2)).betti, (1, 1))
        self.assertEqual(self.cx.remove((0, 1)).betti, (1, 0))

    def test_remove_vertex_with_edge_coface_rejected(self):
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.remove((0,))
        self.assertEqual(cm.exception.reason, "coface_present")
        self.assert_unchanged()

    def test_remove_vertex_with_triangle_coface_rejected(self):
        cx = DynamicComplex()
        cx.apply([(ADD, (0,)), (ADD, (1,)), (ADD, (0, 1))])
        with self.assertRaises(PersistenceInputError) as cm:
            cx.remove((1,))
        self.assertEqual(cm.exception.reason, "coface_present")
        self.assertEqual(cx.betti(), (1, 0))

    def test_remove_absent_simplex_rejected(self):
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.remove((9,))
        self.assertEqual(cm.exception.reason, "not_present")
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.remove((0, 3))
        self.assertEqual(cm.exception.reason, "not_present")
        self.assert_unchanged()

    def test_remove_from_empty_complex_rejected(self):
        cx = DynamicComplex()
        with self.assertRaises(PersistenceInputError) as cm:
            cx.remove((0,))
        self.assertEqual(cm.exception.reason, "not_present")
        self.assertEqual(cx.betti(), (0, 0))

    def test_add_edge_with_missing_vertex_rejected(self):
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.add((0, 5))
        self.assertEqual(cm.exception.reason, "missing_face")
        self.assert_unchanged()

    def test_add_triangle_with_missing_edge_rejected(self):
        cx = DynamicComplex()
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)),
        ])
        with self.assertRaises(PersistenceInputError) as cm:
            cx.add((0, 1, 2))
        self.assertEqual(cm.exception.reason, "missing_face")
        self.assertEqual(len(cx), 5)

    def test_add_duplicate_rejected(self):
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.add((0, 1))
        self.assertEqual(cm.exception.reason, "duplicate_simplex")
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.add((0,))
        self.assertEqual(cm.exception.reason, "duplicate_simplex")
        self.assert_unchanged()

    def test_bad_vertices_rejected(self):
        for bad, reason in [
            ((1, 0), "unsorted_vertices"),
            ((0, 0), "duplicate_vertex_in_simplex"),
            ((True,), "bad_vertex_id"),
            ((-1,), "bad_vertex_id"),
            ((), "bad_dimension"),
            ((0, 1, 2, 3), "bad_dimension"),
            ("ab", "bad_vertices"),
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(PersistenceInputError) as cm:
                    self.cx.add(bad)
                self.assertEqual(cm.exception.reason, reason)
                with self.assertRaises(PersistenceInputError):
                    self.cx.remove(bad)
        self.assert_unchanged()

    def test_too_many_simplices(self):
        cx = DynamicComplex()
        for v in range(MAX_SIMPLICES):
            cx.add((v,))
        self.assertEqual(len(cx), MAX_SIMPLICES)
        with self.assertRaises(PersistenceInputError) as cm:
            cx.add((MAX_SIMPLICES,))
        self.assertEqual(cm.exception.reason, "too_many_simplices")
        self.assertEqual(len(cx), MAX_SIMPLICES)
        # Removing one makes room again.
        cx.remove((0,))
        cx.add((MAX_SIMPLICES,))
        self.assertEqual(len(cx), MAX_SIMPLICES)

    def test_bad_events_rejected(self):
        for bad, reason in [
            ([("delete", (0,))], "bad_event"),
            ([("add",)], "bad_event"),
            ([42], "bad_event"),
            ([(ADD, (0,), 1)], "bad_event"),
            ("add", "bad_input"),
            (42, "bad_input"),
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(PersistenceInputError) as cm:
                    self.cx.apply(bad)
                self.assertEqual(cm.exception.reason, reason)
        self.assert_unchanged()

    def test_apply_is_transactional(self):
        # The 4th event (remove an edge with a coface) is illegal: the
        # whole batch must be refused and nothing may change.
        batch = [
            (ADD, (10,)),
            (ADD, (11,)),
            (ADD, (10, 11)),
            (REMOVE, (0, 1)),   # illegal: triangle (0,1,2) still present
        ]
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.apply(batch)
        self.assertEqual(cm.exception.reason, "coface_present")
        self.assertEqual(cm.exception.index, 3)
        self.assert_unchanged()
        self.assertNotIn((10,), self.cx)

    def test_apply_error_index_points_at_offender(self):
        batch = [
            (ADD, (10,)),
            (ADD, (10,)),  # duplicate within the same batch
        ]
        with self.assertRaises(PersistenceInputError) as cm:
            self.cx.apply(batch)
        self.assertEqual(cm.exception.reason, "duplicate_simplex")
        self.assertEqual(cm.exception.index, 1)
        self.assert_unchanged()

    def test_error_is_value_error_family(self):
        try:
            self.cx.remove((0, 1))
        except ValueError:
            pass
        else:  # pragma: no cover
            self.fail("expected ValueError family")


class TestRandomizedModelCrossCheck(unittest.TestCase):
    """Random event streams checked against a set model + independent betti.

    The test itself decides, with plain set logic, whether each random
    event is legal; the library must agree on acceptance, on the rejection
    reason, on the resulting contents, and on the Betti numbers (checked
    against ``reference.dynamic_betti``).
    """

    def test_random_event_stream(self):
        rng = random.Random(20260924)
        cx = DynamicComplex()
        model = set()
        nverts = 6
        pool = []
        for k in (1, 2, 3):
            pool.extend(tuple(c) for c in combinations(range(nverts), k))
        for step_no in range(600):
            verts = pool[rng.randrange(len(pool))]
            op = ADD if rng.random() < 0.55 else REMOVE
            if op == ADD:
                want_err = model_add_error(model, verts)
            else:
                want_err = model_remove_error(model, verts)
            if want_err is None:
                if op == ADD:
                    step = cx.add(verts)
                    model.add((len(verts) - 1, verts))
                else:
                    step = cx.remove(verts)
                    model.discard((len(verts) - 1, verts))
                self.assertEqual(step.event, op)
                self.assertEqual(step.vertices, verts)
                self.assertEqual(step.betti, cx.betti())
            else:
                with self.assertRaises(PersistenceInputError) as cm:
                    cx.add(verts) if op == ADD else cx.remove(verts)
                self.assertEqual(
                    cm.exception.reason, want_err, (step_no, op, verts)
                )
            self.assertEqual(cx.simplices(), tuple(v for _d, v in sorted(model)))
            self.assertEqual(
                cx.betti(),
                ref.dynamic_betti([v for _d, v in sorted(model)]),
                (step_no, op, verts),
            )

    def test_random_transactional_batches(self):
        rng = random.Random(97531)
        cx = DynamicComplex()
        model = set()
        pool = [
            tuple(c)
            for k in (1, 2, 3)
            for c in combinations(range(5), k)
        ]
        for _trial in range(120):
            batch = [
                (ADD if rng.random() < 0.5 else REMOVE,
                 pool[rng.randrange(len(pool))])
                for _ in range(rng.randint(1, 8))
            ]
            # Simulate on a scratch copy of the model.
            work = set(model)
            first_error = None
            for i, (op, verts) in enumerate(batch):
                err = (
                    model_add_error(work, verts)
                    if op == ADD
                    else model_remove_error(work, verts)
                )
                if err is not None:
                    first_error = (i, err)
                    break
                key = (len(verts) - 1, verts)
                if op == ADD:
                    work.add(key)
                else:
                    work.discard(key)
            before = cx.simplices()
            if first_error is not None:
                i, reason = first_error
                with self.assertRaises(PersistenceInputError) as cm:
                    cx.apply(batch)
                self.assertEqual(cm.exception.reason, reason)
                self.assertEqual(cm.exception.index, i)
                self.assertEqual(cx.simplices(), before)  # untouched
            else:
                steps = cx.apply(batch)
                self.assertEqual(len(steps), len(batch))
                model = work
                self.assertEqual(cx.simplices(), tuple(v for _d, v in sorted(model)))
            self.assertEqual(
                cx.betti(),
                ref.dynamic_betti([v for _d, v in sorted(model)]),
            )


class TestStaticInterfaceUnaffected(unittest.TestCase):
    """The pre-existing static entry point keeps working side by side."""

    def test_static_compute_still_works(self):
        res = compute_persistence([
            ((0,), 0), ((1,), 0), ((2,), 0),
            ((0, 1), 1), ((1, 2), 1), ((0, 2), 1),
            ((0, 1, 2), 2),
        ])
        self.assertEqual(res.betti_at(1), (1, 1))
        self.assertEqual(res.betti_at(2), (1, 0))

    def test_dynamic_and_static_agree_on_final_complex(self):
        # Build the filled triangle dynamically; the static filtration that
        # adds everything at t=0 must give the same Betti numbers.
        cx = DynamicComplex()
        cx.apply([
            (ADD, (0,)), (ADD, (1,)), (ADD, (2,)),
            (ADD, (0, 1)), (ADD, (1, 2)), (ADD, (0, 2)),
            (ADD, (0, 1, 2)),
        ])
        res = compute_persistence([(s, 0) for s in cx.simplices()])
        self.assertEqual(cx.betti(), res.betti_at(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
