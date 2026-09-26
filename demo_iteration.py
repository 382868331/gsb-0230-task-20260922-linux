"""Offline demo: H1 x H1 -> H2 cup product on a fixed filtration slice.

Builds the 7-vertex torus and the wedge S1 v S1 v S2 (same Betti numbers
(1, 2, 1), different multiplication), evaluates the Alexander-Whitney cup
product over GF(2), reduces to H^2 coordinates, checks invariance under
coboundary changes, and triggers one real rejection.  Pure standard
library, deterministic, no sleep, no network, no prerecorded results.
Runs in well under 90 seconds (actual: milliseconds).
"""

from __future__ import annotations

import time

from pershom import (
    PersistenceInputError,
    cohomology_slice,
    compute_persistence,
    cup_product,
)


def complex_from(triangles=(), extra_edges=()):
    tris = sorted({tuple(sorted(t)) for t in triangles})
    edges = {tuple(sorted(e)) for e in extra_edges}
    for a, b, c in tris:
        edges |= {(a, b), (a, c), (b, c)}
    edges = sorted(edges)
    verts = sorted({v for e in edges for v in e})
    return (
        [((v,), 0) for v in verts]
        + [(e, 0) for e in edges]
        + [(t, 0) for t in tris]
    )


def torus7():
    tris = []
    for i in range(7):
        tris.append(tuple(sorted((i, (i + 1) % 7, (i + 3) % 7))))
        tris.append(tuple(sorted((i, (i + 2) % 7, (i + 3) % 7))))
    return complex_from(tris)


def wedge_s1_s1_s2():
    sphere = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    loops = [(0, 4), (4, 5), (0, 5), (0, 6), (6, 7), (0, 7)]
    return complex_from(sphere, extra_edges=loops)


def show(name, simplices):
    sl = cohomology_slice(simplices, 0)
    pers = compute_persistence(simplices)
    print(f"== {name} ==")
    print(f"  slice: {sl!r}")
    print(f"  persistence betti at t=0: {pers.betti_at(0)}")
    print(f"  H^1 basis ({sl.b1} classes):")
    for i, cocycle in enumerate(sl.h1_basis):
        print(f"    h1[{i}] = {list(cocycle)}")
    print(f"  H^2 basis ({sl.b2} classes): {[list(c) for c in sl.h2_basis]}")
    for i in range(sl.b1):
        for j in range(sl.b1):
            res = cup_product(sl, sl.h1_basis[i], sl.h1_basis[j])
            print(
                f"    h1[{i}] ~ h1[{j}]: raw={list(res.raw)} "
                f"coords={res.coordinates} zero={res.is_zero}"
            )
    return sl


def main():
    t0 = time.perf_counter()

    torus = torus7()
    wedge = wedge_s1_s1_s2()

    sl_torus = show("7-vertex torus", torus)
    print()
    show("wedge S1 v S1 v S2 (same betti, trivial product)", wedge)
    print()

    # Invariance: adding a coboundary (vertex star) must not move the class.
    a, b = sl_torus.h1_basis
    base = cup_product(sl_torus, a, b).coordinates
    star = [e for e in sl_torus.edges if 0 in e]  # delta_0 of vertex 0
    a_shifted = sorted(set(a) ^ set(star))
    shifted = cup_product(sl_torus, a_shifted, b).coordinates
    print(f"coboundary invariance: {base} -> {shifted} (must be equal)")
    assert shifted == base

    # A real rejection: a single edge is not a closed 1-cocycle.
    try:
        cup_product(sl_torus, [(0, 1)], b)
    except PersistenceInputError as exc:
        print(f"rejection demo: reason={exc.reason!r} :: {exc}")

    dt = time.perf_counter() - t0
    print(f"done in {dt:.3f}s (budget 90s)")


if __name__ == "__main__":
    main()

