"""Offline demo of the H^1 x H^1 -> H^2 cup-product query (GF(2)).

Builds two complexes with identical Betti numbers (1, 2, 1) -- a 3x3 grid
torus and a wedge S^1 v S^1 v S^2 -- and shows that the cup product tells
them apart.  Every printed number is computed live; no sleeps, no network,
no prerecorded results.  Runtime is well under 90 seconds (typ. < 0.1 s).
"""

import itertools
import time

from pershom import (
    PersistenceInputError,
    check_cup_product,
    compute_cohomology,
    compute_persistence,
)


def torus_complex():
    def vid(i, j):
        return i * 3 + j

    simps = [((vid(i, j),), 0) for i in range(3) for j in range(3)]
    edges, tris = set(), set()
    for i in range(3):
        for j in range(3):
            a, b = vid(i, j), vid((i + 1) % 3, j)
            c, d = vid(i, (j + 1) % 3), vid((i + 1) % 3, (j + 1) % 3)
            for t in (tuple(sorted((a, b, d))), tuple(sorted((a, c, d)))):
                tris.add(t)
                edges.update({(t[0], t[1]), (t[0], t[2]), (t[1], t[2])})
    return simps + [(e, 1) for e in sorted(edges)] + [(t, 2) for t in sorted(tris)]


def wedge_complex():
    circles = [(0, 1, 2), (0, 3, 4)]
    sphere = [(0, 5, 6), (0, 5, 7), (0, 6, 7), (5, 6, 7)]
    verts = sorted({v for t in circles + sphere for v in t})
    edges = set()
    for t in circles + sphere:
        edges.update({(t[0], t[1]), (t[0], t[2]), (t[1], t[2])})
    return (
        [((v,), 0) for v in verts]
        + [(e, 0) for e in sorted(edges)]
        + [(t, 0) for t in sorted(sphere)]
    )


def show(name, simps, threshold=None):
    cs = compute_cohomology(simps, threshold=threshold)
    pers = compute_persistence(simps)
    print(f"== {name} ==")
    print(f"  slice: {cs!r}")
    print(f"  persistent Betti at slice: beta0..beta1 = {pers.betti_at(threshold if threshold is not None else 2)}")
    print(f"  H^1 basis ({cs.betti1} cocycles):")
    for k, cyc in enumerate(cs.h1_basis[:6]):
        print(f"    a{k} = {[cs.edges[i] for i in cyc]}")
    if cs.betti1 > 6:
        print(f"    ... ({cs.betti1 - 6} more)")
    print(f"  H^2 basis ({cs.betti2} classes): "
          f"{[[cs.triangles[i] for i in h] for h in cs.h2_basis]}")
    print("  cup-product matrix in the H^2 basis:")
    shown = min(cs.betti1, 2)
    for i, j in itertools.product(range(cs.betti1), repeat=2):
        cp = cs.cup_product(i, j)
        check_cup_product(cs, cp)  # independent recomputation
    for i, j in itertools.product(range(shown), repeat=2):
        cp = cs.cup_product(i, j)
        raw = [cs.triangles[t] for t in cp.raw_product]
        print(f"    a{i} ~ a{j}: coords={cp.h2_coordinates} "
              f"zero={cp.is_zero} raw_support={raw}")
    if cs.betti1 > shown:
        print(f"    ... all {cs.betti1}x{cs.betti1} pairs computed and "
              f"verified; only the first {shown}x{shown} shown")
    return cs


def main():
    t0 = time.perf_counter()

    torus = show("3x3 grid torus (threshold=None, full complex)", torus_complex())

    # Coboundary invariance, checked live on the torus.
    b0 = torus.b1_basis[0]
    shifted = sorted(set(torus.h1_basis[0]) ^ set(b0))
    cp = torus.cup_product_of(shifted, torus.h1_basis[1])
    print("  coboundary-shifted a0 ~ a1: coords =", cp.h2_coordinates,
          "(unchanged, raw support moved to",
          [torus.triangles[t] for t in cp.raw_product], ")")

    show("wedge S^1 v S^1 v S^2 (same Betti numbers)", wedge_complex())

    # A growing filtration slice of the torus: edges only at t=1.
    show("torus slice at threshold=1 (no triangles yet)", torus_complex(), threshold=1)

    # One genuinely triggered rejection.
    try:
        torus.cup_product_of((0,), torus.h1_basis[0])
    except PersistenceInputError as exc:
        print(f"rejected non-cocycle input as expected: {exc}")

    print(f"total runtime: {time.perf_counter() - t0:.3f}s (budget 90s)")


if __name__ == "__main__":
    main()
