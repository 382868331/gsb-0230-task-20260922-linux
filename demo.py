#!/usr/bin/env python3
"""Fixed-input demonstration for the pershom library.

Run with::

    python demo.py

It builds two small deterministic complexes, computes H0/H1 persistence
with GF(2) chain certificates, prints the results, and then performs one
input that genuinely fails validation (a triangle whose edge face is
missing).  Everything is computed live; nothing is pre-recorded and no
network or third-party package is used.
"""

from __future__ import annotations

from pershom import (
    PersistenceInputError,
    check_certificate,
    compute_persistence,
)


def triangle_ring_filled():
    """Three vertices at t=0, three edges at t=1 (a ring), fill at t=2."""
    return [
        ((0,), 0), ((1,), 0), ((2,), 0),
        ((0, 1), 1), ((1, 2), 1), ((0, 2), 1),
        ((0, 1, 2), 2),
    ]


def two_disjoint_rings():
    """Two separate triangle boundaries; only the second is filled.

    Expected topology: beta0 == 2 throughout; at t=1 beta1 == 2, at t=2
    beta1 == 1 (one ring stays open forever -> essential H1).
    """
    data = []
    for v in (0, 1, 2, 10, 11, 12):
        data.append(((v,), 0))
    for a, b in ((0, 1), (1, 2), (0, 2), (10, 11), (11, 12), (10, 12)):
        data.append(((a, b), 1))
    data.append(((10, 11, 12), 2))
    return data


def _fmt_value(v):
    return "inf" if v == float("inf") else repr(v)


def show(title, data):
    print("=" * 72)
    print(title)
    print("=" * 72)
    result = compute_persistence(data)

    print("\nFiltration order (index: dimension vertices @ filtration):")
    for cell in result.cells:
        print(f"  [{cell.index:2d}] d{cell.dimension} {str(cell.vertices):12s}"
              f" f={cell.filtration!r}")

    print("\nBarcode (half-open intervals [birth, death)):")
    for bar in result.bars:
        tag = "  zero-length" if bar.zero_length else ""
        print(f"  H{bar.dimension}  [{_fmt_value(bar.birth)}, "
              f"{_fmt_value(bar.death)}){tag}")

    print("\nBetti numbers (beta0, beta1) at each filtration value:")
    for t, betti in result.betti_profile():
        print(f"  t={_fmt_value(t):>4s} -> beta0={betti[0]}, beta1={betti[1]}")

    print("\nH1 generator cycles alive in each threshold subcomplex")
    print("(slice_cycles(t); every cycle uses only edges present at t):")
    for t, _ in result.betti_profile():
        cycles = result.slice_cycles(t)
        if not cycles:
            print(f"  t={_fmt_value(t):>4s} -> no live H1 class")
        for sc in cycles:
            kind = "essential" if sc.essential else (
                f"bar [{_fmt_value(sc.birth)}, {_fmt_value(sc.death)})")
            edges = result.chain_vertices(sc.cycle)
            print(f"  t={_fmt_value(t):>4s} -> {kind}: cycle edges {edges}")
            if sc.homology_chain or sc.residual_chain:
                witness = result.chain_vertices(sc.homology_chain)
                residual = result.chain_vertices(sc.residual_chain)
                print("         representative swap vs birth_chain tracked by"
                      f" triangles {witness}, residual earlier-cycle"
                      f" {residual}")

    finite_h1 = [b for b in result.h1_bars if b.finite]
    if finite_h1:
        print("\nFinite H1 certificates (GF(2) chains, verified independently):")
    for bar in finite_h1:
        birth_edges = result.chain_vertices(bar.birth_chain)
        death_tris = result.chain_vertices(bar.death_chain)
        print(f"  bar [{bar.birth!r}, {bar.death!r})")
        print(f"    birth 1-cycle (edges): {birth_edges}")
        print(f"    death 2-chain (triangles): {death_tris}")
        # Library self-check and an explicit boundary sanity print:
        check_certificate(result, bar)
        parity: dict[int, int] = {}
        for a, b in birth_edges:
            parity[a] = parity.get(a, 0) ^ 1
            parity[b] = parity.get(b, 0) ^ 1
        assert all(p == 0 for p in parity.values()), "birth chain has a boundary"
        boundary_edges: set[tuple[int, int]] = set()
        for tri in death_tris:
            a, b, c = tri
            for e in ((b, c), (a, c), (a, b)):
                boundary_edges.symmetric_difference_update([e])
        assert boundary_edges == set(birth_edges), "death boundary != birth cycle"
        print("    check: d(birth cycle)=0 and d(death chain)=birth cycle  OK")
    print()
    return result


def show_rejection():
    print("=" * 72)
    print("Error boundary (a triangle with a missing edge face is rejected)")
    print("=" * 72)
    bad = [
        ((0,), 0), ((1,), 0), ((2,), 0),
        ((0, 1), 1), ((1, 2), 1),   # edge (0, 2) deliberately omitted
        ((0, 1, 2), 2),
    ]
    print("\nAttempting to parse:")
    for verts, f in bad:
        print(f"  {str(verts):12s} @ {f}")
    try:
        compute_persistence(bad)
    except PersistenceInputError as exc:
        print(f"\nPersistenceInputError raised (locus input[{exc.index}]):")
        print(f"  reason : {exc.reason}")
        print(f"  message: {exc}")
        print("  -> no result object was produced; the input list was not "
              "modified.")
        return exc
    raise AssertionError("invalid complex was unexpectedly accepted")


def main():
    print("pershom demo - persistence homology pairing with GF(2) chains\n")
    show("Example 1: triangle ring born at t=1, filled (dies) at t=2",
         triangle_ring_filled())
    show("Example 2: two disjoint rings; one filled at t=2, one stays open",
         two_disjoint_rings())
    show_rejection()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
