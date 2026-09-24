"""Independent reference routines for the test-suite.

Nothing here imports :mod:`pershom`.  The barcode is recovered a different
way from the core algorithm -- from the **rank invariant** of the
persistence module maps, built with direct Gaussian elimination on GF(2)
subspaces -- and Betti numbers are computed by rank-nullity.  H0 is
cross-checked a third way, with union-find.
"""

from __future__ import annotations

from itertools import combinations


# ---------------------------------------------------------------------------
# Tiny GF(2) linear algebra on labelled coordinates (chains = sets of labels)
# ---------------------------------------------------------------------------

def gf2_rank(vectors):
    """Rank of a collection of GF(2) vectors represented as ``set``/frozenset."""
    pivots: dict[int, frozenset] = {}
    for vec in vectors:
        v = frozenset(vec)
        while v:
            p = max(v)
            if p in pivots:
                v = v.symmetric_difference(pivots[p])
            else:
                pivots[p] = v
                break
    return len(pivots)


def subspace_intersection_dim(vectors_a, vectors_b):
    """dim(U ∩ W) = dim U + dim W − dim(U + W) over GF(2)."""
    a = [frozenset(v) for v in vectors_a]
    b = [frozenset(v) for v in vectors_b]
    ra = gf2_rank(a)
    rb = gf2_rank(b)
    rs = gf2_rank(a + b)
    return ra + rb - rs


# ---------------------------------------------------------------------------
# Raw complex handling (input is a list of (vertices_tuple, filtration))
# ---------------------------------------------------------------------------

def _normalise(raw_simplices):
    verts_of = {}
    for verts, f in raw_simplices:
        verts_of[(len(verts) - 1, tuple(verts))] = f
    return verts_of


def filtration_values(raw_simplices):
    return sorted({f for _, f in raw_simplices})


def edge_boundary(edge):
    a, b = edge
    return frozenset([(0, (a,)), (0, (b,))])


def triangle_boundary(tri):
    a, b, c = tri
    return frozenset(
        [(1, (b, c)), (1, (a, c)), (1, (a, b))]
    )


def betti_ranks(raw_simplices, t):
    """Betti numbers (beta0, beta1) of the subcomplex with filtration <= t.

    Pure rank-nullity on the boundary maps, independently of the reduction.
    """
    vo = _normalise(raw_simplices)
    V = [k for k in vo if k[0] == 0 and vo[k] <= t]
    E = [k for k in vo if k[0] == 1 and vo[k] <= t]
    F = [k for k in vo if k[0] == 2 and vo[k] <= t]

    d1 = [edge_boundary(k[1]) for k in E]
    d2 = [triangle_boundary(k[1]) for k in F]
    r1 = gf2_rank(d1)
    r2 = gf2_rank(d2)
    beta0 = len(V) - r1
    beta1 = len(E) - r1 - r2
    return beta0, beta1


def dynamic_betti(simplices):
    """Betti numbers (beta0, beta1) of a bare (unfiltered) complex.

    ``simplices`` is any iterable of vertex tuples (length 1..3).  Pure
    rank-nullity over GF(2) on independently built boundary maps; shares no
    code with :mod:`pershom`.
    """
    verts = [tuple(v) for v in simplices]
    V = [v for v in verts if len(v) == 1]
    E = [v for v in verts if len(v) == 2]
    F = [v for v in verts if len(v) == 3]
    d1 = [edge_boundary(e) for e in E]
    d2 = [triangle_boundary(t) for t in F]
    r1 = gf2_rank(d1)
    r2 = gf2_rank(d2)
    return len(V) - r1, len(E) - r1 - r2


def betti_union_find(raw_simplices, t):
    """Independent beta0 via plain union-find."""
    parent = {}
    vo = _normalise(raw_simplices)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # First pass: activate every vertex present at t (input may be shuffled).
    for (dim, verts), f in vo.items():
        if dim == 0 and f <= t:
            parent.setdefault(verts[0], verts[0])
    for (dim, verts), f in vo.items():
        if dim == 1 and f <= t:
            a, b = verts
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    return len({find(v) for v in parent})


# ---------------------------------------------------------------------------
# Rank invariant and the independently recovered barcode (positive bars)
# ---------------------------------------------------------------------------

def _rank_k_at(vo, k, p_labels, t_q):
    """r^k(p, q) = rank(H_k(X_p) -> H_k(X_q)).

    ``p_labels`` are the k-cells present at p; the target subcomplex is cut
    at ``t_q``.  Inclusion is literal (global labels), so the rank is
    obtained from dimensions of intersections of cycle/boundary spaces.
    """
    if k == 0:
        Vp = {k_[1][0] for k_ in p_labels}
        E_q = [k_ for k_ in vo if k_[0] == 1 and vo[k_] <= t_q]
        bdy_q = [edge_boundary(e[1]) for e in E_q]
        units = [frozenset([(0, (v,))]) for v in Vp]
        inter = subspace_intersection_dim(bdy_q, units)
        return len(Vp) - inter
    else:  # k == 1
        Ep = {e[1] for e in p_labels}
        E_q = [k_ for k_ in vo if k_[0] == 1 and vo[k_] <= t_q]
        F_q = [k_ for k_ in vo if k_[0] == 2 and vo[k_] <= t_q]
        d1_q = [edge_boundary(e[1]) for e in E_q]
        bdy2_q = [triangle_boundary(f[1]) for f in F_q]
        n_ep = len(Ep)
        ker_dim_p = n_ep - gf2_rank(
            [edge_boundary(e) for e in Ep]
        )
        units = [frozenset([(1, tuple(sorted(e)))]) for e in Ep]
        # B1(q) ∩ C1(p); elements of B1(q) supported only on E_p are
        # automatically cycles of X_p.
        inter = subspace_intersection_dim(bdy2_q, units)
        return ker_dim_p - inter


def rank_invariant_table(raw_simplices):
    """Returns (values, r0, r1) with rk[i][j] = rank map H_k(X_i)->H_k(X_kj).

    Index -1 (empty complex) is represented as zero; callers guard for it.
    """
    vo = _normalise(raw_simplices)
    values = filtration_values(raw_simplices)

    def cells_up_to(k, t):
        return [key for key in vo if key[0] == k and vo[key] <= t]

    r0, r1 = {}, {}
    for i, fi in enumerate(values):
        for j in range(i, len(values)):
            tq = values[j]
            r0[(i, j)] = _rank_k_at(vo, 0, cells_up_to(0, fi), tq)
            r1[(i, j)] = _rank_k_at(vo, 1, cells_up_to(1, fi), tq)
    return values, r0, r1


def _r(table, i, j):
    """Rank table lookup treating the empty complex (index -1) as rank 0."""
    if i < 0 or j < 0:
        return 0
    return table[(i, j)]


def positive_barcode(raw_simplices, k):
    """Multiset of (birth_value, death_value) for bars with birth < death.

    Standard four-corner formula on the rank invariant
    (r(i, j) = rank(H(X_i) -> H(X_j)), i <= j):
        mu(i, j) = r(i, j-1) - r(i-1, j-1) - r(i, j) + r(i-1, j),  i < j
        mu(i, inf) = r(i, m) - r(i-1, m)
    Zero-length pairs are invisible to homology and are handled elsewhere.
    """
    values, r0, r1 = rank_invariant_table(raw_simplices)
    table = r0 if k == 0 else r1
    m = len(values) - 1
    bars = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            mu = (
                _r(table, i, j - 1)
                - _r(table, i - 1, j - 1)
                - _r(table, i, j)
                + _r(table, i - 1, j)
            )
            assert mu >= 0, ("rank invariant gave negative multiplicity", i, j)
            bars.extend([(values[i], values[j])] * mu)
        essential = _r(table, i, m) - _r(table, i - 1, m)
        bars.extend([(values[i], None)] * essential)
    return bars


# ---------------------------------------------------------------------------
# Independent H1 certificate check (operates on plain vertex data)
# ---------------------------------------------------------------------------

def verify_h1_certificate_cells(cells, birth_index, death_index,
                                birth_chain, death_chain):
    """Same check with an explicit ordered cell table ``(verts, filt)``."""
    edge_lookup = {
        tuple(cells[i][0]): i
        for i in range(len(cells))
        if len(cells[i][0]) == 2
    }
    if not birth_chain:
        return False, "empty birth chain"
    if not death_chain:
        return False, "empty death chain"
    if len(set(birth_chain)) != len(birth_chain):
        return False, "repeated edge in birth chain"
    if len(set(death_chain)) != len(death_chain):
        return False, "repeated triangle in death chain"
    if death_index not in death_chain:
        return False, "death triangle missing from death chain"

    fb = cells[birth_index][1]
    fd = cells[death_index][1]
    if fd < fb:
        return False, "death before birth"

    parity = {}
    for i in birth_chain:
        verts = cells[i][0]
        if len(verts) != 2:
            return False, f"cell {i} in birth chain is not an edge"
        if cells[i][1] > fd:
            return False, f"birth edge {verts} appears after death"
        for v in verts:
            parity[v] = parity.get(v, 0) ^ 1
    odd = sorted(v for v, p in parity.items() if p)
    if odd:
        return False, f"birth chain has boundary at vertices {odd}"
    if max(birth_chain) != birth_index:
        return False, "birth cell is not the pivot edge"

    produced = set()
    for i in death_chain:
        verts = cells[i][0]
        if len(verts) != 3:
            return False, f"cell {i} in death chain is not a triangle"
        if i > death_index:
            return False, "death chain uses a cell later than death"
        if cells[i][1] > fd:
            return False, f"triangle {verts} appears after death"
        a, b, c = verts
        for ev in ((b, c), (a, c), (a, b)):
            ei = edge_lookup.get(ev)
            if ei is None:
                return False, f"edge face {ev} missing"
            produced.symmetric_difference_update([ei])

    if produced != set(birth_chain):
        return False, (
            f"boundary mismatch: extra {sorted(produced - set(birth_chain))}, "
            f"missing {sorted(set(birth_chain) - produced)}"
        )
    return True, None


# ---------------------------------------------------------------------------
# Deterministic example complexes
# ---------------------------------------------------------------------------

def triangle_ring_example():
    """Boundary triangle at t=1, filled at t=2; vertices at t=0."""
    return [
        ((0,), 0), ((1,), 0), ((2,), 0),
        ((0, 1), 1), ((1, 2), 1), ((0, 2), 1),
        ((0, 1, 2), 2),
    ]


def filled_grid_with_hole(n=4, m=3):
    """Grid triangulation with one square left open (one essential H1).

    vertices t=0; grid edges t=1; the diagonal edge of each triangulated
    square t=2; the two triangles t=3.  The hole square at (1, 1) gets
    neither a diagonal nor triangles, so the complex stays face-closed.
    """
    simp = []
    for i in range(n):
        for j in range(m):
            simp.append(((i * m + j,), 0))
    for i in range(n):
        for j in range(m):
            v = i * m + j
            if i + 1 < n:
                simp.append(((v, (i + 1) * m + j), 1))
            if j + 1 < m:
                simp.append(((v, v + 1), 1))
    for i in range(n - 1):
        for j in range(m - 1):
            if (i, j) == (1, 1):
                continue  # open square: no diagonal, no triangles
            a = i * m + j
            b = a + 1
            c = a + m
            d = c + 1
            simp.append(((b, c), 2))
            simp.append(((a, b, c), 3))
            simp.append(((b, c, d), 3))
    return simp


def two_rings_one_filled():
    """Two disjoint triangle boundaries; only the second gets a fill."""
    simp = []
    ring0 = [0, 1, 2]
    ring1 = [10, 11, 12]
    for v in ring0 + ring1:
        simp.append(((v,), 0))
    for ring in (ring0, ring1):
        for a, b in combinations(ring, 2):
            simp.append(((a, b), 1))
    simp.append(((10, 11, 12), 2))
    return simp
