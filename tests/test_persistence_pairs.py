"""persistence_pairs 的必要测试。

测试中的期望值全部来自手算解析例，或来自*独立*参考实现
（:func:`_ref_betti` 用并查集 + 整数位掩码 GF(2) 消元自行计算阈值 Betti 数，
不导入、不调用被测核心的任何函数）。对被测核心只做“断言”，不用它生成期望。
"""

import math
import random
import unittest
from io import StringIO

import persistence_pairs as pp


# --------------------------------------------------------------------------- #
# 独立参考：给定原始 (单形, 值) 数据，自行算阈值 t 的 (b0, b1)。
# 与被测实现的表示/算法完全不同：并查集算 b0，整数位掩码消元算 ∂2 的秩。
# --------------------------------------------------------------------------- #
def _ref_betti(vertices, edges, triangles, t):
    """返回 (b0, b1, b2)。2 维复形中 b2 = 活动三角形数 - rank(∂2)。"""
    verts = sorted(v for v, val in vertices if val <= t)
    vidx = {v: i for i, v in enumerate(verts)}

    e_list = sorted(tuple(sorted(e)) for e, val in edges if val <= t)
    eidx = {e: i for i, e in enumerate(e_list)}

    active_tris = [tri for tri, val in triangles if val <= t]

    parent = list(range(len(verts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in e_list:
        ru, rv = find(vidx[u]), find(vidx[v])
        if ru != rv:
            parent[ru] = rv

    b0 = len({find(i) for i in range(len(verts))}) if verts else 0
    # 1 维圈空间维数 = |E| - |V| + b0。
    cycle_dim = len(e_list) - len(verts) + b0

    # 独立地对三角形边界列做 GF(2) 主元消元（位掩码版），求 im ∂2 的秩。
    pivots = {}  # 最高位 -> 掩码
    rank2 = 0
    for tri in active_tris:
        u, v, w = sorted(tri)
        mask = (
            (1 << eidx[(u, v)])
            | (1 << eidx[(u, w)])
            | (1 << eidx[(v, w)])
        )
        while mask:
            high = mask.bit_length() - 1
            if high in pivots:
                mask ^= pivots[high]
            else:
                pivots[high] = mask
                rank2 += 1
                break

    b1 = cycle_dim - rank2
    b2 = len(active_tris) - rank2
    return b0, b1, b2


def _boundary1(edge_chain):
    """独立手算：边链（元组集合）的 GF(2) 顶点边界。"""
    out = set()
    for u, v in edge_chain:
        out.symmetric_difference_update([u, v])
    return out


def _boundary2(tri_chain):
    """独立手算：三角形链（元组集合）的 GF(2) 边边界。"""
    out = set()
    for u, v, w in tri_chain:
        out.symmetric_difference_update([(u, v), (u, w), (v, w)])
    return out


# --------------------------------------------------------------------------- #
# 解析例 1：三角环 + 填面
# --------------------------------------------------------------------------- #
def _triangle_ring_filled():
    vertices = [("a", 0), ("b", 0), ("c", 0)]
    edges = [(("a", "b"), 1), (("b", "c"), 1), (("c", "a"), 1)]
    triangles = [(("a", "b", "c"), 2)]
    return vertices, edges, triangles


class TestTriangleRingFilled(unittest.TestCase):
    def setUp(self):
        self.vertices, self.edges, self.triangles = _triangle_ring_filled()
        self.res = pp.compute_persistence(self.vertices, self.edges, self.triangles)

    def test_pairing_analytic(self):
        h0 = sorted(self.res.bars(0), key=lambda p: p.birth_simplex)
        h1 = self.res.bars(1)

        # H0：两条有限条（值 0 -> 1），一条无限条。
        finite_h0 = [p for p in h0 if not p.essential]
        ess_h0 = [p for p in h0 if p.essential]
        self.assertEqual(len(finite_h0), 2)
        self.assertEqual(len(ess_h0), 1)
        for p in finite_h0:
            self.assertEqual(p.birth_value, 0)
            self.assertEqual(p.death_value, 1)
            self.assertFalse(p.zero_length)
        # 同值顶点按顶点序：a 先，故无限条出生于 a。
        self.assertEqual(ess_h0[0].birth_simplex, ("a",))
        self.assertEqual(ess_h0[0].death_value, math.inf)
        self.assertTrue(ess_h0[0].essential)

        # H1：恰一条有限条，1 -> 2，死于唯一三角形。
        self.assertEqual(len(h1), 1)
        bar = h1[0]
        self.assertFalse(bar.essential)
        self.assertEqual(bar.birth_value, 1)
        self.assertEqual(bar.death_value, 2)
        self.assertEqual(bar.death_simplex, ("a", "b", "c"))
        self.assertIsNotNone(bar.certificate)

    def test_birth_edge_is_part_of_cycle(self):
        bar = self.res.bars(1)[0]
        self.assertIn(
            bar.birth_simplex,
            set(bar.certificate.birth_edges),
        )

    def test_certificate_chain_relations_independent(self):
        bar = self.res.bars(1)[0]
        cert = bar.certificate

        # 独立手算：出生环边界为 0。
        self.assertEqual(_boundary1(set(cert.birth_edges)), set())
        # 独立手算：∂(填充链) == 出生环链。
        self.assertEqual(
            _boundary2(set(cert.filling_triangles)),
            set(cert.birth_edges),
        )
        # 填充链就是那一个三角形；出生环就是三条边。
        self.assertEqual(set(cert.filling_triangles), {("a", "b", "c")})
        self.assertEqual(
            set(cert.birth_edges), {("a", "b"), ("b", "c"), ("a", "c")}
        )

    def test_verify_certificate_helper(self):
        bar = self.res.bars(1)[0]
        self.assertTrue(pp.verify_certificate(bar, self.res))

    def test_betti_numbers_against_thresholds(self):
        raw = (self.vertices, self.edges, self.triangles)
        cases = [(-1, (0, 0)), (0, (3, 0)), (1, (1, 1)), (2, (1, 0)), (5, (1, 0))]
        for t, expect in cases:
            self.assertEqual(self.res.betti_at(t), expect, f"threshold {t}")
            # 与独立参考逐项核对（参考返回 (b0,b1,b2)，这里比较前两维）。
            self.assertEqual(_ref_betti(*raw, t=t)[:2], expect, f"ref threshold {t}")
            self.assertEqual(self.res.betti_at(t), _ref_betti(*raw, t=t)[:2])
            # H2 在本例恒为 0。
            self.assertEqual(self.res.betti_at(t, 2), 0)

    def test_betti_at_boundary_is_left_closed_right_open(self):
        # 在死亡值 2 处条已结束（[birth, death)）。
        self.assertEqual(self.res.betti_at(1.999), (1, 1))
        self.assertEqual(self.res.betti_at(2.0), (1, 0))
        # 出生值 0 处组件立即出现。
        self.assertEqual(self.res.betti_at(0.0), (3, 0))


# --------------------------------------------------------------------------- #
# 解析例 2：不填的三角环 -> 无限 H1
# --------------------------------------------------------------------------- #
class TestUnfilledRingEssentialH1(unittest.TestCase):
    def test_essential_h1(self):
        vertices = [(0, 0), (1, 0), (2, 0)]
        edges = [((0, 1), 1), ((1, 2), 1), ((0, 2), 1)]
        res = pp.compute_persistence(vertices, edges)
        h1 = res.bars(1)
        self.assertEqual(len(h1), 1)
        self.assertTrue(h1[0].essential)
        self.assertIsNone(h1[0].death_simplex)
        self.assertEqual(h1[0].death_value, math.inf)
        self.assertIsNone(h1[0].certificate)
        self.assertEqual(res.betti_at(1), (1, 1))


# --------------------------------------------------------------------------- #
# 同值决胜：维数优先（顶点先于边），再按顶点序
# --------------------------------------------------------------------------- #
class TestEqualValuesTieBreak(unittest.TestCase):
    def test_all_same_value_dimension_then_vertex_order(self):
        # 顶点与边同值 0：顶点必先出现，b 死于边 ab，无限条属于 a。
        vertices = [("a", 0), ("b", 0)]
        edges = [(("a", "b"), 0)]
        res = pp.compute_persistence(vertices, edges)
        h0 = res.bars(0)
        self.assertEqual(len(h0), 2)
        ess = [p for p in h0 if p.essential]
        fin = [p for p in h0 if not p.essential]
        self.assertEqual(len(ess), 1)
        self.assertEqual(ess[0].birth_simplex, ("a",))
        self.assertEqual(len(fin), 1)
        # 零长度条必须保留且显式标记。
        self.assertTrue(fin[0].zero_length)
        self.assertEqual(fin[0].birth_value, fin[0].death_value)

    def test_zero_length_bars_are_retained_not_dropped(self):
        vertices = [(i, 0) for i in range(4)]
        edges = [((0, 1), 0), ((1, 2), 0), ((2, 3), 0)]
        res = pp.compute_persistence(vertices, edges)
        fin = [p for p in res.bars(0) if not p.essential]
        self.assertEqual(len(fin), 3)
        self.assertTrue(all(p.zero_length for p in fin))
        # 零长度条在任何阈值都不存活。
        self.assertEqual(res.betti_at(0), (1, 0))

    def test_equal_value_vertex_order_swaps_essential(self):
        # 显式以顶点序决定存活者：先声明顺序无关，排序后 (1,) 在 (2,) 前。
        res = pp.compute_persistence([(2, 0), (1, 0)], [( (1, 2), 0)])
        ess = [p for p in res.bars(0) if p.essential]
        self.assertEqual(ess[0].birth_simplex, (1,))


# --------------------------------------------------------------------------- #
# 分离组件
# --------------------------------------------------------------------------- #
class TestDisconnectedComponents(unittest.TestCase):
    def test_two_components(self):
        vertices = [("a", 0), ("b", 0), ("c", 0), ("d", 0)]
        edges = [(("a", "b"), 1), (("c", "d"), 2)]
        res = pp.compute_persistence(vertices, edges)
        ess = [p for p in res.bars(0) if p.essential]
        fin = [p for p in res.bars(0) if not p.essential]
        self.assertEqual(len(ess), 2)
        self.assertEqual(len(fin), 2)
        self.assertEqual(res.betti_at(0), (4, 0))
        self.assertEqual(res.betti_at(1), (3, 0))
        self.assertEqual(res.betti_at(2), (2, 0))

    def test_isolated_vertex_survives_forever(self):
        res = pp.compute_persistence([("x", 3), ("y", 7)], [])
        self.assertEqual(res.betti_at(100), (2, 0))
        self.assertEqual(res.betti_at(5), (1, 0))
        self.assertEqual(res.betti_at(2), (0, 0))


# --------------------------------------------------------------------------- #
# 扇形圆盘：多个同值零长度 H1 条 + 一个大环死于最后一个三角形
# --------------------------------------------------------------------------- #
class TestFanDisk(unittest.TestCase):
    def _build(self, n=6):
        vertices = [(i, 0.0) for i in range(n)]
        edges = [((i, (i + 1) % n), 1.0) for i in range(n)]
        edges += [((0, i), 2.0) for i in range(2, n - 1)]  # 扇形弦
        triangles = [((0, i, i + 1), 2.0) for i in range(1, n - 1)]
        return vertices, edges, triangles

    def test_big_cycle_birth_death_and_zero_length_h1(self):
        vertices, edges, triangles = self._build()
        res = pp.compute_persistence(vertices, edges, triangles)
        h1 = res.bars(1)
        essential = [p for p in h1 if p.essential]
        finite = [p for p in h1 if not p.essential]
        self.assertEqual(essential, [])
        # 大环：1.0 -> 2.0；其余为同值出生死亡的零长度条。
        big = [p for p in finite if not p.zero_length]
        zero = [p for p in finite if p.zero_length]
        self.assertEqual(len(big), 1)
        self.assertEqual(big[0].birth_value, 1.0)
        self.assertEqual(big[0].death_value, 2.0)
        self.assertEqual(big[0].death_simplex, (0, 4, 5))
        self.assertTrue(pp.verify_certificate(big[0], res))
        self.assertTrue(zero)  # n-3 个小三角形圈，同值即生即灭但保留
        for p in zero:
            self.assertEqual((p.birth_value, p.death_value), (2.0, 2.0))

    def test_betti_matches_independent_reference(self):
        raw = self._build()
        res = pp.compute_persistence(*raw)
        for t in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 9.0):
            self.assertEqual(res.betti_at(t), _ref_betti(*raw, t=t)[:2], f"t={t}")
            self.assertEqual(res.betti_at(t, 2), _ref_betti(*raw, t=t)[2], f"t={t}")
        self.assertEqual(res.betti_at(1.0), (1, 1))
        self.assertEqual(res.betti_at(2.0), (1, 0))


# --------------------------------------------------------------------------- #
# 输入校验与错误语义
# --------------------------------------------------------------------------- #
class TestInputValidation(unittest.TestCase):
    def _expect(self, *args, **kw):
        with self.assertRaises(pp.InputError) as cm:
            pp.Filtration(*args, **kw)
        return cm.exception

    def test_nan_rejected(self):
        err = self._expect([("a", float("nan"))], [], [])
        self.assertIsNotNone(err.location)
        self.assertIn("vertices[0]", err.location)

    def test_infinity_rejected(self):
        self._expect([("a", float("inf"))])
        self._expect([("a", float("-inf"))])

    def test_bool_value_rejected(self):
        err = self._expect([("a", True)])
        self.assertIn("bool", str(err))

    def test_rejected_value_types_are_localizable(self):
        err = self._expect([], [(("a", "b"), "soon")])
        self.assertIn("edges[0]", err.location)

    def test_bool_dimension_param_rejected(self):
        res = pp.compute_persistence([("a", 0)])
        with self.assertRaises(pp.InputError):
            res.betti_at(0, dimension=True)
        with self.assertRaises(pp.InputError):
            res.bars(True)

    def test_repeated_vertex_in_simplex_rejected(self):
        err = self._expect([(1, 0), (2, 0)], [((1, 1), 0)])
        self.assertIn("repeated vertex", str(err))
        err = self._expect(
            [(1, 0), (2, 0)],
            [((1, 2), 0)],
            [((1, 1, 2), 0)],
        )
        self.assertIn("repeated vertex", str(err))

    def test_missing_vertex_endpoint_rejected(self):
        err = self._expect([("a", 0)], [(("a", "z"), 0)])
        self.assertIn("declared vertices", str(err))

    def test_missing_face_edge_rejected(self):
        err = self._expect(
            [("a", 0), ("b", 0), ("c", 0)],
            [(("a", "b"), 0)],  # 缺 (a,c),(b,c)
            [(("a", "b", "c"), 1)],
        )
        self.assertIn("declared edges", str(err))
        self.assertIn("triangles[0]", err.location)

    def test_face_value_greater_than_simplex_rejected(self):
        # 顶点面值大于边值。
        err = self._expect([("a", 9), ("b", 9)], [(("a", "b"), 5)])
        self.assertIn("face", str(err))
        # 边面值大于三角形值。
        err = self._expect(
            [("a", 0), ("b", 0), ("c", 0)],
            [(("a", "b"), 0), (("a", "c"), 0), (("b", "c"), 4)],
            [(("a", "b", "c"), 3)],
        )
        self.assertIn("face", str(err))
        # 合法情形（面值 <= 单形值）不应被拒。
        pp.Filtration(
            [("a", 0), ("b", 0)],
            [(("a", "b"), 5)],
        )

    def test_duplicate_simplices_rejected(self):
        self._expect([("a", 0), ("a", 1)])
        self._expect(
            [("a", 0), ("b", 0)],
            [(("a", "b"), 0), (("b", "a"), 1)],
        )

    def test_limit_300_per_dimension(self):
        ok = [((i, 0.0)) for i in range(300)]
        # 300 个顶点合法且可计算。
        res = pp.Filtration(ok).compute_persistence()
        self.assertEqual(len(res.records), 300)
        # 301 个顶点在构造期被拒绝。
        with self.assertRaises(pp.InputError):
            pp.Filtration([(i, 0.0) for i in range(301)])

    def test_bad_pair_shape_rejected(self):
        self._expect([("a",)])
        self._expect([], [("ab", 0)])

    def test_unhashable_vertex_rejected(self):
        with self.assertRaises(pp.InputError):
            pp.Filtration([(["x"], 0)])

    def test_failed_construction_is_atomic(self):
        # 构造失败时调用方根本拿不到对象（无半成品）。
        with self.assertRaises(pp.InputError):
            obj = pp.Filtration(
                [("a", 0)],
                [(("a", "b"), 0)],  # b 未声明
            )
            self.assertIsNone(obj)


# --------------------------------------------------------------------------- #
# 结果结构与值的精确性
# --------------------------------------------------------------------------- #
class TestResultStructure(unittest.TestCase):
    def test_records_follow_filtration_order(self):
        res = pp.compute_persistence(
            [("b", 0), ("a", 0), ("c", 1)],
            [(("a", "b"), 1)],
        )
        seq = [(r.dimension, r.vertices, r.value) for r in res.records]
        # 值 0 的顶点先（顶点序 a,b），然后值 1：顶点 c（维数 0）先于边。
        self.assertEqual(
            seq,
            [
                (0, ("a",), 0),
                (0, ("b",), 0),
                (0, ("c",), 1),
                (1, ("a", "b"), 1),
            ],
        )
        self.assertTrue(all(r.index == i for i, r in enumerate(res.records)))

    def test_filter_values_passthrough_exactly(self):
        val = 1 / 7
        res = pp.compute_persistence([("a", val), ("b", val)], [(("a", "b"), val)])
        # 过滤值原样回填（同一对象，算法不做任何浮点运算）。
        for rec in res.records:
            self.assertIs(rec.value, val)
        # 同值零长度判定是精确比较。
        p = [q for q in res.bars(0) if not q.essential][0]
        self.assertTrue(p.zero_length)

    def test_compute_is_repeatable_and_pure(self):
        f = pp.Filtration([("a", 0), ("b", 0)], [(("a", "b"), 1)])
        r1 = f.compute_persistence()
        r2 = f.compute_persistence()
        self.assertEqual(
            [(p.birth_index, p.death_index) for p in r1.pairs],
            [(p.birth_index, p.death_index) for p in r2.pairs],
        )

    def test_int_one_equals_float_one_for_bars(self):
        res = pp.compute_persistence(
            [(0, 0), (1, 0)],
            [((0, 1), 1.0)],
        )
        self.assertEqual(res.betti_at(1), (1, 0))  # int 阈值 1 与 float 1.0 等价

    def test_record_for_lookup(self):
        res = pp.compute_persistence([("a", 0)], [])
        self.assertEqual(res.record_for(("a",)).index, 0)
        with self.assertRaises(pp.InputError):
            res.record_for(("zzz",))


# --------------------------------------------------------------------------- #
# 证书防伪造：篡改证书后核验必须失败
# --------------------------------------------------------------------------- #
class TestCertificateTamper(unittest.TestCase):
    def setUp(self):
        v, e, t = _triangle_ring_filled()
        self.res = pp.compute_persistence(v, e, t)
        self.bar = self.res.bars(1)[0]

    def test_drop_an_edge_fails(self):
        bad = pp.Certificate(
            birth_edges=self.bar.certificate.birth_edges[:-1],
            filling_triangles=self.bar.certificate.filling_triangles,
        )
        tampered = self.bar.__class__(**{**self.bar.__dict__, "certificate": bad})
        self.assertFalse(pp.verify_certificate(tampered, self.res))

    def test_extra_triangle_fails_identity(self):
        # 加入一个不属于该复形的三角形 -> 核验 False。
        bad = pp.Certificate(
            birth_edges=self.bar.certificate.birth_edges,
            filling_triangles=self.bar.certificate.filling_triangles
            + (("a", "b", "x"),),
        )
        tampered = self.bar.__class__(**{**self.bar.__dict__, "certificate": bad})
        self.assertFalse(pp.verify_certificate(tampered, self.res))


# --------------------------------------------------------------------------- #
# 确定性随机复形交叉检验：随机生成合法过滤（面值 ≤ 单形值，允许同值），
# 所有阈值 Betti 数与独立参考逐项比对；每条有限 H1 的证书独立验证链恒等式。
# --------------------------------------------------------------------------- #
class TestRandomCrossCheck(unittest.TestCase):
    @staticmethod
    def _random_complex(rng, n):
        vertices = [(i, rng.randrange(0, 3)) for i in range(n)]
        vv = dict(vertices)
        edges = []
        ev = {}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.55:
                    # 面值 ≤ 单形值，允许 +0 同值。
                    val = max(vv[i], vv[j]) + rng.randrange(0, 3)
                    edges.append(((i, j), val))
                    ev[(i, j)] = val
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (
                        (i, j) in ev
                        and (i, k) in ev
                        and (j, k) in ev
                        and rng.random() < 0.5
                    ):
                        val = max(ev[(i, j)], ev[(i, k)], ev[(j, k)]) + rng.randrange(0, 3)
                        triangles.append(((i, j, k), val))
        return vertices, edges, triangles

    def test_cross_check(self):
        rng = random.Random(20260923)
        for trial in range(25):
            n = rng.randrange(3, 10)
            raw = self._random_complex(rng, n)
            vertices, edges, triangles = raw
            res = pp.compute_persistence(vertices, edges, triangles)

            all_values = sorted(
                {v for _, v in vertices}
                | {v for _, v in edges}
                | {v for _, v in triangles}
            )
            # 在每个关键阈值（含边界上下浮动）与独立参考比对。
            probes = {all_values[0] - 1, all_values[-1] + 1}
            probes.update(all_values)
            for t in probes:
                ref = _ref_betti(*raw, t=t)
                self.assertEqual(res.betti_at(t), ref[:2], f"trial={trial} t={t}")
                self.assertEqual(res.betti_at(t, 2), ref[2], f"trial={trial} t={t}")

            # 最终连通分支数 = H0 无限条数；最终 b1/b2 与无限条一致。
            ref_final = _ref_betti(*raw, t=all_values[-1] + 1)
            _, final_b1, final_b2 = ref_final
            ess_h0 = [p for p in res.bars(0) if p.essential]
            b0_final, _ = res.betti_at(all_values[-1] + 1)
            self.assertEqual(len(ess_h0), b0_final, f"trial={trial}")
            # H1/H2 无限条在独立参考的最终 Betti 数中（未填的环/空腔）。
            ess_h1 = [p for p in res.bars(1) if p.essential]
            self.assertEqual(len(ess_h1), final_b1, f"trial={trial}")
            ess_h2 = [p for p in res.pairs if p.dimension == 2 and p.essential]
            self.assertEqual(len(ess_h2), final_b2, f"trial={trial}")

            # 每条有限 H1：独立链恒等式 + 内置核验。
            for p in res.bars(1):
                if p.essential:
                    continue
                self.assertIsNotNone(p.certificate, f"trial={trial}")
                self.assertEqual(
                    _boundary1(set(p.certificate.birth_edges)), set(),
                    f"trial={trial}",
                )
                self.assertEqual(
                    _boundary2(set(p.certificate.filling_triangles)),
                    set(p.certificate.birth_edges),
                    f"trial={trial}",
                )
                self.assertTrue(pp.verify_certificate(p, res), f"trial={trial}")
                # 出生下标严格先于死亡下标（面序保证），零长度看值。
                self.assertLess(p.birth_index, p.death_index, f"trial={trial}")
                self.assertEqual(
                    p.zero_length, p.birth_value == p.death_value, f"trial={trial}"
                )

            # 组合恒等式：每个单形要么是死亡列、要么是出生列；
            # 总条数 = 出生数 = 总单形数 - 有限配对数（H2 无限条也占一个出生）。
            n_total = len(vertices) + len(edges) + len(triangles)
            n_finite = len([p for p in res.pairs if not p.essential])
            self.assertEqual(len(res.pairs), n_total - n_finite, f"trial={trial}")
            # H0 条数恒等于顶点数；H1 条数 = 边数 - H0 有限配对数。
            n_finite_h0 = len([p for p in res.bars(0) if not p.essential])
            self.assertEqual(len(res.bars(0)), len(vertices), f"trial={trial}")
            self.assertEqual(
                len(res.bars(1)), len(edges) - n_finite_h0, f"trial={trial}"
            )
            # H2 无限条数 = 零列三角形数 = T - H1 有限配对数（2 维复形无四面体）。
            n_finite_h1 = len([p for p in res.bars(1) if not p.essential])
            self.assertEqual(
                len([p for p in res.pairs if p.dimension == 2]),
                len(triangles) - n_finite_h1,
                f"trial={trial}",
            )


# --------------------------------------------------------------------------- #
# 解析例：四面体表面（4 顶点 6 边 4 三角形，无四面体）——封闭空腔，b2=1
# --------------------------------------------------------------------------- #
class TestTetrahedronBoundaryCavity(unittest.TestCase):
    def test_cavity_gives_essential_h2(self):
        import itertools

        edges = list(itertools.combinations(range(4), 2))
        tris = list(itertools.combinations(range(4), 3))
        vertices = [(i, 0) for i in range(4)]
        edges = [(e, 1) for e in edges]
        triangles = [(t, 2) for t in tris]
        res = pp.compute_persistence(vertices, edges, triangles)

        # 表面是 S^2 的三角剖分：b0=1, b1=0, b2=1。
        self.assertEqual(res.betti_at(2), (1, 0))
        self.assertEqual(res.betti_at(2, 0), 1)
        self.assertEqual(res.betti_at(2, 1), 0)
        self.assertEqual(res.betti_at(2, 2), 1)

        h2 = [p for p in res.pairs if p.dimension == 2]
        self.assertEqual(len(h2), 1)
        self.assertTrue(h2[0].essential)
        self.assertIsNone(h2[0].certificate)
        # 6 条边中 3 条杀死 H0 组件，另 3 条约化为空并在值 2 被三角形杀死：
        # 恰 3 条有限 H1（1 -> 2），1 个三角形零列成为无限 H2。
        h1 = res.bars(1)
        self.assertEqual(len(h1), 3)
        self.assertTrue(all(not p.essential for p in h1))
        self.assertTrue(all(p.birth_value == 1 and p.death_value == 2 for p in h1))
        for p in h1:
            self.assertTrue(pp.verify_certificate(p, res))
            self.assertEqual(
                _boundary1(set(p.certificate.birth_edges)), set()
            )
            self.assertEqual(
                _boundary2(set(p.certificate.filling_triangles)),
                set(p.certificate.birth_edges),
            )
        # 阈值 1 时尚未填三角形：1 个连通体 + 3 个独立环，b2=0。
        self.assertEqual(res.betti_at(1), (1, 3))
        self.assertEqual(res.betti_at(1, 2), 0)
        # 与独立参考一致。
        raw = (vertices, edges, triangles)
        for t in (0, 1, 2, 3):
            ref = _ref_betti(*raw, t=t)
            self.assertEqual(res.betti_at(t), ref[:2], f"t={t}")
            self.assertEqual(res.betti_at(t, 2), ref[2], f"t={t}")


# --------------------------------------------------------------------------- #
# demo 冒烟测试（固定输入，真实计算；不校验耗时，规模很小）
# --------------------------------------------------------------------------- #
class TestDemoSmoke(unittest.TestCase):
    def test_demo_main_prints_success_and_rejection(self):
        import demo

        buf = StringIO()
        rc = demo.main(out=buf)
        text = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("H1", text)
        self.assertIn("InputError", text)


if __name__ == "__main__":
    unittest.main()
