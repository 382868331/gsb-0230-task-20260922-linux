#!/usr/bin/env python3
"""固定输入演示：三角环出生、填面死亡；并实际触发一次输入拒绝。

运行：``python demo.py``（约毫秒级，远小于 8 秒预算；纯标准库、完全离线）。
所有数据由本文件确定性地给出，无随机数、无网络、无预录计算结果——
打印的配对与证书均由 persistence_pairs 当场计算。
"""

from __future__ import annotations

import sys

import persistence_pairs as pp


def _print_header(out, title):
    bar = "=" * 64
    print(bar, file=out)
    print(title, file=out)
    print(bar, file=out)


def main(out=None) -> int:
    out = out if out is not None else sys.stdout

    # ------------------------------------------------------------------ #
    # 正常结果：三个顶点(值0) -> 三条边成环(值1) -> 一个三角形填面(值2)
    # ------------------------------------------------------------------ #
    _print_header(out, "[1] 正常计算：三角环在值 1 出生，在值 2 被填面杀死")
    vertices = [("a", 0), ("b", 0), ("c", 0)]
    edges = [
        (("a", "b"), 1),
        (("b", "c"), 1),
        (("c", "a"), 1),
    ]
    triangles = [(("a", "b", "c"), 2)]

    result = pp.compute_persistence(vertices, edges, triangles)

    print("过滤顺序（值, 维数, 单形）：", file=out)
    for r in result.records:
        print(f"  #{r.index:<2} value={r.value}  dim={r.dimension}  {r.vertices}",
              file=out)

    print("\n持久配对：", file=out)
    for p in result.pairs:
        end = "inf" if p.essential else repr(p.death_value)
        tag = "  [零长度条-保留]" if p.zero_length else ""
        print(
            f"  H{p.dimension}: {p.birth_value!r:>4} -> {end:>4}   "
            f"birth={p.birth_simplex} "
            f"death={p.death_simplex if p.death_simplex is not None else '-'}"
            f"{tag}",
            file=out,
        )

    print("\n逐阈值 Betti 数（独立区间计数）：", file=out)
    for t in (0, 1, 2):
        b0, b1 = result.betti_at(t)
        print(f"  t={t}:  b0={b0}, b1={b1}", file=out)

    h1 = result.bars(1)[0]
    cert = h1.certificate
    print("\nH1 有限条的可检验证书：", file=out)
    print(f"  出生环链 birth_edges      = {list(cert.birth_edges)}", file=out)
    print(f"  死亡填充链 filling_triangles = {list(cert.filling_triangles)}",
          file=out)
    ok = pp.verify_certificate(h1, result)
    print(f"  verify_certificate: {ok}  (GF(2) 恒等式 ∂填充链 == 出生环链)",
          file=out)
    assert ok, "证书核验必须通过"

    # 同例的“环不填”对照：同一条 H1 变为无限条（death = inf）。
    open_ring = pp.compute_persistence(vertices, edges)
    only = open_ring.bars(1)[0]
    print(f"\n对照：去掉填面后 H1 成为无限条 essential={only.essential}, "
          f"death={only.death_value}", file=out)

    # ------------------------------------------------------------------ #
    # 异常边界：缺面三角形（边 (a,c) 未声明）必须被当场拒绝
    # ------------------------------------------------------------------ #
    _print_header(out, "[2] 拒绝边界：三角形缺少已声明的面边")
    bad_edges = [(("a", "b"), 1), (("b", "c"), 1)]  # 缺 (c, a)
    print("输入：三角形 ('a','b','c') 引用了未声明的边 ('a','c')", file=out)
    try:
        pp.Filtration(vertices, bad_edges, triangles)
    except pp.InputError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}", file=out)
        print("  （构造失败即拒绝，不产生任何半成品对象/部分结果）", file=out)
    else:  # pragma: no cover - 防御：不应发生
        print("  错误：非法输入未被拒绝", file=out)
        return 1

    print("\n演示完成：以上配对、Betti 数与证书均为现场真实计算。", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
