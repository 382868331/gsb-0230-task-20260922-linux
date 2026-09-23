"""persistence_pairs — 带方向链证书的持久同调配对（GF(2)，维数 0/1/2）。

本模块是一个自包含的本题库，仅依赖 Python 3.10+ 标准库（开发环境 3.14.7）。

公开接口
--------
* :class:`Filtration`              —— 校验并规范化过滤（复形 + 过滤值）
* :func:`compute_persistence`      —— 一次性计算的便捷函数
* :class:`PersistenceResult`       —— 过滤记录、配对条、按阈值 Betti 数
* :class:`Pair` / :class:`Certificate` / :class:`SimplexRecord`
* :func:`verify_certificate`       —— 对 H1 出生环/死亡填充证书做独立核验
* :class:`PersistenceError` / :class:`InputError`

数学约定
--------
* 系数域 GF(2)，链 = 单形下标集合（对称差即加法）。
* 过滤顺序：(过滤值, 维数, 顶点元组字典序)。同值时顶点先于边、边先于三角形；
  面必须先于单形出现（由校验“面存在且面值 ≤ 单形值”保证）。
* 边界矩阵 D 的列用低位 pivot 列约化：R = D·V，V 对角为 1 的上三角阵。
  以 low(j)=i 收尾的列 j 给出有限配对 (i, j)；约化为空且从未作为 low 的
  列 i 给出无限条（death = inf）。
* 对有限 H1 配对（出生边 i，死亡三角形 j）：
  - 出生环链 birth_cycle = R[j]（边链，下标上界恰为 i，含出生边，边界为 0）；
  - 死亡填充链 filling = V[j]（三角形链，含死亡三角形 j）；
  - 可检验恒等式：∂ filling = birth_cycle（GF(2)）。

误差口径
--------
算法不做任何浮点运算：过滤值只参与排序/比较并原样回填到结果中。区间端点与
输入值逐位相同；零长度判定、阈值查询均为精确比较（int 1 与 float 1.0 按相等
处理，与排序一致）。模块内不存在 epsilon/容差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

__all__ = [
    "MAX_SIMPLICES_PER_DIMENSION",
    "PersistenceError",
    "InputError",
    "SimplexRecord",
    "Certificate",
    "Pair",
    "PersistenceResult",
    "Filtration",
    "compute_persistence",
    "verify_certificate",
]

#: 接口上限：顶点 / 边 / 三角形各不超过 300 个。
MAX_SIMPLICES_PER_DIMENSION = 300

_VERTICES = "vertices"
_EDGES = "edges"
_TRIANGLES = "triangles"


# --------------------------------------------------------------------------- #
# 错误类型
# --------------------------------------------------------------------------- #
class PersistenceError(ValueError):
    """本模块所有错误的基类（同时是 ValueError）。"""


class InputError(PersistenceError):
    """非法输入。:attr:`location` 给出可定位位置，例如 ``"triangles[2]"``。"""

    def __init__(self, message: str, location: Optional[str] = None) -> None:
        self.location = location
        text = message if location is None else f"{location}: {message}"
        super().__init__(text)


# --------------------------------------------------------------------------- #
# 公开数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimplexRecord:
    """过滤序列中的一个单形。

    ``vertices`` 为规范化顶点元组：顶点 ``(v,)``、边 ``(u, v)``（升序）、
    三角形 ``(u, v, w)``（升序）。
    """

    index: int
    dimension: int
    vertices: Tuple[Any, ...]
    value: Any


@dataclass(frozen=True)
class Certificate:
    """有限 H1 配对的方向链证书（GF(2) 链，以单形元组表示）。

    * ``birth_edges``       —— 出生环链（边的集合，按下标升序）；
    * ``filling_triangles`` —— 死亡填充链（三角形的集合，按下标升序，含死亡三角形）。

    可检验关系：∂(filling_triangles) == birth_edges，且 birth_edges 自身边界为 0。
    """

    birth_edges: Tuple[Tuple[Any, Any], ...]
    filling_triangles: Tuple[Tuple[Any, Any, Any], ...]


@dataclass(frozen=True)
class Pair:
    """一条持久配对（区间条）。

    * ``dimension``  —— 同调维数（0 或 1）；
    * ``birth_*``    —— 出生单形（列约化为空的单形）的下标/元组/值；
    * ``death_*``    —— 死亡单形（取得低位 pivot 的列）的下标/元组/值；
                        无限条为 ``None`` / ``None`` / ``math.inf``；
    * ``essential``  —— 是否无限条；
    * ``zero_length``—— 出生值与死亡值是否精确相等（保留并显式标记，不删除）；
    * ``certificate``—— 有限 H1 的链证书；其余为 ``None``。
    """

    dimension: int
    birth_index: int
    death_index: Optional[int]
    birth_simplex: Tuple[Any, ...]
    death_simplex: Optional[Tuple[Any, ...]]
    birth_value: Any
    death_value: Any
    essential: bool
    zero_length: bool
    certificate: Optional[Certificate]


class PersistenceResult:
    """持久同调计算结果（由 :class:`Filtration` 产生，视为只读）。"""

    def __init__(
        self,
        records: Sequence[SimplexRecord],
        pairs: Sequence[Pair],
    ) -> None:
        self._records: Tuple[SimplexRecord, ...] = tuple(records)
        self._pairs: Tuple[Pair, ...] = tuple(pairs)
        self._by_tuple: Dict[Tuple[Any, ...], SimplexRecord] = {
            r.vertices: r for r in self._records
        }

    # ------------------------------------------------------------------ #
    @property
    def records(self) -> Tuple[SimplexRecord, ...]:
        """按过滤顺序排列的全部单形记录。"""
        return self._records

    @property
    def pairs(self) -> Tuple[Pair, ...]:
        """全部配对条，按出生下标升序。"""
        return self._pairs

    def bars(self, dimension: Optional[int] = None) -> Tuple[Pair, ...]:
        """返回指定维数的配对条（``dimension=None`` 时返回全部）。"""
        if dimension is not None:
            if isinstance(dimension, bool) or dimension not in (0, 1, 2):
                raise InputError(
                    "dimension must be one of 0, 1, 2 (bool is rejected)",
                    "dimension",
                )
        if dimension is None:
            return self._pairs
        return tuple(p for p in self._pairs if p.dimension == dimension)

    def record_for(self, simplex: Tuple[Any, ...]) -> SimplexRecord:
        """按规范化顶点元组取过滤记录；不存在则抛 :class:`InputError`。"""
        try:
            return self._by_tuple[tuple(simplex)]
        except (TypeError, KeyError):
            raise InputError(f"simplex {simplex!r} not in filtration") from None

    def betti_at(self, value: Any, dimension: Optional[int] = None) -> Any:
        """阈值 ``value`` 下的 Betti 数（包含所有过滤值 ≤ value 的单形）。

        区间按左闭右开 ``[birth, death)`` 计数：条在 ``birth_value ≤ value <
        death_value`` 时存活；无限条只比较出生值。零长度条在任何阈值都不存活。
        ``dimension=None`` 时返回 ``(b0, b1)``（H2 空腔不计入元组），
        否则可传 0、1、2 返回对应整数。
        ``value`` 同样拒绝 bool / NaN / Infinity。
        """
        _check_value(value, "value")
        counts = [0, 0, 0]
        for p in self._pairs:
            alive = p.birth_value <= value and (
                p.essential or value < p.death_value
            )
            if alive:
                counts[p.dimension] += 1
        if dimension is None:
            return counts[0], counts[1]
        if isinstance(dimension, bool) or dimension not in (0, 1, 2):
            raise InputError(
                "dimension must be 0, 1 or 2 (bool is rejected)", "dimension"
            )
        return counts[dimension]

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"PersistenceResult(simplices={len(self._records)}, "
            f"pairs={len(self._pairs)})"
        )


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _check_value(value: Any, location: str) -> None:
    # bool 是 int 的子类，必须先排除。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(
            "filter value must be int or float (bool is rejected)", location
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise InputError(
            "filter value must be finite (NaN and +/-Infinity are rejected)",
            location,
        )


def _materialize(raw: Any, kind: str) -> list:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise InputError(f"{kind} must be an iterable of (simplex, value) pairs")
    items = list(raw)
    if len(items) > MAX_SIMPLICES_PER_DIMENSION:
        raise InputError(
            f"at most {MAX_SIMPLICES_PER_DIMENSION} {kind} allowed, got {len(items)}",
            kind,
        )
    return items


def _as_vertex_tuple(simplex: Any, location: str, arity: int) -> Tuple[Any, ...]:
    if not isinstance(simplex, (tuple, list)) or len(simplex) != arity:
        raise InputError(
            f"simplex must be a {arity}-element tuple/list of vertex ids", location
        )
    # 可哈希
    seen: Set[Any] = set()
    for v in simplex:
        try:
            if v in seen:
                raise InputError(
                    "repeated vertex id inside one simplex is rejected", location
                )
            seen.add(v)
        except TypeError:
            raise InputError("vertex ids must be hashable", location) from None
    # 可排序（用于规范化与同值决胜）
    try:
        return tuple(sorted(simplex))
    except TypeError:
        raise InputError(
            "vertex ids must be mutually comparable for tie-breaking ordering",
            location,
        ) from None


# --------------------------------------------------------------------------- #
# Filtration
# --------------------------------------------------------------------------- #
class Filtration:
    """经过校验的单纯复形过滤。

    参数（均可为列表或元组，元素为 ``(单形, 过滤值)``）：

    * ``vertices``  —— ``(vid, value)``
    * ``edges``     —— ``((u, v), value)``，u ≠ v
    * ``triangles`` —— ``((u, v, w), value)``，三个顶点互异

    构造时一次性完成全部校验；任何一项不合法都抛 :class:`InputError`，
    不会产生半成品对象。校验规则：

    1. 过滤值仅限 int/float 且有限；拒绝 bool、NaN、±Infinity；
    2. 每类单形至多 300 个；顶点 id 可哈希、同类 id 间可比较；
    3. 拒绝单形内重复顶点、重复单形；
    4. 边的端点必须是已声明顶点；三角形的三条边必须全部已声明；
    5. 每个面的过滤值必须 ≤ 单形过滤值。
    """

    def __init__(
        self,
        vertices: Iterable = (),
        edges: Iterable = (),
        triangles: Iterable = (),
    ) -> None:
        v_items = _materialize(vertices, _VERTICES)
        e_items = _materialize(edges, _EDGES)
        t_items = _materialize(triangles, _TRIANGLES)

        # --- 解析顶点 -------------------------------------------------- #
        v_value: Dict[Any, Any] = {}
        for pos, item in enumerate(v_items):
            loc = f"{_VERTICES}[{pos}]"
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise InputError("expected (vertex_id, value) pair", loc)
            vid, val = item
            _check_value(val, loc)
            try:
                if vid in v_value:
                    raise InputError(f"duplicate vertex id {vid!r}", loc)
                v_value[vid] = val
            except TypeError:
                raise InputError("vertex ids must be hashable", loc) from None

        # --- 解析边 ---------------------------------------------------- #
        e_value: Dict[Tuple[Any, Any], Any] = {}
        e_loc: Dict[Tuple[Any, Any], str] = {}
        for pos, item in enumerate(e_items):
            loc = f"{_EDGES}[{pos}]"
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise InputError("expected ((u, v), value) pair", loc)
            raw_edge, val = item
            _check_value(val, loc)
            edge = _as_vertex_tuple(raw_edge, loc, 2)
            if edge in e_value:
                raise InputError(f"duplicate edge {edge!r}", loc)
            for v in edge:
                if v not in v_value:
                    raise InputError(
                        f"endpoints must be declared vertices; {v!r} is not", loc
                    )
                if not v_value[v] <= val:
                    raise InputError(
                        f"face vertex {v!r} has value {v_value[v]!r} > edge "
                        f"value {val!r}",
                        loc,
                    )
            e_value[edge] = val
            e_loc[edge] = loc

        # --- 解析三角形 ------------------------------------------------ #
        t_value: Dict[Tuple[Any, Any, Any], Any] = {}
        for pos, item in enumerate(t_items):
            loc = f"{_TRIANGLES}[{pos}]"
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise InputError("expected ((u, v, w), value) pair", loc)
            raw_tri, val = item
            _check_value(val, loc)
            tri = _as_vertex_tuple(raw_tri, loc, 3)
            if tri in t_value:
                raise InputError(f"duplicate triangle {tri!r}", loc)
            u, v, w = tri
            for a, b in ((u, v), (u, w), (v, w)):
                face = (a, b)
                if face not in e_value:
                    raise InputError(
                        f"all faces must be declared edges; {face!r} is not", loc
                    )
                if not e_value[face] <= val:
                    raise InputError(
                        f"face edge {face!r} has value {e_value[face]!r} > "
                        f"triangle value {val!r}",
                        loc,
                    )
            t_value[tri] = val

        # --- 规范化排序 ------------------------------------------------ #
        raw_records = (
            [(0, (v,), val) for v, val in v_value.items()]
            + [(1, e, val) for e, val in e_value.items()]
            + [(2, t, val) for t, val in t_value.items()]
        )
        try:
            raw_records.sort(key=lambda r: (r[2], r[0], r[1]))
        except TypeError:
            # 不同顶点 id 之间不可比较（例如混用不可互比类型）时在此暴露。
            raise InputError(
                "vertex ids must be mutually comparable so that equal-value "
                "simplices can be ordered by vertex order"
            ) from None

        records = tuple(
            SimplexRecord(index=i, dimension=d, vertices=verts, value=val)
            for i, (d, verts, val) in enumerate(raw_records)
        )
        self._records: Tuple[SimplexRecord, ...] = records
        self._index_of: Dict[Tuple[Any, ...], int] = {
            r.vertices: r.index for r in records
        }

    # ------------------------------------------------------------------ #
    @property
    def records(self) -> Tuple[SimplexRecord, ...]:
        return self._records

    def compute_persistence(self) -> PersistenceResult:
        """运行低位 pivot 列约化，返回 :class:`PersistenceResult`。

        纯函数：不修改本对象，可重复调用，每次独立计算。
        """
        return _reduce(self._records, self._index_of)


def compute_persistence(
    vertices: Iterable = (),
    edges: Iterable = (),
    triangles: Iterable = (),
) -> PersistenceResult:
    """构造 :class:`Filtration` 并立即计算的便捷函数。"""
    return Filtration(vertices, edges, triangles).compute_persistence()


# --------------------------------------------------------------------------- #
# 核心：GF(2) 边界矩阵 + 低位 pivot 列约化（R = D V）
# --------------------------------------------------------------------------- #
def _boundary_column(rec: SimplexRecord, index_of: Dict[Tuple[Any, ...], int]) -> Set[int]:
    if rec.dimension == 0:
        return set()
    if rec.dimension == 1:
        return {index_of[(v,)] for v in rec.vertices}
    u, v, w = rec.vertices
    return {
        index_of[(u, v)],
        index_of[(u, w)],
        index_of[(v, w)],
    }


def _reduce(
    records: Tuple[SimplexRecord, ...],
    index_of: Dict[Tuple[Any, ...], int],
) -> PersistenceResult:
    n = len(records)

    # R[j]：约化后的列（GF(2) 稀疏链，用下标集合表示）。
    R: list[Set[int]] = [_boundary_column(rec, index_of) for rec in records]
    # V[j]：使 R = D·V 成立的右乘链；初始 V = I，故 V[j] = {j}。
    V: list[Set[int]] = [{j} for j in range(n)]

    pivot_column_of_row: Dict[int, int] = {}

    for j in range(n):
        while R[j]:
            low = max(R[j])
            owner = pivot_column_of_row.get(low)
            if owner is None:
                # 列 j 以 pivot low 收尾：low 出生，j 死亡，二者配对。
                pivot_column_of_row[low] = j
                break
            R[j] ^= R[owner]
            V[j] ^= V[owner]
        # 列约化为空 => j 是出生列（可能在别处作为 low 被有限配对，
        # 否则是无限条）。

    paired_rows: Set[int] = set()
    finite_pairs: list[Tuple[int, int]] = []
    for j, col in enumerate(R):
        if col:
            low = max(col)
            paired_rows.add(low)
            finite_pairs.append((low, j))

    pairs: list[Pair] = []

    # 有限条（含零长度条：保留并显式标记）。
    for birth_i, death_j in finite_pairs:
        birth_rec = records[birth_i]
        death_rec = records[death_j]
        dim = death_rec.dimension - 1
        cert: Optional[Certificate] = None
        if death_rec.dimension == 2:
            cert = Certificate(
                birth_edges=tuple(
                    records[i].vertices for i in sorted(R[death_j])
                ),
                filling_triangles=tuple(
                    records[i].vertices for i in sorted(V[death_j])
                ),
            )
        pairs.append(
            Pair(
                dimension=dim,
                birth_index=birth_i,
                death_index=death_j,
                birth_simplex=birth_rec.vertices,
                death_simplex=death_rec.vertices,
                birth_value=birth_rec.value,
                death_value=death_rec.value,
                essential=False,
                zero_length=birth_rec.value == death_rec.value,
                certificate=cert,
            )
        )

    # 无限条：列约化为空，且从未作为任何列的 low。
    for i in range(n):
        if not R[i] and i not in paired_rows:
            rec = records[i]
            pairs.append(
                Pair(
                    dimension=rec.dimension,
                    birth_index=i,
                    death_index=None,
                    birth_simplex=rec.vertices,
                    death_simplex=None,
                    birth_value=rec.value,
                    death_value=math.inf,
                    essential=True,
                    zero_length=False,
                    certificate=None,
                )
            )

    pairs.sort(key=lambda p: p.birth_index)
    return PersistenceResult(records, pairs)


# --------------------------------------------------------------------------- #
# 证书核验
# --------------------------------------------------------------------------- #
def _parity(elements: Iterable[Any]) -> Set[Any]:
    out: Set[Any] = set()
    for x in elements:
        if x in out:
            out.discard(x)
        else:
            out.add(x)
    return out


def verify_certificate(pair: Pair, result: PersistenceResult) -> bool:
    """核验有限 H1 配对的链证书。全部满足返回 ``True``，否则 ``False``：

    1. ``pair`` 是有限 H1 条且带证书；
    2. 证书中的边/三角形都存在于过滤中，无重复、无越界单形；
    3. 出生环含出生边，最年轻边恰为出生边（下标上界 = birth_index），
       且边界为 0（是环）；
    4. 填充链含死亡三角形，最年轻三角形恰为死亡单形；
    5. GF(2) 恒等式 ∂(filling_triangles) == birth_edges 成立。
    """
    try:
        if pair.dimension != 1 or pair.essential or pair.certificate is None:
            return False
        cert = pair.certificate
        by_tuple = result._by_tuple  # 只读访问
        if pair.birth_simplex not in by_tuple or pair.death_simplex not in by_tuple:
            return False
        if by_tuple[pair.birth_simplex].index != pair.birth_index:
            return False
        if by_tuple[pair.death_simplex].index != pair.death_index:
            return False

        edge_recs = []
        for e in cert.birth_edges:
            rec = by_tuple.get(e if isinstance(e, tuple) else tuple(e))
            if rec is None or rec.dimension != 1:
                return False
            edge_recs.append(rec)
        tri_recs = []
        for t in cert.filling_triangles:
            rec = by_tuple.get(t if isinstance(t, tuple) else tuple(t))
            if rec is None or rec.dimension != 2:
                return False
            tri_recs.append(rec)

        edge_indices = {r.index for r in edge_recs}
        tri_indices = {r.index for r in tri_recs}
        if len(edge_indices) != len(edge_recs) or len(tri_indices) != len(tri_recs):
            return False

        # 出生环含出生边，且不含更年轻的边。
        if pair.birth_index not in edge_indices:
            return False
        if max(edge_indices) != pair.birth_index:
            return False
        # 填充链含死亡三角形，且不含更年轻之外的三角形（≤ death 由最大元保证）。
        if pair.death_index not in tri_indices or max(tri_indices) != pair.death_index:
            return False

        # 出生环自身边界为 0。
        cycle_vertices = _parity(v for e in cert.birth_edges for v in e)
        if cycle_vertices:
            return False

        # ∂(填充三角形链) == 出生环链。
        filled_edges = _parity(
            edge
            for (u, v, w) in cert.filling_triangles
            for edge in ((u, v), (u, w), (v, w))
        )
        if filled_edges != set(cert.birth_edges):
            return False
        return True
    except Exception:
        return False
