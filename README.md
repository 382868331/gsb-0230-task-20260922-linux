# persistence_pairs — 带方向链证书的持久同调配对

在 GF(2) 上对顶点 / 边 / 三角形组成的过滤单纯复形计算 **H0 / H1 持久同调配对**，
并为每条**有限 H1**条给出可独立检验的“出生环链 + 死亡填充链”证书。
纯 Python 标准库实现，核心算法（边界矩阵列约化）未调用任何第三方同类库。

## 运行环境

- Python **3.14.7**（语法仅依赖 3.10+ 的注解特性，3.10 及以上应可运行，以 3.14.7 为准）
- Linux 原生运行；**无第三方依赖**，无需安装任何包、无网络、无账号/密钥、
  无数据库/中间件/云 API/Docker。
- 核心、测试、演示全部离线；示例数据由仓库内确定性代码生成。

## 准确命令

```bash
# 必要测试（38 个用例，含独立参考交叉验证）
python -m unittest discover -s tests -v

# 固定输入演示（实测约 0.04 秒，远小于 8 秒预算；无 sleep、无预录结果）
python demo.py
```

## 快速上手

```python
import persistence_pairs as pp

vertices  = [("a", 0), ("b", 0), ("c", 0)]
edges     = [(("a", "b"), 1), (("b", "c"), 1), (("c", "a"), 1)]
triangles = [(("a", "b", "c"), 2)]

res = pp.compute_persistence(vertices, edges, triangles)

for p in res.pairs:
    print(p.dimension, p.birth_value, p.death_value, p.birth_simplex, p.death_simplex)
# H0  0 inf   ('a',) None          # 无限条
# H0  0 1     ('b',) ('a','b')     # 有限条
# H0  0 1     ('c',) ('a','c')
# H1  1 2     ('b','c') ('a','b','c')

res.betti_at(1)          # (1, 1)：一个连通体 + 一个环
res.betti_at(2)          # (1, 0)：环被三角形填掉

bar = res.bars(1)[0]
bar.certificate.birth_edges          # 出生环链（边的 GF(2) 链）
bar.certificate.filling_triangles    # 死亡填充链（三角形的 GF(2) 链）
pp.verify_certificate(bar, res)      # True：∂填充链 == 出生环链
```

## 公开接口与数据结构

| 名称 | 说明 |
| --- | --- |
| `Filtration(vertices, edges, triangles)` | 校验并规范化过滤；构造失败抛 `InputError`，不产生半成品对象。`compute_persistence()` 可重复纯函数调用。 |
| `compute_persistence(vertices, edges, triangles)` | 构造 + 计算的便捷函数，返回 `PersistenceResult`。 |
| `PersistenceResult.records` | 按过滤顺序排列的 `SimplexRecord(index, dimension, vertices, value)`。 |
| `PersistenceResult.pairs` / `.bars(dim)` | 全部配对条 / 指定维数条；按出生下标升序。 |
| `PersistenceResult.betti_at(value, dimension=None)` | 阈值 Betti 数。`None` 返回 `(b0, b1)`；可传 0/1/2 取单个数。 |
| `PersistenceResult.record_for(simplex)` | 按规范化顶点元组查记录。 |
| `Pair` | `dimension, birth_index, death_index, birth_simplex, death_simplex, birth_value, death_value, essential, zero_length, certificate`。无限条 `death_index/death_simplex=None`、`death_value=math.inf`。 |
| `Certificate` | 有限 H1 条的 `birth_edges`（出生环链）与 `filling_triangles`（死亡填充链），均为按过滤下标升序的顶点元组元组。 |
| `verify_certificate(pair, result)` | 独立核验证书（成员资格、最年轻单形、环边界为 0、∂填充链 == 出生环链）。 |
| `PersistenceError` / `InputError` | 错误基类 / 输入错误；`InputError.location` 形如 `"triangles[2]"` 便于定位。 |
| `MAX_SIMPLICES_PER_DIMENSION` | `300`。 |

输入元素形式：

- `vertices`: `(vertex_id, value)`
- `edges`: `((u, v), value)`
- `triangles`: `((u, v, w), value)`

顶点 id 任意可哈希且同类间可比较（常用 `int` / `str`）；单形顶点会按升序规范化，
因此 `("b","a")` 与 `("a","b")` 视为同一个单形（重复即拒绝）。

## 输入上限与校验（错误语义）

- 顶点、边、三角形**各不超过 300 个**；超限在构造期拒绝并指明类别与实际数量。
- 过滤值只接受有限 `int` / `float`；**拒绝 `bool`、`NaN`、`+Infinity`、`-Infinity`**。
  `betti_at` 的阈值与维数参数同样拒绝 `bool`。
- 拒绝单形内重复顶点（如 `(x, x)`、`(x, x, y)`）、重复单形。
- **面必须存在**：边的端点必须是已声明顶点；三角形的三条边必须全部已声明。
- **面值不大于单形值**：每个顶点面 ≤ 边值，每条边面 ≤ 三角形值。
- 非法输入一律抛 `InputError`（`ValueError` 子类），带可定位位置，
  不做静默纠正/补全；`Filtration` 构造是原子的，失败时没有部分状态外泄，
  `compute_persistence()` 不修改其输入与已构造对象。
- 本问题无“搜索预算耗尽”情形：算法对给定输入必然终止；
  “输入无解”（非法复形）在构造期以异常明确报出，二者不存在混淆。

## 算法与设计取舍

1. **过滤顺序**：按 `(过滤值, 维数, 顶点元组字典序)` 排序。同值时顶点先于边、
   边先于三角形，同维再按顶点序；面序由“面存在且面值 ≤ 单形值”的校验保证。
2. **边界矩阵**：GF(2) 稀疏列（Python `set[int]`，对称差即加法），
   边的边界为两个端点，三角形的边界为三条边。
3. **低位 pivot 列约化**：维护 `R = D·V`（`V` 初始为单位阵，对角为 1、上三角）。
   列 `j` 以 `low(j)=i` 收尾时配对 `(i, j)`（`i` 出生、`j` 死亡）；
   约化为空且从未充当 `low` 的列为无限条。
4. **条的维数**：有限条维数 = 死亡单形维数 − 1（边杀死 H0，三角形杀死 H1）；
   无限条维数 = 出生单形维数（含可能的 H2 空腔条）。
5. **证书（方向链）**：有限 H1 配对 `(i, j)` 取
   - 出生环链 `R[j]`：下标上界恰为出生边 `i` 的边链，含出生边，边界为 0；
   - 死亡填充链 `V[j]`：使 `∂(V[j]) = R[j]` 成立的三角形链，含死亡三角形 `j`。
   二者均以单形顶点元组输出，可用任意独立 GF(2) 链运算复核，不必信任本库。
6. **零长度条保留并显式标记**（`zero_length=True`），不删除；它们在任何阈值的
   Betti 数中都不存活（区间左闭右开 `[birth, death)`）。
7. **Betti 数**：直接由区间计数得到，`b_k(t) = #{bar in H_k : birth ≤ t < death}`，
   无限条只比较出生值。`betti_at(..., 2)` 额外给出空腔数 b2。
8. 规模小（≤ 900 单形）且一次性计算，选择直接的集合对称差实现，
   换取可读、可核验与零依赖；不做压缩同调/分块等优化。

## 浮点结果与误差口径

- **核心算法不执行任何浮点算术**。过滤值只用于排序与比较，并原样回填到
  `Pair` / `SimplexRecord` 中（同一对象，不复制、不换算），区间端点与输入逐位相同。
- **不存在 epsilon/容差**：零长度判定、面值 ≤ 单形值校验、阈值存活判定全部是
  Python 精确比较。按 Python 语义，`1` 与 `1.0` 相等且排序一致（二者视为同值，
  决胜再按维数/顶点序）；若需要严格区分 `1` 与 `1.0`，应在调用前自行编码。
- 若输入值彼此相等但并非同一数值表示（例如 `0.1+0.2` 与 `0.3`），按其真实比较
  结果处理——本库不会用容差把它们“拉平”，也不会因此产生任何数值漂移。

## 测试的独立性说明

- 解析例（三角环+填面、不填环、同值决胜、分离组件、扇形圆盘、四面体表面空腔）
  的期望配对为手算。
- 阈值 Betti 数与**独立参考** `_ref_betti` 逐项比对：该参考用并查集算 b0、
  用整数位掩码 GF(2) 主元消元自行计算 rank ∂2 得到 b1/b2，
  **不导入也不调用被测核心**；被测核心不用于生成期望值。
- 25 个固定随机种子（`random.Random(20260923)`）生成的合法复形上交叉验证
  所有关键阈值的 b0/b1/b2，并对每条有限 H1 用独立链运算复核
  “出生环边界为 0”与“∂填充链 == 出生环链”。
- 与被测实现的往返一致性仅作补充断言（组合计数恒等式、证书防篡改）。

## 已知限制 / 非目标

- 仅实现本题契约：维数 0/1/2 的单纯复形、GF(2) 系数、一次性批处理。
  **不宣称**兼容任意持久同调行业标准格式（如 Dipha/GUDHI 文件协议）、
  不含广义同调、上同调、Z/p（p≠2）系数、任意维单形或流式增量 API。
- 顶点 id 必须同类间可比较（混用互不可比的类型会在排序时收到 `InputError`）。
- 上限 300/类是接口边界而非性能极限；实测 300+300+300 约 10 ms 完成，
  但超过 300 一律拒绝。
- demo 只展示一个正常结果（环出生/死亡 + 证书核验）与一个真实触发的拒绝
  （缺面三角形）；不做压力测试、不是 CLI 产品。

## 文件清单

- `persistence_pairs.py` — 核心库（校验、列约化、配对、证书、Betti 计数）
- `tests/test_persistence_pairs.py` — 必要测试（解析例 + 独立参考 + 交叉验证）
- `demo.py` — 固定输入演示
- `README.md` — 本文件
