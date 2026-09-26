# pershom — 带方向链证书的持久同调配对（GF(2)，H0/H1）

离线、确定性、仅依赖 Python 标准库的持久同调计算模块：对顶点/边/三角形
过滤复形，在 **GF(2)** 上建立边界矩阵，用标准**最低位 pivot**（lowest-one /
bottom pivot）列约化给出 H0、H1 的出生–死亡配对，并为**每条有限 H1 条**
同时交付可独立检验的**出生环 1-链**与**死亡填充 2-链**证书。

增量能力（v1.1）：在**固定过滤截面**上提供 **H1xH1 -> H2 杯积**查询：
先求该截面的上同调基（H1、H2）与余边界空间，再用顶点全序下的单形
Alexander-Whitney 公式求两个 1-cocycle 的乘积，并把原始 2-cochain
约化为 H2 基下的坐标；不是只比较 Betti 数。

- Python 版本：**Python 3.14.7**（Linux 原生验证；任何 3.10+ 理论上可运行，
  但只按 3.14.7 交付与验收）
- 第三方依赖：**无**。安装 Python 后直接运行，不需要 pip install、网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。
- 全部计算离线、确定性；示例数据由仓库内代码确定性生成，无外部语料。

## 目录

```
pershom/             被测库（核心算法从零实现）
  __init__.py        公开接口
  persistence.py     输入校验、边界矩阵、pivot 约化、配对、证书与验证
  cohomology.py      截面上同调空间 + Alexander-Whitney 杯积（GF(2)）
tests/
  reference.py       独立参考：GF(2) 高斯消元 + 秩不变量 + 并查集 + 杯积（不 import pershom）
  test_persistence.py
  test_cohomology.py
demo.py              固定输入演示（正常结果 + 一个真实触发的拒绝）
demo_iteration.py    杯积演示：环面 vs 楔合（同 Betti、不同乘法）
```

## 环境准备（隔离 venv）

不需要任何第三方包；venv 仅用于隔离，非必需：

```bash
python3.14 -m venv .venv        # 或任何指向 Python 3.14.7 的解释器
. .venv/bin/activate
# 无需 pip install：requirements.txt 只有注释，标准库即可
```

## 运行命令

```bash
# 必要测试（验收命令）
python -m unittest discover -s tests -v

# 演示（约 0.05 秒，远小于 8 秒预算；真实计算，无 sleep、无预录结果）
python demo.py

# 杯积增量演示（真实离线计算，实测约 0.1 秒，远低于 90 秒预算）
python demo_iteration.py
```

## 公开接口

```python
from pershom import compute_persistence, check_certificate, INF

result = compute_persistence(simplices)
```

### 输入

`compute_persistence(simplices)` 接受一个可迭代对象，每项为二元组
`(vertices, filtration)`：

- `vertices`：长度 1–3 的元组/列表，元素为**非负 int**，必须**严格递增**
  （即 0-单形 `(v,)`、1-单形 `(a,b)` 且 a<b、2-单形 `(a,b,c)` 且 a<b<c）。
  库**不会**静默重排；顺序错误直接拒绝。
- `filtration`：有限的 `int` 或 `float`，原样保存、原样比较与回显。
- 复形闭合：每个边的两个顶点面、每个三角形的三条边面与顶点面都必须在
  输入中出现，且面过滤值 `<=` 单形过滤值；否则拒绝。
- 同一单形不得重复给出；单形内部不得有重复顶点。
- **上限：至多 300 个单形**（顶点/边/三角形合计）。第 301 项被拒绝。
- 维数仅支持 0、1、2；不接受空顶点序列或 4 个及以上顶点。

### 全序与“同值”规则

过滤总序为 `(filtration, dimension, vertices 字典序)`，即：

1. 过滤值升序；
2. 同值时按维数 0 → 1 → 2；
3. 再同则按顶点元组升序（即顶点 id 序）。

输入顺序不影响结果；输出 `result.cells` 的下标即矩阵列/行号。

### 输出

- `result.cells`：`Cell(index, dimension, vertices, filtration)` 元组，
  按全序排列。
- `result.bars` / `result.h0_bars` / `result.h1_bars`：`Bar`：
  - `dimension`、`birth_index`、`death_index`（本质类为 `None`）；
  - `birth`、`death`（无死亡时 `death == pershom.INF`，即 `float("inf")`）；
  - `zero_length`：`birth == death` 的条**保留并标记**，不删除；
  - 区间语义为半开 **`[birth, death)`**：出生时刻存活，死亡时刻不存活；
  - `birth_chain` / `death_chain`：仅**有限 H1** 条非空，为 GF(2) 链
    （`cells` 下标元组，表示所列单形的对称差/模 2 和）。H0 条与本质 H1
    条的两个字段为 `None`。
- `result.betti_at(t) -> (beta0, beta1)`：过滤值 `<= t` 的子复形 Betti 数。
- `result.betti_profile(thresholds=None)`：默认在每个不同过滤值处取样。
- `result.chain_vertices(chain)`：把链下标翻译成顶点元组列表，便于检验。
- `check_certificate(result, bar) -> True`：用**只依赖单元顶点数据**的
  组合检查复核证书，失败抛 `CertificateError`。计算内部返回前也会对每条
  有限 H1 自动执行该校验（300 单形规模下成本可忽略）。

### H1 链证书的精确含义

对有限 H1 条（出生边列 i，死亡三角形列 j）：

- `birth_chain` 是一条非空 GF(2) 边环：每个顶点在链中出现偶数次，即
  模 2 边界为 0；出生边是环中过滤序**最晚**（下标最大）的边（它就是
  死亡三角形约化列的最低 pivot 行）；环上所有边在死亡时刻均已存在。
- `death_chain` 是非空三角形模 2 和，包含死亡三角形本身，且只使用死亡
  列之前（含）的三角形；其边界（三角形边界的对称差）与 `birth_chain`
  **逐边相等**（按集合精确比较，无任何容差）。

即：`∂(death_chain) = birth_chain`、`∂(birth_chain) = 0`。测试中的
`tests/reference.py` 用第二份独立实现重复该检验，并额外把整条条形码与
“秩不变量四拐角公式 + 独立 GF(2) 高斯消元 + 并查集 β0”逐一比对。

## 截面上同调与杯积（v1.1 增量）

```python
from pershom import cohomology_slice, cup_product

sl = cohomology_slice(simplices, t)     # simplices 与 compute_persistence 同一契约
res = cup_product(sl, cocycle_a, cocycle_b)
```

### cohomology_slice(simplices, threshold)

- 输入校验复用 compute_persistence 的同一解析器（同一批 reason）；
  threshold 须为有限 int/float（否则 bad_threshold）。
- 截面 = 过滤值 <= threshold 的子复形；输出 CohomologySlice：
  - vertices / edges / triangles：截面单元（按顶点元组排序）；
  - h1_basis / h2_basis：H1、H2 的 GF(2) 基，每个基元是一个
    cocycle（边/三角形的元组）；b0 / b1 / b2 为截面 Betti 数；
  - 内部同时计算余边界空间 B1 = im δ0、B2 = im δ1（GF(2) 消元），
    杯积坐标即“模 B2”的商坐标。

### cup_product(slice, left, right)

- left / right：1-cocycle，给法为顶点对可迭代对象，如
  [(0, 3), (1, 4)]；对内顺序无所谓，但必须都是截面内的边且
  **GF(2) 闭合**（在每个三角形上余边界为 0），否则拒绝。
- 乘积用顶点全序（顶点 id 升序）下的 Alexander-Whitney 公式：
  对三角形 (u, v, w)（u<v<w），(a⌣b)([u,v,w]) = a([u,v])·b([v,w])。
- 返回 CupProductResult：
  - left / right：规范化后的两个输入 cocycle；
  - raw：原始 2-cochain（取值为 1 的三角形元组）；
  - coordinates：乘积类在 slice.h2_basis 下的 GF(2) 坐标；
  - is_zero：乘积类是否为 0（即 raw 是否落在 B2 内）。
- 闭性在求积前逐条检查；给任一输入加上余边界（顶点星形 δ0）后
  坐标不变——测试对此逐顶点验证。
- 顶点重标号会改变基与代表元，但乘积矩阵的**零/非零格局与秩**不变
  （测试用随机置换验证）。

### 杯积新增 reason

| reason | 触发条件 |
|---|---|
| bad_threshold | 截面阈值不是有限数 |
| bad_cocycle | cocycle 不是顶点对可迭代对象（bool 顶点同样拒绝） |
| edge_not_in_slice | cocycle 用了截面中不存在的边 |
| cocycle_not_closed | 余边界在某个三角形上非零（报错定位到该三角形） |

### 验证方式

- tests/test_cohomology.py：7 顶点环面三角化（betti (1,2,1)，交形式
  [[0,1],[1,0]]，乘积非零）与楔合 S1∨S1∨S2（同样 betti (1,2,1)，乘积
  全零）对照；每个乘积都与 tests/reference.py 中**独立的集合版
  GF(2) 高斯消元**（不 import pershom）逐对交叉验证 raw 与零/非零。
- 原持久配对接口（compute_persistence、check_certificate 等）
  完全保留，基线测试原样通过。

## 错误语义

异常层级：`PersistenceError(ValueError)`，具体输入错误为
`PersistenceInputError`（含稳定字段 `reason`、可定位的 `index`（输入可迭代
对象中的 0 基下标）、`simplex` 原文）；证书核验失败为 `CertificateError`。

| reason | 触发条件 |
|---|---|
| `bad_input` | 顶层不是可迭代的单形序列 |
| `bad_simplex` | 单项不是长度 2 的 `(vertices, filtration)` |
| `bad_vertices` | 顶点字段不是元组/列表 |
| `bad_vertex_id` | 顶点不是非负 int（`bool` 一律拒绝，`True` 不等于 1） |
| `duplicate_vertex_in_simplex` | 单形内顶点重复 |
| `unsorted_vertices` | 顶点未严格递增（不静默重排） |
| `bad_dimension` | 顶点数不在 1..3 |
| `bad_filtration_type` | 过滤值不是 int/float；`bool`、字符串等拒绝 |
| `non_finite_filtration` | `NaN`、`+inf`、`-inf` |
| `missing_face` | 顶点面/边面不存在 |
| `face_filtration_order` | 面过滤值大于单形过滤值 |
| `duplicate_simplex` | 同一单形重复给出 |
| `too_many_simplices` | 超过 300 个单形（报错定位到第 301 项） |
| `empty_input` | 空输入 |
| `bad_threshold` | Betti 查询阈值不是有限数 |

其他固定约定：

- **数值策略**：`NaN`/`Infinity` 一律拒绝；**整数参数（顶点 id、以及
  bool 形态的过滤值）拒绝 `bool`**。
- **原子性**：`compute_persistence` 是纯函数，结果在局部状态中完整构造；
  任何校验失败都在返回前抛出，不存在“部分结果”或对输入的修改。本库没有
  有状态/批次写操作，也没有带搜索预算的算法——约化是确定性有限过程，
  因此不存在“预算耗尽”与“证明无解”的区分需求；非法输入与计算失败通过
  异常类型显式区分。
- **浮点误差口径**：本库**不做任何浮点算术**，不过滤、不合并近似值、
  不使用 epsilon。过滤值只参与 Python 原生比较（`==`、`<=`、`<`）并
  原样回显；因此两条只在浮点意义下“接近”的过滤值被视为不同值，
  `0.1 + 0.2` 之类的表示差异按位保留，不会被容差掩盖。Betti 数为整数，
  精确。若调用方需要分箱/归并，请在入库前自行量化。

## 设计取舍

1. **GF(2) 系数**：链即 bitset（Python 大整数），列约化用整数异或实现；
   不跟踪有向 ±1 系数。题目要求的“方向链”体现为过滤总序方向（出生→
   死亡）与每条链可检验的边界证书，而非整系数定向。
2. **经典最低 pivot 约化**（Edelsbrunner–Letscher–Zomorodian 风格的
   left-to-right reduction）：pivot 取列中**最大行号**（矩阵最下方的 1）。
   配对规则：非零约化列的 pivot 行与该列构成配对（边杀顶点→H0 死亡；
   三角形杀边→H1 死亡）；正列（约化后为 0）创建对应维数的出生类；
   无 pivot 认领的正列给本质条（`death = inf`）。
3. **证书即约化副产物**：约化时同时维护换基掩码 `V`，使死亡三角形列的
   约化结果直接是出生环，`V` 列直接是死亡 2-链，无需二次搜索。
4. **零长度条保留**：同值同刻“出生即死亡”的类按约定显式保留并打
   `zero_length`，它们不对任何阈值切片贡献维数（半开区间）。
5. **规模与复杂度**：上限 300 个单形；最坏列约化 O(n³) 个 bitset 字操作，
   在该规模实测毫秒级（300 单形约 5 ms）。
6. **不调用任何第三方同类实现**：边界矩阵、约化、配对全部在
   `pershom/persistence.py` 内手写；测试参考 `tests/reference.py` 同样只
   用标准库，且**不 import 被测核心**；与库的一致性是独立交叉验证，
   “往返一致”只作为补充断言。

## 已知限制（本契约的数学/协议子集）

- 仅维数 0、1、2（顶点/边/三角形）；无四面体及更高维。
- 仅 GF(2) 系数；不提供整数/任意域系数、定向符号或同调群扭结信息。
- 至多 300 个单形；超限拒绝而非分页处理。
- 只计算普通（非扩张）持续同调的区间与链代表元；不输出
  persistence diagram 图像、瓶颈/Wasserstein 距离、上同调、zigzag、
  多参数/圆形/层级持续同调，也不声称与任何行业文件格式
  （如 dipha/perseus 二进制格式）兼容。
- 本质类只给 `inf` 终点，不附链（GF(2) 下本质 H1 代表元可由边环另行
  枚举，当前接口未暴露）。
- 杯积仅限 H1×H1→H2、GF(2) 系数、至多二维复形；不扩展到任意系数环、
  更高次乘积或 Steenrod 运算；基随顶点次序变化，只有零/非零与秩
  是不变量。
- 链代表元依赖固定全序（同值维数序+顶点序）；换序可能得到不同但同调的
  代表链，配对端点不变。

## 快速使用示例

```python
from pershom import compute_persistence

ring_and_fill = [
    ((0,), 0), ((1,), 0), ((2,), 0),
    ((0, 1), 1), ((1, 2), 1), ((0, 2), 1),
    ((0, 1, 2), 2),
]
p = compute_persistence(ring_and_fill)
print(p.betti_at(1))          # (1, 1)
print(p.betti_at(2))          # (1, 0)
bar = p.h1_bars[0]            # H1 [1, 2)
print(p.chain_vertices(bar.birth_chain))  # 三角形的三条边
print(p.chain_vertices(bar.death_chain))  # 填充三角形
```
