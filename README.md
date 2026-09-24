# pershom — 带方向链证书的持久同调配对 + 动态增删复形（GF(2)，H0/H1）

离线、确定性、仅依赖 Python 标准库的持久同调计算模块，包含两块能力：

1. **静态过滤复形**：对顶点/边/三角形过滤复形，在 **GF(2)** 上建立边界
   矩阵，用标准**最低位 pivot**（lowest-one / bottom pivot）列约化给出
   H0、H1 的出生–死亡配对，并为**每条有限 H1 条**同时交付可独立检验的
   **出生环 1-链**与**死亡填充 2-链**证书。
2. **动态增删复形**（本次迭代新增）：`DynamicComplex` 接受最高维度 2 的
   GF(2) 复形的**逐条增删事件**；删除一个仍有共面（coface）存在的单形
   会被**原子拒绝**；每步事件后用同一套边界矩阵约化**全量重算**
   H0/H1 维数（Betti 数）并随步骤返回。

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
  dynamic.py         动态复形：增删事件、原子拒绝、逐步 Betti 重算
tests/
  reference.py       独立参考：GF(2) 高斯消元 + 秩不变量 + 并查集（不 import pershom）
  test_persistence.py  静态过滤复形测试
  test_dynamic.py      动态增删测试（正常/边界/失败 + 随机模型对拍）
demo.py              固定输入演示（正常结果 + 真实触发的拒绝 + 动态增删）
```

## 运行命令

```bash
# 必要测试（验收命令）
python -m unittest discover -s tests -v

# 演示（约 0.05 秒，远小于 8 秒预算；真实计算，无 sleep、无预录结果）
python demo.py
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

## 动态增删复形（DynamicComplex）

```python
from pershom import DynamicComplex, ADD, REMOVE

cx = DynamicComplex()          # 空复形起步，维度 <= 2，上限同为 300 个单形
step = cx.add((0, 1))          # 单个事件；返回 DynamicStep
steps = cx.apply([("add", (0,)), ("add", (1,)), ("add", (0, 1))])  # 事务性批次
cx.betti()                     # 当前 (beta0, beta1)
cx.simplices()                 # 当前全部单形（按 (维数, 顶点) 排序）
(0, 1) in cx                   # 成员查询
```

### 事件语义

- `add(vertices)`：顶点规则与静态接口完全一致（1–3 个严格递增非负
  int，拒绝 bool/重复/乱序）。要求所有余维 1 面**已经存在**（由此归纳
  保证复形始终面闭合），否则以 `missing_face` 拒绝；重复添加以
  `duplicate_simplex` 拒绝；超过 300 个单形以 `too_many_simplices` 拒绝。
- `remove(vertices)`：单形必须存在（否则 `not_present`），且**不得有
  任何共面仍存在**——有共面时以 `coface_present` **原子拒绝**，复形
  保持原样。要删一条被三角形引用的边，必须先删该三角形。
- 两个操作都先完整校验、后修改，失败时不留下任何部分变更。
- `apply(events)`：**事务性**批次。事件为 `(op, vertices)` 二元组，
  `op ∈ {"add", "remove"}`（`pershom.ADD` / `pershom.REMOVE`）。整个
  批次先在私有副本上模拟校验，全部合法才提交；任一事件非法则整批
  拒绝、复形与调用前完全一致，异常 `index` 定位到批次内 0 基事件下标。
  空批次是合法空操作，返回 `[]`。
- 每个被应用的事件返回 `DynamicStep(event, vertices, dimension, betti)`，
  其中 `betti` 是该事件**之后**的 `(beta0, beta1)`；`apply` 按事件顺序
  返回列表。
- Betti 数每步**全量重算**：按 `(维数, 顶点)` 排序建边界矩阵，用与静态
  接口相同的 GF(2) 最低位 pivot 约化取 `rank(∂1)`、`rank(∂2)`，再按
  秩–零化度 `β0 = n0 − rank(∂1)`、`β1 = n1 − rank(∂1) − rank(∂2)`。
  空复形的 Betti 为 `(0, 0)`。
- 动态复形**不带过滤值**：它回答的是“当前复形的 H0/H1 维数”，不是
  区间分解（见“已知限制”）。

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
| `not_present` | （动态）删除的单形不在复形中 |
| `coface_present` | （动态）删除的单形仍有共面存在，原子拒绝 |
| `bad_event` | （动态）批次事件不是 `(op, vertices)` 或 op 非法 |

其他固定约定：

- **数值策略**：`NaN`/`Infinity` 一律拒绝；**整数参数（顶点 id、以及
  bool 形态的过滤值）拒绝 `bool`**。
- **原子性**：`compute_persistence` 是纯函数，结果在局部状态中完整构造；
  任何校验失败都在返回前抛出，不存在“部分结果”或对输入的修改。动态
  复形的 `add`/`remove` 先校验后修改，`apply` 整批模拟通过才提交，
  失败同样不留下部分变更。本库没有带搜索预算的算法——约化是确定性
  有限过程，因此不存在“预算耗尽”与“证明无解”的区分需求；非法输入
  与计算失败通过异常类型显式区分。
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
   在该规模实测毫秒级（300 单形约 5 ms）。动态复形每步全量重算 Betti
   （题面允许），600 步随机事件流在测试中约 0.1 秒完成。
6. **动态复形不做增量持久同调**：增删事件的每步只交付当前 Betti 维数，
   复用同一边界约化从零重算，不维护跨步骤的约化状态、不做之字形
   （zigzag）区间分解、不跟踪代表链的更新。
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
- 链代表元依赖固定全序（同值维数序+顶点序）；换序可能得到不同但同调的
  代表链，配对端点不变。
- 动态复形（`DynamicComplex`）只输出每步的 H0/H1 **维数**：不提供增删
  序列的之字形（zigzag）区间分解、不提供代表链/映射证书、不维护增量
  约化状态；单形上限与静态接口共享 300。

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

动态增删：

```python
from pershom import DynamicComplex

cx = DynamicComplex()
for step in cx.apply([
    ("add", (0,)), ("add", (1,)), ("add", (2,)),
    ("add", (0, 1)), ("add", (1, 2)), ("add", (0, 2)),   # 成环 -> (1, 1)
    ("add", (0, 1, 2)),                                    # 填洞 -> (1, 0)
    ("remove", (0, 1, 2)),                                 # 删面 -> (1, 1)
    ("remove", (0, 2)),                                    # 删边 -> (1, 0)
]):
    print(step.event, step.vertices, "->", step.betti)
cx.remove((0, 1))   # 若三角形仍在：PersistenceInputError, reason="coface_present"
```
