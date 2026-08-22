# 11 — 面向物理的特性: 维度分析、量纲约束、单位处理

## 1. 维度分析系统 (DimensionalAnalysis.jl)

### 设计目的
在物理符号回归中，表达式必须在量纲上一致。例如:
- 不能将"米"与"秒"相加
- 常数的量纲必须与被加数的量纲匹配

### 核心类型: WildcardQuantity
```julia
struct WildcardQuantity{Q <: AbstractQuantity}
    val::Q           # 底层物理量 (含数值和单位)
    wildcard::Bool   # 该量是否具有自由(未知)维度
    violates::Bool   # 是否检测到维度不一致
end
```

**通配符常数的语义**:
- 方程中的自由常数初始化为 wildcard=true
- 当通配符常数与有维度的值相加时，常数"继承"该维度
- 当通配符常数与有维度的值相乘时，常数取乘积的维度
- 如果两个有维度但量纲不匹配的值相加，violates=true

### 运算规则
```
加减:
  - dim(A) == dim(B) → 正常组合
  - A.wildcard=true, B.wildcard=false → A 继承 B 的维度
  - B.wildcard=true, A.wildcard=false → B 继承 A 的维度
  - 两者都 wildcard → 结果保持 wildcard
  - 两者都无 wildcard 且维度不同 → violates=true

乘除:
  - 维度按物理规则组合
  - wildcard 传播: A.wildcard || B.wildcard

幂次:
  - 仅允许: 底数为无量纲 或 wildcard
  - 指数必须为无量纲
  - 否则 → violates
```

### 维度违规检测
```julia
violates_dimensional_constraints(tree, dataset, options) → Bool
```
- 递归评估树，每个节点返回 `WildcardQuantity`
- 如果根节点的 `violates=true` 或输出维度与 `y_units` 不匹配 → 违规
- `dimensionless_constants_only=true` → 禁用通配符 (常数必须立即匹配)

### 维度惩罚
```
if violates_dimensional_constraints:
    loss += options.dimensional_constraint_penalty  # 默认 1000
```
这使维度不一致的表达式在进化中被迅速淘汰。

## 2. 单位系统 (DynamicQuantities.jl)

### 数据集的单位标注
```python
# Python 端
model.fit(X, y, X_units=["m", "s", "kg"], y_units="J")
```

在 Julia 端:
- `X_units`: 每个输入特征的单位
- `y_units`: 目标变量的单位
- 支持 SI 单位和符号单位 (通过 `DynamicQuantities`)

### Dataset 中的单位字段
```julia
Dataset:
  X_units::AbstractVector    # 输入单位向量
  y_units::AbstractQuantity  # 目标单位
  X_sym_units                # 符号单位 (用于显示)
  y_sym_units
```

### 单位类型支持
- `DynamicQuantities.SymbolicDimensions` — 符号量纲如 `length`, `time`, `mass`
- `DynamicQuantities.Quantity` — 数值量纲如 `m`, `s`, `kg`

### 输入数据自动解包
当传入带 `Quantity` 的 Matrix 时:
- MLJ 接口自动调用 `unwrap_units_single` 剥离单位
- 保留单位信息传递给 Julia 后端
- 预测结果可选地重新包裹单位

## 3. 数据预处理 (Python 端)

### Gaussian Process 去噪
```python
PySRRegressor(denoise=True)
```
- 使用 `GaussianProcessRegressor` (RBF + WhiteKernel + ConstantKernel)
- 对 (X, y) 拟合 GP，使用 GP 预测值替代原始 y
- 50 次优化器重启动
- 支持多输出: `multi_denoise()` 独立处理每个输出列
- 支持 `Xresampled` 重采样到不同网格点

### 特征选择
```python
PySRRegressor(select_k_features=True)
```
- 使用 `RandomForestRegressor` (100棵树, max_depth=3)
- 基于特征重要性选择 top-k
- `SelectFromModel(threshold=-inf, max_features=select_k_features)`
- 返回 bool mask + 截断的 X

## 4. 操作符安全性 (Operators.jl)

### 物理计算中的 safe_* 函数族
所有可能导致 NaN/Inf/DomainError 的操作符都有 safe 版本:

| 标准操作符 | Safe 版本 | 保护措施 |
|-----------|----------|---------|
| `x^y` | `safe_pow(x, y)` | 防止 `0^负`, `负^非整数` |
| `sqrt(x)` | `safe_sqrt(x)` | 返回 NaN for x < 0 |
| `log(x)` | `safe_log(x)` | 返回 NaN for x ≤ 0 |
| `asin(x)` | `safe_asin(x)` | 检查域 |
| `acos(x)` | `safe_acos(x)` | 检查域 |
| `acosh(x)` | `safe_acosh(x)` | 返回 NaN for x < 1 |
| `atanh(x)` | `safe_atanh(x)` | 检查域 |
| `gamma(x)` | 内部检查 | 返回 NaN for 无穷结果 |

### 操作符映射机制
```julia
get_safe_op(op)  # 将标准 Julia 操作符映射到 safe 版本
# 例如: ^ → safe_pow, sqrt → safe_sqrt
```
- 在 `OperatorEnum` 创建时自动应用
- 评估时通过 `get_safe_op` 查找
- 用户提供的操作符也会被映射

## 5. 物理相关的 PySR 使用模式

### 模式 1: 维度约束搜索
```python
model = PySRRegressor(
    operators={2: ["+", "-", "*", "/"], 1: ["sqrt", "exp", "log"]},
    dimensional_constraint_penalty=1000,
    dimensionless_constants_only=False,  # 允许常数带维度
)
model.fit(X, y, X_units=["m", "m/s", "kg"], y_units="J")
```

### 模式 2: 无维度数据 + 操作符约束
```python
model = PySRRegressor(
    operators={2: ["+", "*", "-", "/", "^"]},
    constraints={"/": (-1, 3)},  # 分母复杂度 ≤ 3
    nested_constraints={"^": {"sin": 1, "cos": 1}},  # 幂指数不能嵌套三角函数
)
```

### 模式 3: 多输出物理公式
```python
model = PySRRegressor(
    operators={2: ["+", "-", "*", "/"], 1: ["sqrt", "square"]},
    populations=40,  # 增加群体数
    niterations=100,  # 增加迭代
)
model.fit(X, y)  # y shape (n_samples, 2)
# model.equations_ → [df_output0, df_output1]
```

### 模式 4: 已知部分形式 (Template)
```python
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    combine="f * exp(-g) + p1"
)
model = PySRRegressor(
    expression_spec=spec,
    operators={2: ["+", "-", "*", "/"], 1: ["exp", "log", "cos"]},
)
model.fit(X, y)
# 搜索 f 和 g 的最佳形式，通过模板组合
```

## 6. 维度分析的有效性边界

### 当前支持
- 加减: 维度自动检测和匹配
- 乘除: 维度按物理规则传播
- 幂次: 受限于无量纲底数/指数
- 常数: wildcard 机制自动确定维度

### 当前限制
- 不支持函数如 `sin`, `cos`, `exp` 的维度检查 (这些函数的参数必须无量纲，但系统不强制执行)
- TemplateExpression 暂不支持维度分析 (stub 返回 false)
- 不支持自定义维度函数 (如 "对维度为 A 的量取 log 应该得到 log(A)")
- GraphNode 的维度分析有限

### 与常规表达式的区别
- 维度分析只在默认 `Expression` 类型中完整实现
- `TemplateExpression` 和 `ParametricExpression` 的维度分析是 stubbed
