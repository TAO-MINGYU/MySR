# 07 — 表达式类型与表示

## 1. 表达式类型层次

```
AbstractExpressionNode{T}          (DynamicExpressions.jl)
├── Node{T}                        # 标准树节点
├── GraphNode{T}                   # 支持共享子节点的 DAG 节点
└── ParametricNode{T}              # 支持参数引用的节点
    (每个节点可以是: 常数 | 特征引用 | 操作符)

AbstractExpression{T, N}           (DynamicExpressions.jl)
├── Expression{T, N}               # 标准表达式 (Node + 元数据)
│   └── 元数据: operators, variable_names
├── ComposableExpression{T, N, D}  # 可组合表达式 (支持向量化评估)
│   ├── 元数据: operators, variable_names, eval_options
│   └── 核心: ValidVector — 带有效性标记的向量
├── TemplateExpression{T, F, N, E, TS, D}  # 多表达式模板
│   ├── trees::NamedTuple   # 多个 ComposableExpression 子表达式
│   ├── parameters::NamedTuple{ParamVector}  # 参数向量
│   └── structure::TemplateStructure  # combine 函数 + 约束
└── ParametricExpression{T, N}     # [已弃用] 类条件参数表达式
```

## 2. Node{T} — 标准树节点

```
节点分为三类:
  1. 常数节点: degree=0, constant=true, val=...
  2. 特征节点: degree=0, constant=false, feature=Int
  3. 操作符节点: degree>0, op=Int (指向 OperatorEnum 的索引)
```

- 使用 `AbstractExpressionNode{T}` 的 union 类型表示
- `T` 为数据类型: `Float32`, `Float64`, 或 `Complex{Float32}` 等
- 树的评估通过 `eval_tree_array(tree, X, operators)` 完成

## 3. ComposableExpression + ValidVector 系统

### ValidVector
```julia
struct ValidVector{A <: AbstractVector}
    x::A          # 向量数据
    valid::Bool   # 是否有效 (无 NaN, Inf, DomainError)
end
```

### 设计目的
- 自动传播数值无效性: `ValidVector(valid=false) + anything → valid=false`
- 避免被零除/无效域错误中断计算
- 支持表达式组合: `ex1(ex2(X))` — 将 ex2 的输出作为 ex1 的输入

### 运算符重载
所有标准数学运算 (`+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `exp`, `log`, `sqrt` 等) 都被重载以正确处理 `ValidVector`:
- 两个都有效 → 正常计算
- 任一无效 → 返回无效

## 4. TemplateExpression — 多表达式模板

### 结构
```
TemplateExpression:
  trees:
    f = ComposableExpression("cos(x1) - 1.5")
    g = ComposableExpression("x1 + x2*0.3")
  parameters:
    p1 = ParamVector([0.1, 0.2, 0.3])  # 每类一个值
  structure:
    combine = (f, g, p1) -> f * exp(-g) + p1
    num_features = (f=2, g=3)   # f 可用特征数, g 可用特征数
    num_parameters = (p1=3)     # p1 的长度
```

### 约束机制
- `num_features`: 每个子表达式只能看到特征的一个子集
- `has_invalid_variables`: 检查子表达式是否访问了超出范围的特征
- 特征数量在突变时受 `get_nfeatures_for_mutation` 限制

### 复杂度计算
- TemplateExpression 的复杂度 = **sum(各子表达式的复杂度)**
- 而非将子表达式拼接成完整树的复杂度
- 因为子表达式是独立进化的

### 评估流程
```
(X) → 将每一行包装为 ValidVector
    → combine_function(
        namedtuple_of_expressions,
        parameters_tuple,
        each_row_ValidVector...
    )
    → 验证返回 ValidVector
    → 提取数值结果
```

### @template_spec 宏
```julia
@template_spec(
    expressions=(f, g),
    parameters=(p1=3,),           # 可选
    num_features=(f=2, g=3)       # 可选
) do x1, x2, x3
    f(x1) * exp(-g(x1, x2, x3)) + p1
end
```
- 生成确定性函数名 (`__sr_template_HASH`)
- 自动推断特征使用情况 (通过 `ArgumentRecorder`)
- 返回 `TemplateExpressionSpec`

## 5. ParametricExpression [已弃用]

- 每个样本有一个类别标签，参数值按类别不同
- 存储在参数矩阵 `(num_params × num_classes)` 中
- 评估时 `eval_tree_array(tree, X, class)` 按类别索引参数
- **已弃用**: 推荐用 `TemplateExpressionSpec` + `@template_spec` 替代
- 弃用示例转换:
  ```julia
  # 旧
  ParametricExpressionSpec(max_parameters=3)
  
  # 新
  @template_spec(expressions=(main,), parameters=(p=3,)) do x1
      main(x1) + p
  end
  ```

## 6. ExpressionSpec — 工厂模式

```julia
abstract type AbstractExpressionSpec end

struct ExpressionSpec <: AbstractExpressionSpec
    node_type::Type = Node  # 可改为 GraphNode 等
end

struct TemplateExpressionSpec{ST} <: AbstractExpressionSpec
    structure::ST
end
```

- ExpressionSpec 决定:
  - 表达式类型 (`get_expression_type`)
  - 节点类型 (`get_node_type`)
  - 表达式特定选项 (`get_expression_options`)
- Python 端的 `ExpressionSpec` / `TemplateExpressionSpec` / `ParametricExpressionSpec` 映射到 Julia 端

## 7. ExpressionBuilder — 树→表达式的装配

- `create_expression(tree, options, dataset, Val(embed))` — 将裸节点树转换为完整表达式
- `embed_metadata(expr, options, dataset)` — 注入完整元数据 (运算符、变量名)
- `strip_metadata(expr, options, dataset)` — 剥离元数据 (节省进化时的内存)
- `extra_init_params` — 可重载，让自定义表达式类型注入额外参数
- `consistency_checks` — 验证表达式类型与 options 的兼容性

## 8. 评估接口 (eval_tree_array)

```
eval_tree_array(tree, X, operators; kws...):
    - 对 X 的每列, 遍历树的每个节点
    - 常数节点 → 返回 val
    - 特征节点 → 返回 X[feature, :]
    - 操作符节点 → 递归评估子节点, 调用 operators[degree][op]
    - 返回 (output_array, completion_flag)
```
