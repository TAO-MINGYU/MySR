# 08 — 损失评估、代价计算、常数优化、自适应简约

## 1. 损失评估流水线

### eval_cost (主入口, LossFunctions.jl)
```
eval_cost(dataset, member, options) → (cost, loss)
  │
  ├── eval_loss(tree, dataset, options) → loss
  │   ├── options.loss_function → evaluator(f, tree, dataset, options, idx)  # 自定义(节点级)
  │   ├── options.loss_function_expression → evaluator(f, tree, dataset, options, idx)  # 自定义(表达式级)
  │   └── _eval_loss(tree, dataset, options, regularization)
  │       ├── eval_tree_dispatch(tree, dataset, options) → prediction
  │       ├── _loss(prediction, y, options.elementwise_loss) → raw_loss
  │       └── + dimensional_regularization(tree, dataset, options)
  │
  └── loss_to_cost(loss, use_baseline, baseline, member, options, complexity)
      └── loss / max(baseline_loss, 0.01) + complexity * parsimony
```

### 损失函数类型
- **elementwise_loss** (字符串): `"L2DistLoss()"`, `"L1DistLoss()"`, `"HuberLoss()"` 等 → 通过 `LossFunctions.jl` 解析
- **loss_function** (Julia 函数): `(tree::AbstractExpressionNode, dataset::Dataset, options) -> loss`
- **loss_function_expression** (Julia 函数): `(tree::AbstractExpression, dataset::Dataset, options) -> loss`
- **loss_function_expression 还支持批处理**: `(tree, dataset, options, idx) -> loss`

### baseline_loss 机制
- 初始化为评估常数表达式(如常量0)的损失
- cost = loss / max(baseline_loss, 0.01):
  - 如果 baseline_loss=1.0, loss=0.01 → cost=0.01
  - 如果 baseline_loss=0.001, loss=0.01 → cost=10.0
- 目的: 标准化不同数据集的损失量级

### 维度正则化
- `dimensional_constraint_penalty` (默认1000): 违反维度约束的惩罚
- 计算方式: `dimensional_regularization(tree, dataset, options)`
  - 如果 `violates_dimensional_constraints` →返回 penalty 值
  - 否则 → 0
- 这使违反维度约束的表达式在进化中被淘汰

## 2. 常数优化 (ConstantOptimization.jl)

### optimize_constants 流程
```
optimize_constants(dataset, member, options) → (updated_member, num_evals)
  
  1. 检查 can_optimize(tree, options) → 是否支持优化
  2. 统计常数数量:
     - 0 → 直接返回
     - 1 → 使用 Optim.Newton (1D 更快)
     - >1 → 使用 options.optimizer_algorithm (默认 BFGS)
  
  3. 提取常数: get_scalar_constants(tree) → Vector
  4. 构造 Evaluator:
     - f(x) = eval_loss(tree_with_constants_set_to_x, dataset, options)
  5. 构造 GradEvaluator (如果 autodiff_backend 支持):
     - fg!(value, grad, x) = 评估损失和梯度
  
  6. 主优化:
     Optim.optimize(f, x0, algorithm, options.optimizer_options)
  
  7. nrestarts 次重新优化:
     从 x0 * (1 + 0.5*randn()) 开始
  
  8. 如果最佳结果改进了 baseline:
     更新 member.tree 常数 → 重算 member.loss → 重算 member.cost
     否则恢复原常数
```

### 自动微分后端
- **默认**: 无 (Optim.jl 内部用有限差分)
- **Zygote.jl**: `autodiff_backend=:Zygote`
- **Enzyme.jl**: `autodiff_backend=:Enzyme` — 需要更大的栈空间 (32MB)
- **Mooncake.jl**: `autodiff_backend=:Mooncake` — 支持 TemplateExpression
- 选择规则: `optimizer_algorithm` + `autodiff_backend` 组合决定是否使用梯度

### Evaluator 结构
```julia
Evaluator{N, R, C} <: Function
  tree::N              # 表达式树
  constant_refs::R     # 常数引用 (指向树中的常数节点)
  context::C           # EvaluatorContext (包含 dataset, options)
```

- `(e::Evaluator)(x)` → 设置常数为 x → eval_loss → 返回损失
- `GradEvaluator(_, G, x)` → 计算 value_and_gradient

## 3. 复杂度计算 (Complexity.jl)

### 默认: 节点计数
- `compute_complexity(tree)` = `count_nodes(tree)` → 操作符数 + 叶子数

### 自定义: ComplexityMapping
```julia
ComplexityMapping(
    op_complexities:     Dict(operator => weight)    # 每个操作符的代价
    variable_complexity: Float64 | Vector{Float64}   # 变量代价
    constant_complexity: Float64                      # 常数代价
)
```

- 计算: `sum(各节点的复杂度权重) + sum(变量权重[feature]) + n_constants * constant_complexity`
- 共享节点 (DAG) 只计数一次
- 复杂度 = `round(Int, weighted_sum)`

## 4. 自适应简约 (AdaptiveParsimony.jl)

### RunningSearchStatistics
```julia
mutable struct RunningSearchStatistics
    window_size::Int                    # 窗口大小 (默认 100000)
    frequencies::Vector{Float64}        # 各复杂度出现的次数 (初始全1)
    normalized_frequencies::Vector{Float64}  # 归一化频率
end
```

### 机制
1. **update_frequencies!**: 每次变异接受一个新个体，递增对应复杂度的计数
2. **move_window!**: 当总计数超过 window_size，等量衰减所有 bin
3. **normalize_frequencies!**: 重新归一化

### 使用场景
- **use_frequency (接受阶段)**:
  ```
  prob_accept *= normalized_frequencies[old_complexity] / 
                 normalized_frequencies[new_complexity]
  ```
  如果新复杂度在搜索中较少出现 → 更可能被接受

- **use_frequency_in_tournament (锦标赛选择)**:
  ```
  adjusted_cost = cost * exp(adaptive_parsimony_scaling * 
                             normalized_frequencies[complexity])
  ```
  如果某复杂度过度出现 → tournament 中它的选择代价更高

### 与 parsimony 的区别
- `parsimony` (静态): `cost = loss + complexity * parsimony` — 始终惩罚大方程
- `adaptive_parsimony_scaling` (动态): 惩罚过度表示的复杂度 — 自适应

## 5. 分数计算 (Score)

### score 公式
```
score = relu(-log(cur_loss / prev_loss) / (cur_complexity - prev_complexity))
```
- 物理意义: 每增加单位复杂度的对数损失减少量
- 只在 `loss_scale=:log` 时使用
- 用于 `model_selection="score"` 或 `"best"`

### "best" 选择策略
```
1. 找到损失最小的方程
2. 过滤: 保留损失 ≤ 1.5× 最佳损失的方程
3. 从候选中选择 score 最高的
```
- 平衡准确性和简单性

## 6. 约束检查 (CheckConstraints.jl)

```
check_constraints(expression, options, maxsize, cached_size):
  1. 复杂度 ≤ maxsize
  2. 树深度 ≤ maxdepth
  3. 每个操作符的子节点复杂度 ≤ op_constraints[degree][op][arg]
  4. 嵌套约束: 特定操作符合法的嵌套层数 ≤ nested_constraints
```

- **操作符约束**: `constraints = [^ => (3, -1)]` 表示 `^` 的左子节点复杂度≤3，右子节点不限
- **嵌套约束**: `nested_constraints = [^ => [sin => 1]]` 表示 `^` 内最多嵌套1层 `sin`
