# 06 — 正则化进化与变异系统

## 1. 正则化进化 (RegularizedEvolution.jl)

### 算法
```
reg_evol_cycle(dataset, pop, temperature, curmaxsize, stats, options):
    n_cycles = ceil(pop_size / tournament_selection_n)
    
    for i in 1:n_cycles:
        if rand() > crossover_probability:
            # === 变异路径 ===
            parent = best_of_sample(pop, stats, options)  # 锦标赛
            child, accepted, evals = next_generation(
                dataset, parent, temperature, curmaxsize, stats, options
            )
            if accepted (or not skip_mutation_failures):
                oldest = argmin(pop.members.birth)  # 替换最老的
                pop.members[oldest] = child
        
        else:
            # === 交叉路径 ===
            parent1 = best_of_sample(pop, stats, options)
            parent2 = best_of_sample(pop, stats, options)
            child1, child2, accepted, evals = crossover_generation(
                parent1, parent2, dataset, curmaxsize, options
            )
            if accepted (or not skip_mutation_failures):
                oldest1, oldest2 = 最老的两个
                pop.members[oldest1] = child1
                pop.members[oldest2] = child2
```

### 关键特性
- **稳态替换**：每次只替换1-2个个体，不是整个世代
- **年龄替换**：替换最老的个体 → 保证每个个体存活一定代数 → 维持多样性
- **锦标赛选择**：从群体中随机抽 `tournament_selection_n` 个，选最佳
- **温度退火**：`temperature` 从1.0降到0.0 → 早期更探索，后期更开发

## 2. next_generation (核心变异函数, Mutate.jl)

### 流程
```
next_generation(dataset, member, temperature, curmaxsize, stats, options):
    
    # 1. 条件化变异权重
    weights = copy(options.mutation_weights)
    condition_mutation_weights!(weights, member, options, curmaxsize, nfeatures)
    
    # 2. 采样变异类型
    mutation_choice = sample_mutation(weights)  # 如 :mutate_operator
    
    # 3. 重试循环 (最多10次)
    for attempt in 1:10:
        tree = copy(member.tree)
        result = _dispatch_mutations!(tree, member, mutation_choice, ...)
        
        if result.return_immediately:
            return result  # 如 simplify, optimize
        
        if check_constraints(result.tree, options, curmaxsize):
            break  # 通过约束检查
        else:
            continue  # 重试
    
    if all attempts failed:
        return (rejected: 原个体副本)
    
    # 4. 评估新树
    cost, loss = eval_cost(dataset, tree, options)
    if isnan(cost): return rejected
    
    # 5. 接受/拒绝决策 (模拟退火 + 频率简约)
    prob_accept = 1.0
    if annealing:
        delta = cost - parent.cost
        prob_accept *= exp(-delta / (temperature * alpha))
    if use_frequency:
        prob_accept *= old_freq / new_freq  # 偏向欠表示的复杂度
    
    if rand() > prob_accept:
        return (rejected)
    else:
        return (new_pop_member, accepted)
```

### 接受标准
1. **约束检查**：复杂度 ≤ maxsize，深度 ≤ maxdepth，操作符约束
2. **NaN 检查**：产生 NaN 损失的表达式被拒绝
3. **模拟退火**：`exp(-Δcost / (T * α))` — 较高温度时更可能接受差解
4. **频率简约**：偏向搜索中欠表示的复杂度级别

## 3. 变异类型 (MutationWeights.jl)

### 默认权重
| 变异类型 | 默认权重 | 说明 |
|----------|----------|------|
| `mutate_operator` | 3.63 | 随机更换操作符 |
| `insert_node` | 2.44 | 在任何节点插入新操作符 |
| `rotate_tree` | 1.42 | 树旋转 (保持语义) |
| `do_nothing` | 0.431 | 不变 |
| `delete_node` | 0.369 | 删除一个操作符节点 |
| `mutate_constant` | 0.0353 | 扰动常数 |
| `mutate_feature` | 0.1 | 更改变量引用 |
| `add_node` | 0.0771 | 在叶子上追加操作符 |
| `swap_operands` | 0.00608 | 交换二元操作数 |
| `randomize` | 0.00695 | 替换为完全随机树 |
| `simplify` | 0.00148 | 代数化简 (作为变异) |
| `optimize` | 0.0 | 常数优化 (仅在后处理) |
| `form_connection` | 0.5 | 添加图边 (GraphNode) |
| `break_connection` | 0.1 | 断开图边 (GraphNode) |

### condition_mutation_weights! 动态调整
- 叶子节点：禁用 delete_node, swap_operands, simplify
- 常数叶子：禁用 mutate_feature
- 变量叶子：禁用 mutate_constant
- 接近 maxsize：禁用 add_node, insert_node
- 无二元操作符：禁用 swap_operands
- 无图结构：禁用 form_connection, break_connection

## 4. 变异原语 (MutationFunctions.jl)

### 树结构变异

| 函数 | 操作 | 对树大小的影响 |
|------|------|---------------|
| `mutate_operator` | 随机更换同 arity 的操作符 | 不变 |
| `mutate_constant` | 常数 × 随机因子 `(α*T+1.1)^rand()` | 不变 |
| `mutate_feature` | 更改变量引用到不同特征 | 不变 |
| `swap_operands` | 交换二元操作符的两个子节点 | 不变 |
| `append_random_op` | 在随机叶子上追加操作符 | +arity |
| `insert_random_op` | 在任意位置插入操作符 | +arity |
| `prepend_random_op` | 在根节点前插入操作符 | +arity |
| `delete_random_op!` | 删除操作符，用其子节点替代 | -arity |
| `randomize_tree` | 完全随机生成新树 | 随机 |
| `randomly_rotate_tree!` | 树旋转 (改变拓扑，保持语义) | 不变 |

### 随机树生成
- `gen_random_tree_fixed_size(count, options, nfeatures, rng)`
  - 从叶子开始，逐次在随机位置添加操作符
  - `_arity_picker` 根据剩余空间选择合适的 arity
  - 保证最终树的节点数恰好为 count

### 交叉
- `crossover_trees(ex1, ex2, rng)`
  - 在每个树中随机选一个节点
  - 交换两个子树
  - 最大重试10次直到满足约束

## 5. 编译时优化

- `_dispatch_mutations!` 是 `@generated` 函数，用 `Base.Cartesian.@nif` 生成编译时分发链
- `append_random_op` 等也用 `@generated` 展开 degree 维度的循环
- 消除运行时对 arity 的分发开销
