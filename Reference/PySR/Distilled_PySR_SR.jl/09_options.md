# 09 — Julia 后端 Options 完整参考

## Options 结构体 (OptionsStruct.jl)

```julia
struct Options{CM, OP, NOPS, OP_CONSTRAINTS, N, E, EO, MW, PM,
               _turbo, _bumper, _return_state, AD, print_precision}
```

### 运算符和约束
| 字段 | 类型 | 说明 |
|------|------|------|
| `operators` | `OperatorEnum` | 包含所有操作符及其 safe 版本 |
| `op_constraints` | `NTuple{D, Vector{ConstType}}` | 每个操作符的参数复杂度约束 |
| `nested_constraints` | 约束向量 | 嵌套操作符约束 |
| `complexity_mapping` | `ComplexityMapping` | 操作符/变量/常数的复杂度权重 |
| `define_helper_functions` | `Bool` | 是否定义 safe_* 包装函数 |

### 搜索执行
| 字段 | 默认(v2) | 说明 |
|------|----------|------|
| `populations` | 31 | 岛屿群体数 |
| `population_size` | 27 | 每群体个体数 |
| `ncycles_per_iteration` | 380 | 每次迭代的进化周期 |
| `crossover_probability` | 0.066 | 交叉 vs 变异的概率 |
| `annealing` | true | 是否使用模拟退火 |
| `alpha` | 3.17 | 退火温度系数 |

### 变异权重 (MutationWeights)
| 字段 | 默认值 |
|------|--------|
| `mutate_operator` | 3.63 |
| `insert_node` | 2.44 |
| `rotate_tree` | 1.42 |
| `do_nothing` | 0.431 |
| `delete_node` | 0.369 |
| `mutate_feature` | 0.1 |
| `add_node` | 0.0771 |
| `mutate_constant` | 0.0353 |
| `randomize` | 0.00695 |
| `swap_operands` | 0.00608 |
| `simplify` | 0.00148 |
| `optimize` | 0.0 |
| `form_connection` | 0.5 |
| `break_connection` | 0.1 |

### 变异其他
| 字段 | 说明 |
|------|------|
| `perturbation_factor` | 常数突变因子，默认 1.0 |
| `probability_negate_constant` | 常数取负概率，默认 0.01 |
| `skip_mutation_failures` | 约束检查失败时跳过替换 |

### 锦标赛选择
| 字段 | 默认 | 说明 |
|------|------|------|
| `tournament_selection_n` | 10 | 锦标赛抽样大小 |
| `tournament_selection_p` | 0.5 | 几何分布参数 (p=1 为确定性选最佳) |

### 常数优化
| 字段 | 默认 | 说明 |
|------|------|------|
| `should_optimize_constants` | true | 是否优化常数 |
| `optimizer_algorithm` | `Optim.BFGS` | 优化算法 |
| `optimizer_probability` | 0.1 | 每个体被优化的概率 |
| `optimizer_nrestarts` | 2 | 从随机扰动点重启动次数 |
| `optimizer_options` | `Optim.Options(...)` | 收敛参数 |

### 损失函数
| 字段 | 说明 |
|------|------|
| `elementwise_loss` | 逐元素损失，如 `L2DistLoss()` |
| `loss_function` | 自定义 Julia 损失 `(tree, dataset, options) -> loss` |
| `loss_function_expression` | 自定义损失 `(ex, dataset, options) -> loss` |
| `loss_scale` | `:log` 或 `:linear`; 影响 score 计算 |
| `early_stop_condition` | `(loss, complexity) -> Bool`; 满足时停止 |
| `dimensional_constraint_penalty` | 维度违规惩罚，默认 1000 |

### 复杂度
| 字段 | 默认 | 说明 |
|------|------|------|
| `parsimony` | 0.0 | 复杂度惩罚系数 |
| `adaptive_parsimony_scaling` | 1040.0 | 自适应简约强度 |
| `warmup_maxsize_by` | 0.0 | 预热期 maxsize 缩减 |

### 迁移
| 字段 | 默认 | 说明 |
|------|------|------|
| `migration` | true | 群体间迁移 |
| `hof_migration` | true | 名人堂迁移 |
| `fraction_replaced` | 0.08 | 群体间迁出比例 |
| `fraction_replaced_hof` | 0.07 | 名人堂迁出比例 |
| `fraction_replaced_guesses` | 0.001 | 种子方程注入比例 |
| `topn` | 10 | 最佳子集大小 |

### 批处理
| 字段 | 说明 |
|------|------|
| `batching` | 是否使用批处理 |
| `batch_size` | 批次大小 |

### 表达式类型
| 字段 | 默认 | 说明 |
|------|------|------|
| `node_type` | `Node` | `Node`, `GraphNode`, `ParametricNode` |
| `expression_type` | `Expression` | 表达式类型 |
| `expression_options` | `NamedTuple()` | 表达式类型特定选项 |
| `popmember_type` | `PopMember` | 群体成员类型 |

### 尺寸约束
| 字段 | 默认 | 说明 |
|------|------|------|
| `maxsize` | 30 | 最大复杂度 |
| `maxdepth` | `typemax(Int)` | 最大树深度 |

### 早停
| 字段 | 默认 | 说明 |
|------|------|------|
| `timeout_in_seconds` | `nothing` | 超时秒数 |
| `max_evals` | `nothing` | 最大评估次数 |
| `early_stop_condition` | `nothing` | 自定义停止函数 |

### 性能
| 字段 | 说明 |
|------|------|
| `turbo` | `Val{false}` → `Val{true}` 启用 LoopVectorization |
| `bumper` | `Val{false}` → `Val{true}` 启用 Bumper.jl |
| `autodiff_backend` | `nothing` / `AutoZygote()` / `AutoEnzyme()` / `AutoMooncake()` |

### I/O
| 字段 | 说明 |
|------|------|
| `verbosity` | 日志级别 (0-3) |
| `progress` | 是否显示进度条 |
| `save_to_file` | 是否保存 CSV |
| `output_directory` | 输出目录 |
| `seed` | 随机种子 |
| `deterministic` | 确定性执行 |
| `use_recorder` | 启用 record 快照 |

## 默认配置变化

| 参数 | v0.x (旧) | v1.0+ (当前) |
|------|-----------|-------------|
| `maxsize` | 20 | 30 |
| `populations` | 15 | 31 |
| `population_size` | 33 | 27 |
| `ncycles_per_iteration` | 550 | 380 |
| `parsimony` | 0.0032 | 0.0 |
| `adaptive_parsimony_scaling` | 20.0 | 1040.0 (v2: 20.0) |
| `annealing` | false | true |
| `alpha` | 0.1 | 3.17 |

## 预热兼容性检查 (check_warm_start_compatibility)

以下参数变化时 `warm_start` 会抛出 `WarmStartIncompatibleError`:
- `operators`, `op_constraints`, `nested_constraints`
- `maxsize`, `maxdepth`
- `node_type`, `expression_type`
- `complexity_mapping`, `expression_options`
- `turbo`, `bumper`

因为这些参数改变后，已有的表达式树结构无法与新配置兼容。
