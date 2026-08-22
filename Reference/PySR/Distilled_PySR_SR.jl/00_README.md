# PySR/SymbolicRegression.jl 参考文档

本目录包含对 PySR（Python 前端）和 SymbolicRegression.jl（Julia 后端）源码的完整蒸馏知识。

## 快速导航

| 文件 | 内容 | 适用场景 |
|------|------|----------|
| [01_architecture.md](01_architecture.md) | 整体架构、数据流、PySR↔SR.jl 连接点 | 理解系统全貌 |
| [02_api_parameters.md](02_api_parameters.md) | PySRRegressor 完整 API 参数和用法 | 调参、使用 API |
| [03_julia_bridge.md](03_julia_bridge.md) | Python→Julia 桥接：juliacall、序列化、包管理 | 排查连接问题 |
| [04_export_system.md](04_export_system.md) | 表达式导出：SymPy/NumPy/JAX/PyTorch/LaTeX | 模型导出 |
| [05_search_loop.md](05_search_loop.md) | equation_search 主循环：状态管理、并行调度、早停 | 理解搜索流程 |
| [06_evolution_mutation.md](06_evolution_mutation.md) | 正则化进化、变异算子、模拟退火、交叉 | 理解进化机制 |
| [07_expression_types.md](07_expression_types.md) | 表达式树类型：Node/Composable/Template/Parametric | 表达式表示 |
| [08_loss_evaluation.md](08_loss_evaluation.md) | 损失评估、代价计算、常数优化、自适应简约 | 评估系统 |
| [09_options.md](09_options.md) | 完整 Options 参数参考（Julia 后端） | 深入配置 |
| [10_file_index.md](10_file_index.md) | 文件级索引：每个源文件的内容和依赖 | 定位源码 |

## 代码库位置

- **PySR 前端**：`/home/taomingyu/Development/PySR/src/PySR-master/pysr/`
- **SR.jl 后端**：`/home/taomingyu/Development/SymbolicRegression/src/SymbolicRegression.jl-master/src/`

## 核心依赖关系

```
PySR (Python)                         SymbolicRegression.jl (Julia)
─────────────                         ────────────────────────────
PySRRegressor  ──juliacall/PythonCall──→  equation_search()
    │                                           │
    ├── validation                              ├── _create_workers
    ├── feature_selection                       ├── _initialize_search!
    ├── denoising                               ├── _warmup_search!
    ├── _run() ──→ Julia call ──→               ├── _main_search_loop!
    │                                              │  ├── s_r_cycle()
    │                                              │  │  ├── reg_evol_cycle()
    │                                              │  │  │  ├── next_generation()
    │                                              │  │  │  └── crossover_generation()
    │                                              │  │  └── optimize_and_simplify_population()
    │                                              │  └── migration
    │                                              └── _tear_down!
    ├── get_hof() ←── CSV files ←──              save_to_file()
    └── predict/export
```

## 技术栈

- **表达式树**：DynamicExpressions.jl（`Node{T}`, `Expression{T}`, `OperatorEnum`）
- **损失函数**：LossFunctions.jl（`SupervisedLoss`, 自定义损失）
- **自动微分**：DynamicDiff.jl, Enzyme.jl, Mooncake.jl, Zygote.jl
- **优化器**：Optim.jl（BFGS, NelderMead）
- **维度分析**：DynamicQuantities.jl
- **并行**：Julia `Threads.@spawn`, `Distributed.@spawnat`
- **Python 桥接**：juliacall/PythonCall.jl
