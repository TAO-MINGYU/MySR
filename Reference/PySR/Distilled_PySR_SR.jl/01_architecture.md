# 01 — PySR/SR.jl 整体架构

## 1. 系统层次

```
┌─────────────────────────────────────────────────────────┐
│  PySRRegressor (pysr/sr.py)                             │
│  scikit-learn estimator: fit/predict/sympy/latex/jax    │
├─────────────────────────────────────────────────────────┤
│  juliacall/PythonCall.jl 桥接层                         │
│  julia_import.py, julia_helpers.py, julia_extensions.py │
├─────────────────────────────────────────────────────────┤
│  SymbolicRegression.jl 主模块                           │
│  equation_search() — 搜索入口                          │
├──────────────┬──────────────────┬───────────────────────┤
│  CoreModule  │  Population      │  MLJInterface         │
│  选项、数据、 │  群体、成员、    │  MLJ 框架集成         │
│  运算符、约束 │  名人堂、迁移    │                       │
├──────────────┴──────────────────┴───────────────────────┤
│  DynamicExpressions.jl（表达式树、解析、化简）          │
│  LossFunctions.jl（损失函数族）                         │
│  Optim.jl（常数优化）                                   │
└─────────────────────────────────────────────────────────┘
```

## 2. 核心数据流

### 2.1 一次 fit() 调用的完整流程

```
fit(X, y)
  │
  ├─ _validate_and_modify_params()      # 参数校验、默认值填充
  ├─ _validate_and_set_fit_params()     # sklearn 数据校验
  ├─ _pre_transform_training_data()     # 特征选择 + GP 去噪
  │
  ├─ _run(X, y, ...)
  │   ├─ 构建 Julia 端参数：
  │   │   ├── OperatorEnum         (运算符枚举)
  │   │   ├── MutationWeights      (变异权重)
  │   │   ├── Options              (搜索选项, ~80个字段)
  │   │   └── Dataset              (数据容器)
  │   │
  │   ├─ 调用 Julia: equation_search(X, y, ...)
  │   │
  │   │   ├── _validate_options()          # 运算符测试、配置验证
  │   │   ├── _create_workers()            # 创建并行 workers/线程
  │   │   ├── _initialize_search!()        # 初始化种群+名人堂
  │   │   ├── _warmup_search!()            # 初始搜索循环 (cur_maxsize 从3增长)
  │   │   │
  │   │   ├── _main_search_loop!()         # 主搜索循环
  │   │   │   │
  │   │   │   ├── for each (out, pop) pair:
  │   │   │   │   ├── 检查 channel 是否有结果
  │   │   │   │   ├── 如有: 获取群体 → 更新名人堂 → 迁移
  │   │   │   │   ├── 发送群体到 worker:
  │   │   │   │   │   ├── s_r_cycle()          # ncycles 次进化
  │   │   │   │   │   │   ├── reg_evol_cycle() # 锦标赛+变异/交叉
  │   │   │   │   │   │   └── best_examples_seen (每个复杂度的最佳)
  │   │   │   │   │   └── optimize_and_simplify_population()
  │   │   │   │   │       ├── simplify_tree!() + combine_operators()
  │   │   │   │   │       └── optimize_constants()  (Optim.BFGS)
  │   │   │   │   └── 非阻塞: 继续处理下一个 pair
  │   │   │   │
  │   │   │   └── 检查早停: 损失阈值/超时/最大评估数/用户按q
  │   │   │
  │   │   ├── _tear_down!()                # 关闭 workers, channels
  │   │   └── _info_dump()                 # 输出结果, 保存CSV
  │   │
  │   └── 序列化 Julia 状态 (Options + SearchState → numpy arrays)
  │
  ├─ get_hof()                         # 读取 CSV → DataFrame
  ├─ calculate_scores()                # 计算 score 列
  └─ expression_spec_.create_exports() # 添加 sympy/jax/torch/latex 列
```

### 2.2 数据在 Python 和 Julia 之间的流向

```
Python                             Julia
──────                             ─────
X (numpy, shape=(n, d))     →     AbstractMatrix (features × samples)
y (numpy, shape=(n,))       →     AbstractVector
weights (numpy)             →     AbstractVector
operators (dict)            →     OperatorEnum
options (kwargs)            →     Options struct (序列化/反序列化)
guesses (strings/expressions) →   PopMember 向量
search_state (numpy bytes)  ←→    Julia 序列化状态
hof CSV (output directory)  ←     save_to_file()
equations DataFrame         ←     CSV 读取 + 后处理
```

## 3. 关键设计决策

### 3.1 正则化进化 (Regularized Evolution)
- **年龄替换**：新个体替换群体中最老的个体（而非最差的）
- **目的**：维护多样性，防止过早收敛
- **锦标赛**：从群体中随机抽取 `tournament_selection_n` 个个体，选最佳

### 3.2 岛屿模型 (Migration)
- 多个群体（`populations`）独立进化
- 定期按 `fraction_replaced` 比例从其他群体迁移个体
- 有利变异可以在群体间传播

### 3.3 自适应简约 (Adaptive Parsimony)
- 跟踪各复杂度级别的方程频率
- 过度表示的复杂度级别受到惩罚
- 促使搜索探索所有复杂度级别

### 3.4 动态大小调度 (cur_maxsize)
- 搜索初期限制较小的方程（maxsize从3开始）
- 逐渐增加到用户指定的 maxsize
- 避免过早生成过于复杂的方程

### 3.5 常数优化
- 每次迭代后用 Optim.BFGS 优化一部分个体的常数
- `optimizer_probability` 控制被优化个体的比例
- 支持多 restart 和不同初值点

### 3.6 批处理 (Batching)
- 大规模数据时可以只用部分数据评估
- 仅在最终 `finalize_costs()` 时用全量数据
- 大幅加速搜索

### 3.7 Python-Julia 序列化
- Julia 状态不直接 pickle，而是通过 Julia 的 `Serialization.serialize` 转为 numpy uint8 数组
- `PySRRegressor.warm_start=True` 时反序列化恢复，支持持续搜索
