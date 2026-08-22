# 02 — PySRRegressor API 和参数

## 类签名

```python
class PySRRegressor(
    MultiOutputMixin, RegressorMixin, BaseEstimator
):
    def __init__(
        self,
        # === 模型选择 ===
        model_selection="best",     # "accuracy" | "best" | "score"
        
        # === 搜索空间 ===
        *,
        binary_operators=None,      # 旧 API: ["+", "*", "-", "/"]
        unary_operators=None,       # 旧 API: ["sin", "cos", "exp"]
        operators=None,             # 新 API: {1: [...], 2: [...], ...}
        expression_spec=None,       # ExpressionSpec | TemplateExpressionSpec
        maxsize=30,                 # 最大方程复杂度
        maxdepth=None,              # 最大树深度
        
        # === 搜索规模 ===
        niterations=60,             # 总代数 (实际 cycles = ncycles_per_iteration × niterations)
        populations=31,             # 岛屿群体数
        population_size=27,         # 每个群体的个体数
        ncycles_per_iteration=380,  # 每次迭代的进化周期数
        
        # === 目标函数 ===
        elementwise_loss=None,      # 如 "L2DistLoss()"
        loss_function=None,         # 自定义 Julia 损失函数
        loss_function_expression=None,  # 自定义 Julia 损失(支持批处理)
        loss_scale="log",           # "log" | "linear"
        dimensional_constraint_penalty=1000,
        dimensionless_constants_only=False,
        
        # === 复杂度 ===
        parsimony=0.0,              # 简约压力系数
        constraints=None,           # 操作符约束: {"^": (3, -1)}
        nested_constraints=None,    # 嵌套约束: {"^": {"sin": 1}}
        complexity_of_operators=None,  # 操作符复杂度权重
        complexity_of_constants=1,     # 常数复杂度
        complexity_of_variables=1,     # 变量复杂度
        warmup_maxsize_by=0.0,      # 预热期 maxsize 缩减比例
        use_frequency=True,         # 频率简约 (接受阶段)
        use_frequency_in_tournament=True,  # 频率简约 (锦标赛)
        adaptive_parsimony_scaling=1040.0,  # 自适应简约强度
        should_simplify=True,       # 代数化简
        
        # === 变异 ===
        weight_mutate_constant=0.0353,
        weight_mutate_operator=3.63,
        weight_mutate_feature=0.1,
        weight_swap_operands=0.00608,
        weight_rotate_tree=1.42,
        weight_add_node=0.0771,
        weight_insert_node=2.44,
        weight_delete_node=0.369,
        weight_simplify=0.00148,
        weight_randomize=0.00695,
        weight_do_nothing=0.431,
        weight_optimize=0.0,
        weight_form_connection=0.5,
        weight_break_connection=0.1,
        crossover_probability=0.066,
        skip_mutation_failures=False,
        annealing=True,             # 模拟退火
        alpha=3.17,                 # 退火温度系数
        perturbation_factor=1.0,    # 常数突变因子
        probability_negate_constant=0.01,
        
        # === 锦标赛 ===
        tournament_selection_n=10,
        tournament_selection_p=0.5,  # 自适应锦标赛的几何分布参数
        
        # === 常数优化 ===
        optimizer_algorithm="BFGS",
        optimizer_probability=0.1,
        optimizer_nrestarts=2,
        optimizer_iterations=10,
        optimizer_f_calls_limit=50,
        should_optimize_constants=True,
        
        # === 迁移 ===
        fraction_replaced=0.08,
        fraction_replaced_hof=0.07,
        fraction_replaced_guesses=0.001,
        migration=True,
        hof_migration=True,
        topn=10,
        
        # === 性能 ===
        parallelism="multithreading",  # "serial" | "multithreading" | "multiprocessing"
        procs=None,
        cluster_manager=None,
        heap_size_hint_in_bytes=None,
        worker_timeout=60.0,
        worker_imports=None,
        batching=False,
        batch_size=None,
        precision=32,               # 浮点精度
        fast_cycle=False,
        turbo=False,                # LoopVectorization
        bumper=False,               # 自定义分配器
        autodiff_backend=None,      # "Zygote" | "Enzyme" | "Mooncake"
        
        # === 确定性 ===
        random_state=None,
        deterministic=False,
        warm_start=False,
        guesses=None,               # 种子方程
        
        # === 监控 ===
        verbosity=1,
        update_verbosity=None,
        print_precision=3,
        progress=True,
        logger_spec=None,           # TensorBoardLoggerSpec
        input_stream=sys.stdin,
        
        # === 环境 ===
        temp_equation_file=False,
        tempdir=None,
        delete_tempfiles=True,
        update=False,
        
        # === 导出 ===
        output_directory="result",
        run_id=None,
        output_jax_format=True,
        output_torch_format=True,
        extra_sympy_mappings=None,  # {"my_op": sympy_function}
        extra_torch_mappings=None,  # {julia_func: torch_func}
        extra_jax_mappings=None,    # {julia_func: "jax.func_name"}
    )
```

## 关键方法

### fit
```python
fit(X, y, *, Xresampled=None, weights=None, variable_names=None,
    complexity_of_variables=None, X_units=None, y_units=None, category=None)
```
- 自动处理 DataFrame 列名和 numpy 数组
- Xresampled: 用于 GP 去噪的重采样网格
- category: 类别标签 (ParametricExpression 需要)

### predict
```python
predict(X, index=None, *, category=None) → ndarray
```
- 使用最佳方程的 lambda_format 评估
- index: 指定使用哪个方程 (None=自动选择)

### 导出方法
```python
sympy(index=None) → sympy.Expr            # SymPy 表达式
latex(index=None, precision=3) → str      # LaTeX
jax(index=None) → {"callable": f, "parameters": array}
pytorch(index=None) → nn.Module
```

### 其他
```python
get_best(index=None) → pd.Series          # 获取最佳方程信息
refresh()                                  # 重新读取 CSV 并应用导出映射
from_file(run_directory) → PySRRegressor   # 从保存的目录加载 (类方法)
```

## 参数分组 (与 param_groupings.yml 对应)

### 创建搜索空间
- `binary_operators`, `unary_operators`, `operators`, `expression_spec`, `maxsize`, `maxdepth`

### 设定搜索规模
- `niterations`, `populations`, `population_size`, `ncycles_per_iteration`

### 目标函数
- `elementwise_loss`, `loss_function`, `loss_function_expression`, `loss_scale`, `model_selection`, `dimensional_constraint_penalty`, `dimensionless_constants_only`

### 复杂度控制
- `parsimony`, `constraints`, `nested_constraints`, `complexity_of_operators`, `complexity_of_constants`, `complexity_of_variables`, `warmup_maxsize_by`, `use_frequency`, `use_frequency_in_tournament`, `adaptive_parsimony_scaling`, `should_simplify`

### 变异
- 所有 `weight_*` 参数, `crossover_probability`, `annealing`, `alpha`, `perturbation_factor`, `probability_negate_constant`, `skip_mutation_failures`

### 锦标赛选择
- `tournament_selection_n`, `tournament_selection_p`

### 常数优化
- `optimizer_algorithm`, `optimizer_probability`, `optimizer_nrestarts`, `optimizer_iterations`, `optimizer_f_calls_limit`, `should_optimize_constants`

### 迁移
- `fraction_replaced`, `fraction_replaced_hof`, `fraction_replaced_guesses`, `migration`, `hof_migration`, `topn`

### 数据预处理
- `denoise`, `select_k_features` (通过 `run_feature_selection()` 在 fit 时处理)

### 早停标准
- `max_evals`, `timeout_in_seconds`, `early_stop_condition`

### 性能
- `parallelism`, `procs`, `cluster_manager`, `heap_size_hint_in_bytes`, `worker_timeout`, `worker_imports`, `batching`, `batch_size`, `precision`, `fast_cycle`, `turbo`, `bumper`, `autodiff_backend`

### 确定性
- `random_state`, `deterministic`, `warm_start`, `guesses`

### 监控
- `verbosity`, `update_verbosity`, `print_precision`, `progress`, `logger_spec`, `input_stream`

## 已弃用的参数名 (自动映射)

| 旧名 | 新名 |
|------|------|
| `fractionReplaced` | `fraction_replaced` |
| `npop` | `population_size` |
| `loss` | `elementwise_loss` |
| `full_objective` | `loss_function` |
| `binary_operators` / `unary_operators` | `operators` |
| `multithreading` | `parallelism` |
